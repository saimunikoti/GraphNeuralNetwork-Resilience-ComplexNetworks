# -*- coding: utf-8 -*-
#
# Copyright 2017-2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sps

from .validation import require_dataframe_has_columns


class ExternalIdIndex:
    def __init__(self, *args):
        multiples = [np.unique(ids) for ids in args]
        unique = np.concat(multiples)

        if len(multiples) > 1:
            unique = np.unique(unique)
        unique.sort()

        self._index = pd.Index(unique)
        self._dtype = np.min_scalar_type(len(self._index))

    def to_internal(self, ids, smaller_type=True):
        internal_ids = self._index.get_indexer(ids)
        if smaller_type:
            return internal_ids.astype(self._dtype)
        return internal_ids

    def from_internal(self, internal_ids):
        return self._index[internal_ids]

    def columns_to_internal(self, df, columns):
        rewritten = {c: self.to_internal(df[c]) for c in columns}
        return df.assign(**rewritten)


class SingleTypeData:
    def __init__(self, shared, features):
        if not isinstance(features, (np.ndarray, sps.spmatrix)):
            raise TypeError(
                f"features: expected numpy or scipy array, found {type(features)}"
            )
        if not isinstance(shared, pd.DataFrame):
            raise TypeError(f"shared: expected pandas DataFrame, found {type(shared)}")

        if len(features.shape) != 2:
            raise ValueError(
                f"expected features to be 2 dimensional, found {len(features.shape)}"
            )

        rows, _columns = features.shape
        if len(shared) != rows:
            raise ValueError(
                f"expected one ID per feature row, found {len(shared)} IDs and {rows} feature rows"
            )

        self.shared = shared
        self.features = features


class NodeData:
    def __init__(self, features):
        # FIXME: what to do about missing IDs
        self._type_index = ExternalIdIndex(list(features.keys()))
        self._id_index = ExternalIdIndex(
            *(data.shared.index for _, data in features.items())
        )

        pass


class SingleTripleEdgeData:
    def __init__(self):
        pass


def bits_for_index(idx: pd.Index) -> int:
    return int(np.ceil(np.log2(len(idx))))


def dtype_for_bits(bits: int):
    if bits == 0:
        return None

    return np.min_scalar_type(2 ** bits)


def encode_edge_type_triple(node_data: NodeData, features):
    edge_types = list(features.keys())
    edge_type_index = ExternalIds(edge_types)
    all_edges = pd.concat([data.shared for _, data in features.items()])

    src = all_edges["source"]
    tgt = all_edges["target"]

    node_type_bits = bits_for_index(node_data._type_index)
    edge_type_bits = bits_for_index(edge_type_index)
    total_bits = node_type_bits * 2 + edge_type_bits

    dtype = dtype_for_bits(total_bits)

    src_types_int = node_data._type_index.get_indexer(src).astype(dtype)
    edge_types_int = (
        edge_type_index.to_internal(edge_types, False)
        .astype(dtype)
        .repeat([len(data.shared) for _, data in features.items()])
    )
    tgt_types_int = node_data._type_index.get_indexer(dst).astype(dtype)

    return (
        (edge_types_int << (2 * node_type_bits))
        | (src_types_int << node_type_bits)
        | (tgt_types_int)
    )


def _index(single_type_data, type_col):
    type_start_index = {}
    rows_so_far = 0
    type_dfs = []

    all_types = sorted(single_type_data.keys())
    type_sizes = []

    for type_name, type_data in single_type_data.items():
        type_start_index[type_name] = rows_so_far
        n = len(type_data.shared)
        rows_so_far += n

        type_sizes.append(n)
        type_dfs.append(type_data.shared)

    # there's typically a small number of types, so a categorical column will be great
    type_column = pd.Categorical(all_types).repeat(type_sizes)
    id_to_type = pd.concat(type_dfs).assign(**{type_col: type_column})

    idx = id_to_type.index
    if not idx.is_unique:
        # had some duplicated IDs, which is an error
        duplicated = idx[idx.duplicated()].unique()

        count = len(duplicated)
        assert count > 0
        # in a large graph, printing all duplicated IDs might be rather too many
        limit = 20

        rendered = ", ".join(x for x in duplicated[:limit])
        continuation = f", ... ({count - limit} more)" if count > limit else ""

        raise ValueError(
            f"expected IDs to appear once, found some that appeared more: {rendered}{continuation}"
        )

    return type_start_index, id_to_type


class ElementData:
    def __init__(self, features, type_col):
        if not isinstance(features, dict):
            raise TypeError(f"features: expected dict, found {type(features)}")

        for key, value in features.items():
            if not isinstance(value, SingleTypeData):
                raise TypeError(
                    f"features[{key!r}]: expected 'SingleTypeData', found {type(value)}"
                )
            if type_col in value.shared.columns:
                raise ValueError(
                    f"features[{key!r}]: expected no column called {type_col!r}, found existing column that would be overwritten"
                )

        self._features = {
            type_name: type_data.features for type_name, type_data in features.items()
        }
        self._type_start_indices, self._id_to_type = _index(features, type_col)
        self._type_col = type_col

    def __len__(self):
        return len(self._id_to_type)

    def __contains__(self, item):
        return item in self._id_to_type.index

    def ids(self):
        """
        Returns:
             All of the IDs of these elements.
        """
        return self._id_to_type.index

    def types(self):
        """
        Returns:
             All of the types of these elements.
        """
        return self._features.keys()

    def type(self, ids):
        """
        Return the types of the ID(s)

        Args:
            ids (Any or Iterable): a single ID of an element, or an iterable of IDs of eleeents

        Returns:
             A sequence of types, corresponding to each of the ID(s)
        """
        return self._id_to_type.loc[ids, self._type_col]

    def features(self, type_name, ids):
        """
        Return features for a set of IDs within a given type.

        Args:
            type_name (hashable): the name of the type for all of the IDs
            ids (iterable of IDs): a sequence of IDs of elements of type type_name

        Returns:
            A 2D numpy array, where the rows correspond to the ids
        """
        indices = self._id_to_type.index.get_indexer(ids)
        start = self._type_start_indices[type_name]
        indices -= start

        # FIXME: better error messages
        if (indices < 0).any():
            # ids were < start, e.g. from an earlier type, or unknown (-1)
            raise ValueError("unknown IDs")

        try:
            return self._features[type_name][indices, :]
        except IndexError:
            # some of the indices were too large (from a later type)
            raise ValueError("unknown IDs")

    def feature_sizes(self):
        """
        Returns:
             A dictionary of type_name to an integer representing the size of the features of
             that type.
        """
        return {
            type_name: type_features.shape[1]
            for type_name, type_features in self._features.items()
        }


class NodeDataX(ElementData):
    pass


class EdgeDataX(ElementData):
    def __init__(self, features, type_col, source_col, target_col, weight_col):
        super().__init__(features, type_col)

        columns = [type_col, source_col, target_col, weight_col]
        if len(set(columns)) != len(columns):
            raise ValueError(
                f"expected type_col ({type_col!r}), source_col ({source_col!r}), target_col ({target_col!r}), weight_col ({weight_col!r}) to be different"
            )

        for key, value in features.items():
            require_dataframe_has_columns(
                f"features[{key!r}].shared",
                value.shared,
                {source_col, target_col, weight_col},
            )

        self._target_col = target_col
        self._source_col = source_col
        self._weight_col = weight_col

        self._edges_in = pd.Index(self._id_to_type[target_col])
        self._edges_out = pd.Index(self._id_to_type[source_col])
        # return an empty dataframe in the same format as the grouped ones, for vertices that
        # have no edges in a particular direction
        self._no_edges_df = self._id_to_type.iloc[0:0, :]

    def _degree_single(self, previous, col):
        series = self._id_to_type.groupby(col).size()
        if previous is None:
            return series
        return previous.add(series, fill_value=0)

    def degrees(self, ins=True, outs=True):
        """
        Compute the degrees of every non-isolated node.

        Args:
            ins (bool): count the in-degree
            outs (bool): count the out-degree

        Returns:
            The in-, out- or total (summed) degree of all non-isolated nodes.
        """
        series = None
        if ins:
            series = self._degree_single(series, self._target_col)
        if outs:
            series = self._degree_single(series, self._source_col)

        if series is None:
            raise ValueError("expected at least one of `ins` and `outs` to be True")

        return series

    def all(self, triple):
        """
        Return all edges as a pandas DataFrame.

        Args:
            triple (bool): include the types as well as the source and target

        Returns:
            A pandas DataFrame containing columns for each source and target and (if triple) the
            type.
        """
        columns = [self._source_col, self._target_col]
        if triple:
            columns.append(self._type_col)
        return self._id_to_type[columns]

    def ins(self, target_id):
        """
        Return the incoming edges for the node represented by target_id.

        Args:
            target_id: the ID of the node

        Returns:
            A pandas DataFrame containing all the information the edges entering the node.
        """
        try:
            return self._id_to_type[self._edges_in.get_loc(target_id)]
        except KeyError:
            # This cannot tell the difference between no edges and a vertex not existing,
            # so it has to assume it's the former
            return self._no_edges_df

    def outs(self, source_id):
        """
        Return the outgoing edges for the node represented by source_id.

        Args:
            source_id: the ID of the node

        Returns:
            A pandas DataFrame containing all the information the edges leaving the node.
        """
        try:
            return self._id_to_type[self._edges_out.get_loc(source_id)]
        except KeyError:
            return self._no_edges_df
