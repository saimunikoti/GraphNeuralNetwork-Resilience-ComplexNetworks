function [e, cnt] = norm_op_est(S_op, St_op, size_x, tol)
%NORM_OP_EST Estimate operator norm of given operator
%   Usage:  e = norm_op_est(S_op, St_op, size_x)
%           e = norm_op_est(S_op, St_op, size_x, tol)
%           [e, cnt] = norm_op_est(...)
%
%   Input parameters:
%       S_op    : operator
%       St_op   : the adjoint operator of S
%       size_x  : the size of the input expected by S_op
%       tol     : tolerance used as stopping criterion (default: 1e-6)
%   Output parameters
%       e       : the norm estimation
%       cnt     : the number of iterations needed
%
%   This function uses the power method to estimate the operator norm of a
%   given linear operator. The operator norm is the 2-norm of the
%   corresponding linear matrix.
%
%   It is useful when we have a fast way to compute the operator without
%   actually forming the matrix. It is also useful for finding the
%   Lipschitz constant of differentiable functions.
%
%   Example:
% 
%       n = 50;
%       % create a sparse linear operator with known norm
%       [S, St] = sum_squareform(n);
%       K_op = @(w) S*w;
%       Kt_op = @(z) St*z;
%       norm_est_1 = norm_op_est(K_op, Kt_op, [n*(n-1)/2, 1]);
%       norm_est_2 = norm_op_est(Kt_op, K_op, [n, 1]);
%       norm_known = sqrt(2*(n-1));
%       display(abs(norm_known - norm_est_1));
%       display(abs(norm_known - norm_est_2));
%       
%   See also:  normest
%
%   Url: https://epfl-lts2.github.io/unlocbox-html/doc/utils/norm_op_est.html

% Copyright (C) 2012-2016 Nathanael Perraudin.
% This file is part of UNLOCBOX version 1.7.5
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

% Author: Vassilis Kalofolias
% Date: March 2016


if nargin < 4
    tol = 1e-6;
end

maxiter = 100;
cnt = 0;

% initialize randomly
x = randn(size_x);
normx = norm(x, 'fro');
x = x / normx;
e = sqrt(x);
e0 = 0;

while abs(e-e0) > tol*e
    e0 = e;
    x = St_op(S_op(x));
    normx = norm(x, 'fro');
    x = x / normx;
    e = sqrt(normx);
    cnt = cnt+1;
    if cnt > maxiter
        warning('failed to converge!');
        break;
    end
end


