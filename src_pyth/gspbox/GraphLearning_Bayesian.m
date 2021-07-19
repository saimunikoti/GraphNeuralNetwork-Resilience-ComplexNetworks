clc
clear all
gsp_start;
gsp_reset_seed(1);

%load 
adjacency = load("C:\Users\saimunikoti\Manifestation\centrality_learning\data\Bayesian\plc_5000_gobsnoise_bayesian_wadjacency.mat") 
W= adjacency.wadjacency;
A = adjacency.adjacency;

% plot graph
co = G0.coords;
co = co(1:1000,:);
G = gsp_graph(W, m);
figure; gsp_plot_graph(G);
figure(1); imagesc(W);

% data is asssumed to be smooth vary slowly 
X = load("C:\Users\saimunikoti\Manifestation\centrality_learning\data\Bayesian\plc_5000_egr_bayesian_weighted_supembeddings.mat"); % data of each node is 2 dim vector of coordinates
X = X.emb;
X = cast(X,'double');
X = X';
% First compute the matrix of all pairwise squared distances Z (square norm of distance)
Z = gsp_distanz(X).^2;

% keep edge connection same as original graph. Only estimate link weights 
params.edge_mask = logical(A); 
% we also set the flag fix_zeros to 1:
params.fix_zeros = 1;

%% tuning of alpha and beta 
alphalist = linspace(0.1, 1,10);
betalist = linspace(0.1, 1, 10);

rmse = ones(20,20)*10;
W  = cast(W, 'double');
Wnew = W~=0; 

for countalpha = 1:10
    
    for countbeta = 1:10
        
        [Wpred, info_2] = gsp_learn_graph_log_degrees(Z, alphalist(countalpha), betalist(countbeta), params); 

        Wpred = full(Wpred);

        % compare two graphs
        Wprednew = Wpred~=0;
        
        if isequal(Wnew,Wprednew)          
            % rmse between the 
            rmse(countalpha, countbeta) = sqrt(mean((W-Wpred).^2, 'all' ));
        end
    end
    
end

At = Wpred.';
m  = tril(true(size(At)));
vpred  = At(m).';
At = W.';
m  = tril(true(size(At)));
v  = At(m).';

plot(v(2000:2500));
hold on
plot(vpred(2000:2500));

%% Final parameters of graph learning. alpha=0.7, beta=1
[xind, yind] = find(rmse==min(min(rmse)));
alphatuned = alphalist(xind);
betatuned = betalist(yind);
[Wpred, info_2] = gsp_learn_graph_log_degrees(Z, alphatuned, betatuned, params);
filename = "C:\Users\saimunikoti\Manifestation\centrality_learning\data\Bayesian\WmleSupervised_5000plc.mat"
save(filename,"Wpred")

%% ============================== Extra code for other features ==========

% suppose we want 4 edges per node on average
k = 2;
theta = gsp_compute_graph_learning_theta(Z, k);
% then instead of playing with a and b, we k keep them equal to 1 and
% multiply Z by theta for graph learning:
t1 = tic;
[Wpred, info_1] = gsp_learn_graph_log_degrees(theta * Z, 1, 1, params);
t1 = toc(t1);
% clean edges close to zero
Wpred(Wpred<1e-4) = 0;

% clean less weighted edges
Wpred = Wpred.*1000;
Wpred(Wpred<=0.2) = 0;

% renormalize weights
Wpred = Wpred./1000;

filename = "C:\Users\saimunikoti\Manifestation\centrality_learning\data\Bayesian\WmleSupervised_5000plc.mat"
save(filename,"Wpred")
% indeed, the new adjacency matrix has around 2 non zeros per row!
figure; imagesc(W); title(['Average edges/node: ', num2str(nnz(W)/G.N)]);
% update weights and plot
G = gsp_update_weights(G, W);
figure; gsp_plot_graph(G);

% Suppose for example that we know beforehand that no connection could or
% should be formed before pairs of nodes with distance more than 0.02. We
% create a mask with the pattern of allowed edges and pass it to the
% learning algorithm in a parameters structure:
params.edge_mask = zero_diag(Z < 0.02);
% we also set the flag fix_zeros to 1:
params.fix_zeros = 1;
[W2, info_2] = gsp_learn_graph_log_degrees(theta * Z, 1, 1, params); % info save time
% clean edges close to zero
W2(W2<1e-4) = 0;
% indeed, the new adjacency matrix has around 4 non zeros per row!
figure; imagesc(W2); title(['Average edges/node: ', num2str(nnz(W2)/G.N)]);
% update weights and plot
G = gsp_update_weights(G, W2);
figure; gsp_plot_graph(G);
fprintf('Relative difference between two solutions: %.4f\n', norm(W-W2, 'fro')/norm(W, 'fro'));

%% normalize weights
Wprednew = Wpred;

Wprednorm = Wpred./( 1.4079 );


%% plot
plot(W(10,:))
hold on 
plot(Wpred(10,:)*1000)
close()

%% Actually this feature is the one that allows us to work with big data. 
% Suppose the graph has n nodes. Classical graph learning [Kalofolias 2016]
% costs O(n^2) computations. If the mask we give has O(n) edges, then the 
% computation drops to O(n) as well. The question is, how can we compute 
% efficiently a mask of O(n) allowed edges for general data, even if we 
% don't have prior knowledge? Note that computing the full matrix Z already
% costs O(n^2) and we want to avoid it! The solution is Approximate Nearest
% Neighbors (ANN) that computes an approximate sparse matrix Z with much less
% computations, roughly O(nlog(n)). This is the idea behind [Kalofolias, Perraudin 2017]
% We compute an approximate Nearest Neighbors graph (using the FLANN
% library through GSP-box)

params_NN.use_flann = 1;
% we ask for an approximate k-NN graph with roughly double number of edges.
% The +1 is because FLANN also counts each node as its own neighbor.
params_NN.k = 2 * k + 1;
clock_flann = tic;
[indx, indy, dist, ~, ~, ~, NN, Z_sorted] = gsp_nn_distanz(X, X, params_NN);
time_flann = toc(clock_flann);
fprintf('Time for FLANN: %.3f seconds\n', toc(clock_flann));
% gsp_nn_distanz gives distance matrix in a form ready to use with sparse:
Z_sp = sparse(indx, indy, dist.^2, n, n, params_NN.k * n * 2);
% symmetrize the distance matrix
Z_sp = gsp_symmetrize(Z_sp, 'full');
% get rid or first row that is zero (each node has 0 distance from itself)
Z_sorted = Z_sorted(:, 2:end).^2;   % first row is zero
% Note that FLANN returns Z already sorted, that further saves computation
% for computing the parameter theta.
Z_is_sorted = true;
theta = gsp_compute_graph_learning_theta(Z_sorted, k, 0, Z_is_sorted);
params.edge_mask = Z_sp > 0;
params.fix_zeros = 1;
[W3, info_3] = gsp_learn_graph_log_degrees(Z_sp * theta, 1, 1, params);
W3(W3<1e-4) = 0;
fprintf('Relative difference between two solutions: %.4f\n', norm(W-W2, 'fro')/norm(W, 'fro'));
% Note that the learned graph is sparser than the mask we gave as input.
figure; subplot(1, 2, 1); spy(W3); title('W_3'); subplot(1, 2, 2), spy(params.edge_mask); title('edge mask');

