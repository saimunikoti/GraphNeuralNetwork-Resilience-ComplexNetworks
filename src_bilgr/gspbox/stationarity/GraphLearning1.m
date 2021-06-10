gsp_reset_seed(1);
G = gsp_2dgrid(16);
n = G.N;
W0 = G.W; % weight matrix
G = gsp_graph(G.W, G.coords+.05*rand(size(G.coords)));
G = gsp_graph(G.W, G.coords);
figure; gsp_plot_graph(G);

% data is asssumed to be smooth vary slowly 
X = G.coords'; % data of each node is 2 dim vector of coordinates

% First compute the matrix of all pairwise distances Z (square norm of distance)
Z = gsp_distanz(X).^2;

% learn graph
a = 1;
b = 1;
W = gsp_learn_graph_log_degrees(Z, a, b);