# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 21:13:25 2019

@author: saimunikoti
"""

import networkx as nx
from src.models.Robustness_ISN import randomattack
import numpy as np
import matplotlib.pyplot as plt

# deifne graph 
G = nx.DiGraph()
G.add_weighted_edges_from([(7,1,0.2),(8,1,0.8),(5,2,1),(6,4,0.9),(9,4,0.1),(12,4,0.2),(13,4,0.8), (10,5,1),(1,6,1),(3,7,1)
                            ,(5,8,1),(2,9,1),(2,10,1),(5,11,1),(3,12,1),(5,13,1),(12,14,0.2),(13,14,0.8),(14,15,1)])

############################ multi component diorected graph
#Gnew = nx.DiGraph()
#Gnew.add_weighted_edges_from([(7,1,0.2),(8,1,0.8),(5,2,1),(6,4,0.9),(9,4,0.1),(12,4,0.2),(13,4,0.8), (10,5,1),(1,6,1),(3,7,1)
#                            ,(5,8,1),(2,9,1),(2,10,1),(5,11,1),(3,12,1),(5,13,1),(12,14,0.2),(13,14,0.8),(14,15,1),
#                            (16,17,1),(16,18,0.8)])

############################# plot
#pos = nx.circular_layout(G)
#nx.draw(G,pos ,with_labels=True, node_color='peachpuff')
#
#labels = nx.get_edge_attributes(G,'weight')
#nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
#

Gud = G.to_undirected()
############################### initialize q and crf for nodes
def get_initialattr(G):
    
    for (node, val) in G.degree(weight='weight'):
        G.nodes[node]['qo'] = val
        G.nodes[node]['crf'] = 1
    try:    
        for (node, val) in G.in_degree(weight='weight'):
            G.nodes[node]['indegwt'] = val
    except:
        pass

get_initialattr(G)
get_initialattr(Gud)
        
        
#%%############################### random node attack ########################
### new method  of full trajectory of percolation

randout = randomattack() 
Lcc, Ncc, Sf, Efr, Expcrn, Expcrlcc, Expsf, Expefr, Expn = randout.nodeatk_sim(G, 100) 

### #### old method upto disconnectdness
#Steps, Size = randout.node_attack_sim(G, 500)
#print('The expected size of nodes, whose random attack disconnect the graph is',Steps)
#print('The expected size of giant component at the point of disappearance is ',Size)

#%%########################### random full attack on link #####################
# new method of full trajectory of percolation

Lcc, Ncc, Sf, Efr,Expcrn, Expcrlcc, Expcrsf, Expcrefr, Expn = randout.linkatk_sim(G, 500,1) 

#### old method upto disconnectdness
#Steps, Size = randout.link_attack_sim(Gund, 500, dfact=1)

#%%############################## random partial attack on link ##############

dfactor = [1.0, 0.9, 0.8, 0.7 ,0.6]

Colour = ["limegreen","dodgerblue","coral","lightseagreen","burlywood"]

def plot_base(tempax, y, ylabel, Colour):
    tempax.plot(xind, y, marker='o', color=Colour)
    tempax.xaxis.set_tick_params(labelsize=16)
    tempax.yaxis.set_tick_params(labelsize=16)
            
    tempax.set_ylabel(ylabel, fontsize=20)
    tempax.grid(True)
    
fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))   
fig2, ax2 = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))  
 
for countdf,df in enumerate(dfactor):
    
    Lcc, Ncc, Sf, Efr,Expcrn, Expcrlcc, Expsf, Expefr, Expn = randout.linkatk_sim(G, 500, df) # dfact = weight degradation 
    
    xind = np.arange(0,len(Lcc))
            
    plot_base(ax1[0], Lcc, "LCC", Colour[countdf])
    
    plot_base(ax1[1], Ncc, "NCC", Colour[countdf])
    
    plot_base(ax2[0], Sf, "Service Factor", Colour[countdf])
    plot_base(ax2[1], Efr, "Edge flow robustness", Colour[countdf])
    
ax1[0].legend(("1","0.9","0.8","0.7","0.6"))    
ax1[1].legend(("1","0.9","0.8","0.7","0.6"))    
ax1[1].set_xlabel("Attack event", fontsize=20)
ax2[0].legend(("1","0.9","0.8","0.7","0.6"))    
ax2[1].legend(("1","0.9","0.8","0.7","0.6"))    
ax2[1].set_xlabel("Attack event", fontsize=20)

plt.tight_layout()

#    ax[1].set_xlabel("Attack event", fontsize=20)    

# old method
#Steps=[]
#Size = []
#dfactor = [0.1,0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9 ,1.0]
#
#for df in dfactor:
#    
#    steps, size = randout.link_attack_sim(Gund, 1000, dfact = df) # dfact = weight degradation 
#    Steps.append(steps)
#    Size.append(size)
#

#%%################################### Target node attack ############################

tarout = targetattack()
#### new method

Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr  = tarout.nodeatk(G,"out degree" ,"weight")

Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr  = tarout.nodeatk(G,"weight node betweenness" ,"weight")

#### old method 
#Steps, size = tarout.node_attack(G, nx.out_degree_centrality)

#print('The cardinality of TARGET attack nodes (degree centrality), which could disconnect the graph is',Steps)
#print("Giant component size at the time of disappearance is", size[-1])

#%%################################# Target full link attack #################
###### weighted edge betweenness

Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr = tarout.linkatk(G,nx.edge_betweenness_centrality,"weight",1)

#Steps, size = tarout.link_attack(Gund, nx.edge_betweenness_centrality, 1, "weight")

###### edge betweeness 
Lcc, Ncc, Sf, Efr,criticaln, criticallcc, crsf, crefr = tarout.linkatk(G,nx.edge_betweenness_centrality,None,1)

#Steps, size = tarout.link_attack(Gund, nx.edge_betweenness_centrality, 1, None)

#%%############################  Target partial link attack ##################
###### weighted edge betweenness
Lcc, Ncc, Sf, Efr,criticaln, criticallcc, crsf, crefr  = tarout.linkatk(G,nx.edge_betweenness_centrality,"weight",0.8)

#Steps, size = tarout.link_attack(Gund, nx.edge_betweenness_centrality, 0.8, "weight")

###### edge betweeness 
Lcc, Ncc, Sf, Efr, criticaln, criticallcc, crsf, crefr  = tarout.linkatk(G,nx.edge_betweenness_centrality,None,0.8)

#Steps, size = tarout.link_attack(Gund, nx.edge_betweenness_centrality, 0.9, None)

#%% plot of res metrics 
resout = new_resmetric()
resout.plot_resmetric(Lcc,Ncc, Sf, Efr)

###old method
#fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))
#
#def plot(temp,dfactor,y,ylabel, Color):
#    temp.plot(dfactor, y, marker='o', color=Color)
#    temp.xaxis.set_tick_params(labelsize=17)
#    temp.yaxis.set_tick_params(labelsize=17)
#    
#    temp.set_ylabel(ylabel, fontsize=20)
#    
#    temp.grid(True)
#    
#plot(ax[0], dfactor, Steps, "percolation threshold", "cornflowerblue")    
#plot(ax[1], dfactor, Size, "Size of LCC","coral")  
#  
#ax[1].set_xlabel("Fraction of rated service", fontsize=16)
#plt.tight_layout()
#%%### Target node attack with secondary level 

#### weighted out degree
#tarout = targetattack()
#Steps, size = tarout.node_secondaryattack(G, nx.out_degree_centrality, "weight")
#fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False,sharey=False,figsize=(8,6))
#plot(ax,np.arange(len(size)), size, "size of network", "cornflowerblue") 
##ax.set_title("variation of size with  weighted out degree attack", fontsize=16) 
#ax.set_xlabel("steps", fontsize=20)
#plt.tight_layout()
##### out degree
#Steps, size = tarout.node_secondaryattack(G, nx.out_degree_centrality, None)
#fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False,sharey=False,figsize=(8,6))
#plot(ax,np.arange(len(size)), size, "size of network", "coral") 
##ax.set_title("variation of size with  out degree attack", fontsize=16) 
#ax.set_xlabel("steps", fontsize=20)
#plt.tight_layout()


#%% efficiency of attack factor 






