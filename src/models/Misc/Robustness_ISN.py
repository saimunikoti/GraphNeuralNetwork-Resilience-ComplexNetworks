# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:21:56 2019

@author: saimunikoti
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
###### robusness of the network
class randomattack():
    def __init__(self):
        print("random attack class is invoked")
        from src.models.Robustness_ISN import new_resmetric
        self.resoutput = new_resmetric()
        
    def nodeatk_sim(self, graph_org, n):
        
        def get_nodeattack(graph_org):
            
            criticaln =0
            criticallcc=0
            sf2 = 0
            efr2 = 0
            ncc2flag = 0
            graph = graph_org.copy()

            tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc = [max(tempcc)] # largest connected component
            ncc = [len(tempcc)] # no. of connected comp.
            
            sf  = [self.resoutput.get_servicefactor(graph)] # service factor
            efr = [self.resoutput.get_edgerobustness(graph)] # edge flow robust
            
            while(len(graph.nodes)>2):
                nodeselected = np.random.choice(graph.nodes())
                recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected))
                recchildnodes.remove(nodeselected)
                
                ##### node removal at primary level
                graph.remove_node(nodeselected) 
                                 
                #### effect of input on output- transitive property
                
                for childnode in recchildnodes:
                    indeg = graph.in_degree(childnode, weight="weight")
                                            
                    try:
                        tempratio = indeg/graph.nodes[childnode]['indegwt']
                    except:
                        continue
                    
                    for (nodest, nodeed) in graph.edges(childnode):
                        graph[nodest][nodeed]['weight'] =  tempratio*(graph[nodest][nodeed]['weight'])
                   
                    ##### node removal at secondary level
#                    if indeg ==0:
#                        graph.remove_node(childnode)
                    
                #### collecting metrics   
                tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
                try:
                    lcc.append(max(tempcc))
                except :
                    break
                                    
                ncc.append(len(tempcc))
                                                          
                if len(graph.edges())>1:
                    tempsf = self.resoutput.get_servicefactor(graph)
                    tempefr = self.resoutput.get_edgerobustness(graph)
                    
                sf.append(tempsf)
                efr.append(tempefr)
                
                ####### critical values when graph gets disconnected into 2 components first time
                if len(tempcc)==2 and ncc2flag==0:
                    ncc2flag =1
                    criticaln = len(lcc)
                    criticallcc = max(tempcc)
                    sf2 = tempsf
                    efr2 = tempefr
                    
            return lcc,ncc,sf,efr, criticaln, criticallcc , sf2, efr2
        
        Lcc=[]
        Ncc=[]
        Sf= []
        Efr=[]
        Expcrn=[] # critical n for disconnected average over multiple times
        Expcrlcc=[] # critical size of LCC when  disconnected average over multiple times
        Expcrsf=[] # critical sf when disconnected, average over multiple times
        Expcrefr=[] # critica
        Expn=[]

        for countrun in range(n):
            
            lcc,ncc,sf,efr, criticaln, criticallcc,sf2,efr2 = get_nodeattack(graph_org)
            
            Lcc.append(lcc)
            Ncc.append(ncc)
            Sf.append(sf)
            Efr.append(efr)
            Expcrn.append(criticaln)
            Expcrlcc.append(criticallcc)
            Expcrsf.append(sf2)
            Expcrefr.append(efr2)
            Expn.append(len(ncc))
            
        pad = len(max(Lcc, key=len)) 
                
        Lcc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Lcc]), axis=0)
        Ncc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Ncc]), axis=0)
        Sf = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sf]), axis=0)
        Efr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efr]), axis=0)
        Expcrn = np.mean(np.array(Expcrn))
        Expcrlcc = np.mean(np.array(Expcrlcc))
        Expcrsf = np.mean(np.array(Expcrsf))
        Expcrefr = np.mean(np.array(Expcrefr))
        Expn = np.mean(Expn)
        
        return Lcc,Ncc,Sf,Efr,Expcrn, Expcrlcc, Expcrsf, Expcrefr, Expn
    
    # stop node attack simulation when graph get disconnected - critical case       
#    def node_attack_sim(self, graph_org, n):
#        
#        def attack_node(graph_org):
#            graph = graph_org.copy()
#            steps = 0
#    
#            while nx.is_connected(graph):
#                node = np.random.choice(graph.nodes())
#                graph.remove_node(node)
#                steps = steps + 1
#    
#            else:
#                largest_cc = max(nx.connected_components(graph), key=len)
#    
#                return steps, largest_cc
#            
#        Steps=[]
#        Size=[]
#        for countrun in range(n):
#            tempsteps, largest_cc = attack_node(graph_org)
#            Steps.append(tempsteps)
#            Size.append(len(largest_cc))
#        
#        ## mean number of nodes which should be attacked for disappearance of giant component
#        Steps = np.mean(np.array(Steps))
#        Size = np.mean(np.array(Size))
#        
#        #### color giant component in graph
           
#        color_map = []
#        for countedge,node in enumerate(graph_org):
#            if node in largest_cc:
#                color_map.append('sandybrown')
#    
#            else:
#                color_map.append('antiquewhite')
#        
#        pos = nx.circular_layout(graph_org)
#        nx.draw(graph_org, pos, node_color = color_map, with_labels = True)
#            
#        return Steps, Size
    
    ####### Link attack sim which captures the whole trajectory of percolation
    def linkatk_sim(self, graph_org, n, dfact):
        
        def get_linkattack(graph_org, dfact):
            graph = graph_org.copy()
            ncc2flag =0
            criticaln =0
            criticallcc=0
            sf2 = 0
            efr2 = 0  
            
            for u,v,w in graph.edges(data=True):
                graph[u][v]['newweight']=w['weight']
                
            tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc = [max(tempcc)] # largest connected component
            ncc = [len(tempcc)] # no. of connected comp.
            
            sf  = [self.resoutput.get_servicefactor(graph)] # service factor
            efr = [self.resoutput.get_edgerobustness(graph)] # edge flow robust                
            
            while(len(graph.edges)>1):
                edge_list= list(graph.edges())
                edge_selected = random.choice(edge_list)
                nodeinc = edge_selected[1]
                recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeinc))
                
                graph[edge_selected[0]][edge_selected[1]]['newweight'] = graph[edge_selected[0]][edge_selected[1]]['newweight']-0.1
                
                
                ##### comparing weights with service acceptable level
                if graph[edge_selected[0]][edge_selected[1]]['newweight']< dfact*(graph[edge_selected[0]][edge_selected[1]]['weight']):                    
                    graph.remove_edge(*edge_selected) 
                    
                    #### effect of input on output- transitive property
                    for childnode in recchildnodes:
                        indeg = graph.in_degree(childnode, weight="newweight")
                                                
                        try:
                            tempratio = indeg/graph.nodes[childnode]['indegwt']
                        except:
                            continue
                        
                        for (nodest, nodeed) in graph.edges(childnode):
                            graph[nodest][nodeed]['newweight'] =  tempratio*(graph[nodest][nodeed]['newweight'])
                       
                        ##### node removal at secondary level
#                        if indeg ==0:
#                            graph.remove_node(childnode)                                                    
                else:
                    #### effect of input on output- transitive property
                     for childnode in recchildnodes:
                        indeg = graph.in_degree(childnode, weight="newweight")
                                                
                        try:
                            tempratio = indeg/graph.nodes[childnode]['indegwt']
                        except:
                            continue
                        
                        for (nodest, nodeed) in graph.edges(childnode):
                            graph[nodest][nodeed]['newweight'] =  tempratio*(graph[nodest][nodeed]['newweight'])
                       
#                        ##### node removal at secondary level
#                        if indeg ==0:
#                            graph.remove_node(childnode)   
                        
                ## collecting metrics                
                tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
                lcc.append(max(tempcc))
                ncc.append(len(tempcc))
                          
                tempsf = self.resoutput.get_servicefactor(graph)
                sf.append(tempsf)
                
                if len(graph.edges())>1:
                    tempefr = self.resoutput.get_edgerobustness(graph)
            
                efr.append(tempefr)
                
                ####### critical values when graph disconnected into 2 components first time
                if len(tempcc)==2 and ncc2flag==0:
                    ncc2flag =1
                    criticaln = len(lcc)
                    criticallcc = max(tempcc)
                    sf2 = tempsf
                    efr2 = tempefr
                    
                          
            return lcc,ncc,sf,efr, criticaln, criticallcc , sf2, efr2          
        
        Lcc=[]
        Ncc=[]
        Sf=[]
        Efr=[]
        Expcrn=[]
        Expcrlcc=[]
        Expsf=[]
        Expefr=[]
        Expn=[]
        for countrun in range(n):
            lcc,ncc,sf,efr,criticaln,criticallcc,sf2,efr2 = get_linkattack(graph_org, dfact)
            
            Lcc.append(lcc)
            Ncc.append(ncc)
            Sf.append(sf)
            Efr.append(efr)
            Expcrn.append(criticaln)
            Expcrlcc.append(criticallcc)
            Expsf.append(sf2)
            Expefr.append(efr2)
            Expn.append(len(ncc))
            
        pad = len(max(Lcc, key=len)) 
        
        Lcc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Lcc]), axis=0)
        Ncc = np.mean(np.array([i + [0]*(pad-len(i)) for i in Ncc]), axis=0)
        Sf = np.mean(np.array([i + [0]*(pad-len(i)) for i in Sf]), axis=0)
        Efr = np.mean(np.array([i + [0]*(pad-len(i)) for i in Efr]), axis=0)
        Expcrn = np.mean(np.array(Expcrn))
        Expcrlcc = np.mean(np.array(Expcrlcc))
        Expsf = np.mean(np.array(Expsf))
        Expefr = np.mean(np.array(Expefr))
        Expn = np.mean(Expn)
        
        return Lcc,Ncc,Sf,Efr, Expcrn, Expcrlcc, Expsf, Expefr, Expn        


    ## link attack simulation stops when graph get disconnected critical -condition   
#    def link_attack_sim(self, graph_org, n, dfact):
#
#        def attack_link(graph_org,countfig, dfact):
#            graph = graph_org.copy()
#            steps = 0
#            edge_weight = list(graph.edges.data('weight'))
#    
#            for u,v,w in graph.edges(data=True):
#                graph[u][v]['newweight']=w['weight']
#    
#            while nx.is_connected(graph):
#                edge_list= list(graph.edges())
#                temp=np.random.randint(0,len(edge_list),1)[0]
#                edge_selected = (edge_list[temp])
#                #print("Edge selected for reducing weight",(edge_selected[0]))
#    
#                graph[edge_selected[0]][edge_selected[1]]['newweight'] = graph[edge_selected[0]][edge_selected[1]]['newweight']-0.1
#    
#                if graph[edge_selected[0]][edge_selected[1]]['newweight']<= dfact*(graph[edge_selected[0]][edge_selected[1]]['weight']):
#                    #print("edge is removed",edge_selected[0],edge_selected[1])
#                    graph.remove_edge(*edge_selected)
#                    steps = steps + 1
    
#            else:
#                largestcc = max(nx.connected_components(graph), key=len)
#                edgelist = [e for e in graph.edges]
#                if countfig == n-1:
#                    color_map = []
#                    for edgeiter in graph_org.edges:
#                        if edgeiter in edgelist:
#                            color_map.append('black')
#    
#                        else:
#                            color_map.append('bisque')
#    
#                    pos = nx.circular_layout(graph_org)
#                    plt.figure(1)
#                    nx.draw(graph_org, pos,  with_labels = True, node_color = 'peachpuff', edge_color=color_map)
#    
#                return steps, largestcc
#    
#        Steps=[]
#        Size=[]
#        for countrun in range(n):
#            tempsteps, largest_cc = attack_link(graph_org, countrun, dfact)
#            Steps.append(tempsteps)
#            Size.append(len(largest_cc))
#    
#        ## mean number of nodes which should be attacked for disappearance of giant component
#        Steps = np.mean(np.array(Steps))
#        Size = np.mean(np.array(Size))
    
#        return Steps, Size

class targetattack():
    def __init__(self):
        print("target class is invoked")
        from Robustness_ISN import new_resmetric 
        self.resoutput = new_resmetric()       
        
    ####### Node attack sim which captures the whole trajectory of percolation    
    def nodeatk(self, graph_org, centrality, wt):
        graph = graph_org.copy()
        ncc2flag =0
        criticaln =0
        criticallcc=0
        sf2 = 0
        efr2 = 0     
        
        ###### initilization
        tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
        lcc = [max(tempcc)] # largest connected component
        ncc = [len(tempcc)] # no. of connected comp.
        
        sf  = [self.resoutput.get_servicefactor(graph)] # service factor
        efr = [self.resoutput.get_edgerobustness(graph)] # edge flow robust
 
        while(len(graph.nodes)>2):
            
            if centrality =="out degree":
                nodeimpscore = dict(graph.out_degree(weight=wt))
            else:
                nodeimpscore = nx.betweenness_centrality(graph, weight=wt)
            
            nodeselected = max(nodeimpscore, key=nodeimpscore.get)
            recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeselected))
            recchildnodes.remove(nodeselected)
            
            ##### node removal at primary level
            graph.remove_node(nodeselected) 
                             
            #### effect of input on output- transitive property
            
            for childnode in recchildnodes:
                indeg = graph.in_degree(childnode, weight="weight")
                                        
                try:
                    tempratio = indeg/graph.nodes[childnode]['indegwt']
                except:
                    continue
                
                for (nodest, nodeed) in graph.edges(childnode):
                    graph[nodest][nodeed]['weight'] =  tempratio*(graph[nodest][nodeed]['weight'])
               
                ##### node removal at secondary level
#                if indeg ==0:
#                    graph.remove_node(childnode)

            ### collect metrics
            tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc.append(max(tempcc))
            ncc.append(len(tempcc))
                                  
            if len(graph.edges())>1:
                tempefr = self.resoutput.get_edgerobustness(graph)
                tempsf = self.resoutput.get_servicefactor(graph)
                
            sf.append(tempsf)    
            efr.append(tempefr)
 
            ####### critical values when graph gets disconnected into 2 components first time
            if len(tempcc)==2 and ncc2flag==0:
                ncc2flag =1
                criticaln = len(lcc)
                criticallcc = max(tempcc)
                sf2 = tempsf
                efr2 = tempefr
            
        return lcc,ncc,sf,efr ,criticaln, criticallcc, sf2, efr2 
            
#    def node_attack(self, graph, centrality_metric):
#        graph = graph.copy()
#        steps = 0
#        ranks = centrality_metric(graph)
#        nodes = sorted(graph.nodes(), key=lambda n: ranks[n], reverse=True) # sorts in descending 
#        graph = graph.to_undirected()
#        size=[]
#        
#        while nx.is_connected(graph):
#            largest_cc = max(nx.connected_components(graph), key=len)
#            size.append(len(largest_cc))
#            graph.remove_node(nodes[steps])
#            steps = steps + 1
#    
#        else:
#            largest_cc = max(nx.connected_components(graph), key=len)
#            size.append(len(largest_cc))
#            pos = nx.circular_layout(graph)
#            nx.draw(graph, pos, with_labels=True, node_color='lightgreen')
#            return steps, size
#        
#    def node_secondaryattack(self, graph, centrality_metric, wt):
#        graph = graph.copy()
#        steps = 0
#        size=[]
#                   
#        while(len(list(graph.nodes)) > 2) :
#            
#            nodeoutdeg = dict(graph.out_degree(weight=wt))
#            
##            nodes = sorted(graph.nodes(), key=lambda n: ranks[n], reverse=True) # sorts in descending 
#
#            size.append(len(list(graph.nodes)))
#            selected_node = max(nodeoutdeg, key=nodeoutdeg.get)
#            
#            nb = [nind for nind in graph.neighbors(selected_node)]
#            added_nodes = [ind for ind in nb if graph.in_degree(ind) ==1]
#            graph.remove_node(selected_node)
#            graph.remove_nodes_from(added_nodes)
#            steps = steps + 1
#            
#        else:
#
#            size.append(len(list(graph.nodes)))
#            pos = nx.circular_layout(graph)
#            nx.draw(graph, pos, with_labels=True, node_color='lightgreen')
#            return steps, size
        
    ####### Link attack sim which captures the whole trajectory of percolation
    def linkatk(self, graph_org, centrality,wt, dfact):
        graph = graph_org.copy()
        ncc2flag =0
        criticaln =0
        criticallcc=0
        sf2 = 0
        efr2 = 0    
        
        for u,v,w in graph.edges(data=True):
            graph[u][v]['newweight']=w['weight'] 
        
        ##### initialization
        tempcc= [len(xind) for xind in nx.weakly_connected_components(graph)]
        lcc = [max(tempcc)] # largest connected component
        ncc = [len(tempcc)] # no. of connected comp.
        
        sf  = [self.resoutput.get_servicefactor(graph)] # service factor
        efr = [self.resoutput.get_edgerobustness(graph)] # edge flow robust                
        
        while(len(graph.edges)>1):
            
            ranks = centrality(graph, weight = wt)
            edges_sorted = sorted(graph.edges(), key=lambda n: ranks[n], reverse=True) # sorts in descending
            
            edge_selected = edges_sorted[0]
            nodeinc = edge_selected[1]
            recchildnodes = list(nx.dfs_preorder_nodes(graph, nodeinc))
            
            graph[edge_selected[0]][edge_selected[1]]['newweight'] = graph[edge_selected[0]][edge_selected[1]]['newweight']-0.1
            
            ##### comparing weights with service acceptable level
            if graph[edge_selected[0]][edge_selected[1]]['newweight']< dfact*(graph[edge_selected[0]][edge_selected[1]]['weight']):                    
                graph.remove_edge(*edge_selected) 
                
                #### effect of input on output- transitive property
                for childnode in recchildnodes:
                    indeg = graph.in_degree(childnode, weight="newweight")
                                            
                    try:
                        tempratio = indeg/graph.nodes[childnode]['indegwt']
                    except:
                        continue
                    
                    for (nodest, nodeed) in graph.edges(childnode):
                        graph[nodest][nodeed]['newweight'] =  tempratio*(graph[nodest][nodeed]['newweight'])
                   
                    ##### node removal at secondary level
#                    if indeg ==0:
#                        graph.remove_node(childnode)                                                    
            else:
                #### effect of input on output- transitive property
                 for childnode in recchildnodes:
                    indeg = graph.in_degree(childnode, weight="newweight")
                                            
                    try:
                        tempratio = indeg/graph.nodes[childnode]['indegwt']
                    except:
                        continue
                    
                    for (nodest, nodeed) in graph.edges(childnode):
                        graph[nodest][nodeed]['newweight'] =  tempratio*(graph[nodest][nodeed]['newweight'])
                   
                    ##### node removal at secondary level
#                    if indeg ==0:
#                        graph.remove_node(childnode)   
                        
            ##### collecting metrics                                    
            tempcc = [len(xind) for xind in nx.weakly_connected_components(graph)]
            lcc.append(max(tempcc))
            ncc.append(len(tempcc))
                      
            tempsf = self.resoutput.get_servicefactor(graph)
            sf.append(tempsf)
            
            if len(graph.edges())>1:
                tempefr = self.resoutput.get_edgerobustness(graph)
        
            efr.append(tempefr)
            
            ####### critical values when graph disconnected into 2 components first time
            if len(tempcc)==2 and ncc2flag==0:
                ncc2flag =1
                criticaln = len(lcc)
                criticallcc = max(tempcc)
                sf2 = tempsf
                efr2 = tempefr 
                
        return lcc,ncc,sf,efr, criticaln, criticallcc, sf2, efr2        
        
                
#    def link_attack(self, graph_org, centrality_metric, dfact, wt):
#        graph = graph_org.copy()
#        steps = 0
#        size=[]
#           
#        ##### define new weight data to edge
#        for u,v,w in graph.edges(data=True):
#                graph[u][v]['newweight']=w['weight'] 
#                
#        while nx.is_connected(graph):
#            
#            ranks = centrality_metric(graph, weight = wt)
#            edges_sorted = sorted(graph.edges(), key=lambda n: ranks[n], reverse=True) # sorts in descending
#            
#            largest_cc = max(nx.connected_components(graph), key=len)
#            size.append(len(largest_cc))
#            edge_selected = edges_sorted[0]
#            
#            graph[edge_selected[0]][edge_selected[1]]['newweight'] = graph[edge_selected[0]][edge_selected[1]]['newweight']-0.1
#    
#            if graph[edge_selected[0]][edge_selected[1]]['newweight']<= dfact*(graph[edge_selected[0]][edge_selected[1]]['weight']):
#
#                graph.remove_edge(*edge_selected)
#                steps = steps + 1
#    
#        else:
#            largest_cc = max(nx.connected_components(graph), key=len)
#            size.append(len(largest_cc))
#            color_map = []
#            edgelist = [e for e in graph.edges]
#            for edgeiter in graph_org.edges:
#                if edgeiter in edgelist:
#                    color_map.append('black')
#    
#                else:
#                    color_map.append('bisque')
#    
#            pos = nx.circular_layout(graph_org)
#            plt.figure(1)
#            nx.draw(graph_org, pos,  with_labels = True, node_color = 'lightgreen', edge_color=color_map)
#    
#            return steps, size
    
class new_resmetric():
    
    def __init__(self):
        print("metric class is invoked")
                        
    def get_servicefactor(self, G): 
        sumq = 0
        
        for (node, val) in G.degree(weight='weight'):
            tempval = val/(G.nodes[node]['qo'])
            G.nodes[node]['qcurrent'] = tempval
            
        for countn in G.nodes():
            sumq = sumq + (G.nodes[countn]['qcurrent'])/(G.nodes[countn]['crf'])
        
        servicefac = sumq/(len(G.nodes))
        
        return servicefac
 
    def get_edgerobustness(self,G):
        
        N = len(G.edges)
        try:
            tempcomp = [xind for xind in nx.weakly_connected_components(G)]
        except:
            tempcomp = [xind for xind in nx.connected_components(G)]
        
        sumedgecount = 0
        for countcomp in range(len(tempcomp)):
                
            H = G.subgraph(tempcomp[countcomp])
            edgecount = H.number_of_edges()
            sumedgecount = sumedgecount + edgecount*(edgecount-1)
          
        edgerobustness = sumedgecount/(N*(N-1))
        
        return edgerobustness
    
    def plot_resmetric(self,lcc,ncc,sf,efr):
        
        xind = np.arange(0,len(lcc))
        
        def plot_base(tempax, y, ylabel, Colour):
            tempax.plot(xind, y, marker='o', color=Colour)
            tempax.xaxis.set_tick_params(labelsize=16)
            tempax.yaxis.set_tick_params(labelsize=16)
                    
            tempax.set_ylabel(ylabel, fontsize=20)
            tempax.grid(True)
            
        fig1, ax = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))
        plot_base(ax[0], lcc, "LCC","limegreen")
        plot_base(ax[1], ncc, "NCC","slateblue")
        ax[1].set_xlabel("Attack event", fontsize=20)
        
        fig2, ax = plt.subplots(nrows=2, ncols=1, sharex=False,sharey=False,figsize=(8,6))
        plot_base(ax[0], sf, "Service factor","coral")
        plot_base(ax[1], efr, "Edge flow robustness","dodgerblue")
        ax[1].set_xlabel("Attack event", fontsize=20)
        


        
            
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
