#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 10:02:47 2023

@author: longlee
"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from collections import defaultdict
import sys
import copy
import os


class HyperGraph():
    
    def __init__(self,G):
        self.G = G
        self.e_distribution = {}
        self.v_distribution = {}
        self.codegree = {}
        self.weight_distribution = {}
        self.kcore_distribution = {}
        self.average_edegree = 0
        self.average_vdegree = 0
        
    
    def count_degree_distributation(self):
        
        edges = [node for node in self.G.nodes if self.G.nodes[node]['bipartite'] == 0]
        
        nodes  = [node for node in self.G.nodes if self.G.nodes[node]['bipartite'] == 1]
        
        
        for edge in edges:
            degree = len(self.G[edge])
            if degree not in self.e_distribution.keys():
                self.e_distribution[degree] = 1
            else:
                self.e_distribution[degree] += 1
                
        for node in nodes:
            degree = len(self.G[node])
            if degree not in self.v_distribution:
                self.v_distribution[degree] = 1
            else:
                self.v_distribution[degree] += 1
        return self.e_distribution, self.v_distribution
    
    
    def average_degree(self):
         
        num_e = 0
        for key,value in self.e_distribution.items():
            self.average_edegree += key * value
            num_e += value
        self.average_edegree = self.average_edegree/num_e
        
        
        num_v = 0
        for key,value in self.v_distribution.items():
            self.average_vdegree += key * value
            num_v += value
        self.average_vdegree = self.average_vdegree/num_v
        
        return self.average_edegree, self.average_vdegree
    
    
    def count_codegree_distribute(self):
        left_nodes = {node for node, attr in self.G.nodes(data=True) if attr['bipartite'] == 0}
        right_nodes = set(self.G) - left_nodes

        Y = list(right_nodes)#点
        
        for i in range(len(Y)-1):
            node1 = Y[i]
            for j in range(i+1,len(Y)):
                node2 = Y[j]
                com_nei = list(set(self.G[node1]).intersection(set(self.G[node2])))
                
                tmp = len(com_nei)
                
                if tmp in self.codegree.keys():
                    
                    self.codegree[tmp] += 1
                    
                else:
                    self.codegree[tmp] = 1
                    
        return self.codegree
    
    
    def weight_distribytation(self):
        
        for edge in self.G.edges():
            e = edge[0]
            v = edge[1]
            w = 1/len(self.G[e])+1/(len(self.G[v]))
            
            if w in self.weight_distribution.keys():
               self.weight_distribution[w] += 1
            else:
                self.weight_distribution[w] = 1
        return
    
    
    def graph_nodes(self):
        
        V_nodes = [node for node, attr in self.G.nodes(data=True) if attr['bipartite'] == 1] #V_nodes
        
        E_nodes = [node for node, attr in self.G.nodes(data=True) if attr['bipartite'] == 0] #E_nodes
        
        E = [] #Ini_E The edges contain nodes
        for e in E_nodes:
            tmp = []
            for nei in self.G[e]:
                tmp.append(nei)
                
            E.append(tmp)
            
        elist = {} #Ini_elist
        for node in V_nodes:
            tmp = []
            for nei in self.G[node]:
                tmp.append(nei)
            elist[node] = tmp
        
        return V_nodes, E_nodes, E, elist
    

    def find_six_cycles(self):
        left_nodes = {node for node, attr in self.G.nodes(data=True) if attr['bipartite'] == 0}
        right_nodes = {node for node, attr in self.G.nodes(data=True) if attr['bipartite'] == 1}
        six_cycles = []
        
        for right_node in right_nodes:
            visited = set()
            stack = [(right_node, [right_node])]
            
            while stack:
                current_node, path = stack.pop()
                
                if current_node in visited:
                    continue
                
                visited.add(current_node)
                
                if len(path) >= 6 and path[0] in self.G[current_node]:
                    if set(path) not in six_cycles:

                        six_cycles.append(set(path))
                
                if len(path) < 6:
                    for neighbor in self.G[current_node]:
                        stack.append((neighbor, path + [neighbor]))
        
        infected_cycle = set()
        for node in six_cycles[0]:
            if node in right_nodes:
                infected_cycle.add(node)
        '''
        for cycle in six_cycles:
            for node in cycle:
                if node in right_nodes:
                    infected_cycle.add(node)
        '''
        return len(six_cycles), infected_cycle

    
    def Draw_degree(self):
        # 统计度分布
        nodes  = [node for node in self.G.nodes if self.G.nodes[node]['bipartite'] == 1]
        #edges = [node for node in Graph.nodes if Graph.nodes[node]['bipartite'] == 0]
        v_degree_sequence = [self.G.degree(node) for node in nodes]
        degree_counts = dict()
        for degree in v_degree_sequence:
            if degree in degree_counts:
                degree_counts[degree] += 1
            else:
                degree_counts[degree] = 1
        
        # 绘制度分布直方图
        degrees = list(degree_counts.keys())
        counts = list(degree_counts.values())
        plt.bar(degrees, counts)
        plt.xlabel('Degree')
        plt.ylabel('Count')
        plt.title('Degree Distribution')
        plt.show()
        return
    
    
    def Draw_codegree(self):
        
        Distribution = self.v_distribution
        S = sum(Distribution.values())
        res = 0
        for key, value in Distribution.items():
            res = key * value/S
        print('the expaction of codegree is:',res)
        
        '''
        # 绘制度分布直方图
        degrees = list(Distribution.keys())
        counts = list(Distribution.values())
        plt.bar(degrees, counts)
        plt.xlabel('Co_Degree')
        plt.ylabel('Count')
        plt.title('Degree Distribution')
        plt.show()
        '''
        return res

        
Graph = nx.Graph()

df = open('./madison_hyperedges.txt','r')

e = 0
num_e = 601
while True:
    
    line = df.readline()
    tmp = line.strip().split(' ')#把分隔号抹去
    if line:
        Graph.add_node(e, bipartite=0)  # 0表示属于左侧

        for node in tmp:
            v = int(node)+num_e
            if v not in Graph.nodes():
                
                Graph.add_node(v, bipartite=1)
                
            Graph.add_edge(e, v)
        e = e + 1
    if not line:
        break
df.close()
edges = left_nodes = {node for node, attr in Graph.nodes(data=True) if attr['bipartite'] == 0}
m_edges = len(edges)        
nodes = right_nodes = {node for node, attr in Graph.nodes(data=True) if attr['bipartite'] == 1}
n_nodes = len(nodes)

'''
e_list = [[1, 0], [1, 2], [2, 0], [2, 1], [3, 0], [3, 1], [4, 1], [5, 2]]
for edge in e_list:
    v = edge[0]
    e = edge[1]
    Graph.add_node(e, bipartite=0)
    Graph.add_node(v, bipartite=1)
    Graph.add_edge(e, v)

'''

Ini_Graph = HyperGraph(Graph)



def add_node_to_hyperedge(E1, elist1, v, e_i):
    # Add node v to hyperedge E[e_i]

    if v not in elist1:
        print("Error: Given node is not found.")
        sys.exit()
    if e_i < 0 or len(E1) <= e_i :
        print("Error: Given hyperedge is not found.")
        sys.exit()

    E1[e_i].append(v)
    elist1[v].append(e_i)

    return E1, elist1

def remove_node_from_hyperedge(E1, elist1, v, e_i):
    # Remove node v from hyperedge E[e_i]

    if v not in elist1:
        print("Error: Given node is not found.")
        sys.exit()
    if e_i  < 0 or len(E1) <= e_i :
        print("Error: Given hyperedge is not found.")
        sys.exit()

    if e_i not in elist1[v]:
        print("Error: Given node is not included in the given hyperedge.")
        sys.exit()
    elist1[v].remove(e_i)

    if v not in E1[e_i]:
        print(v)
        print("Error: Given node does not belong to the given hyperedge.")
        sys.exit()
    E1[e_i].remove(v)

    return E1, elist1


def targeting_rewiring_d_v_two(IniG, GenG):
    print("Started targeting-rewiring process with d_v = 2.")
    
    Ini_V = IniG.graph_nodes()[0]
    left_nodes = IniG.graph_nodes()[1]
    Ini_E = IniG.graph_nodes()[2]  
    Ini_elist = IniG.graph_nodes()[3]
    
    
    Gen_V = GenG.graph_nodes()[0]
    Gen_E = GenG.graph_nodes()[2]
    Gen_elist = GenG.graph_nodes()[3]
    

    node_degrees = set()
    for v in Gen_V:

        k = int(len(Gen_elist[v]))
        node_degrees.add(k)


    target_num_jnt_node_deg = {k1: {k2: 0 for k2 in node_degrees} for k1 in node_degrees}

    #print(target_num_jnt_node_deg)

    for e in Ini_E:
        s = int(len(e))
        for i in range(0, s-1):
            u = e[i]
            k1 = int(len(Ini_elist[u]))
            for j in range(i+1, s):
                v = e[j]
                k2 = int(len(Ini_elist[v]))
                target_num_jnt_node_deg[k1][k2] += 1
                target_num_jnt_node_deg[k2][k1] += 1

    #print(target_num_jnt_node_deg)
    
    #所有边中的节点对
    target_sum_num_jnt_node_deg = 0
    
    for e in Ini_E:
        s = len(e)
        target_sum_num_jnt_node_deg += s*(s-1)

    node_degrees1 = set()
    for v in Gen_V:
        k = int(len(Gen_elist[v]))
        node_degrees1.add(k)

    #print(node_degrees1)

    num_jnt_node_deg = {k1: {k2: 0 for k2 in node_degrees1} for k1 in node_degrees1}

    #print(num_jnt_node_deg)

    for e in Gen_E:
        s = int(len(e))
        for i in range(0, s-1):
            u = e[i]
            k1 = int(len(Gen_elist[u]))
            for j in range(i+1, s):
                v = e[j]
                k2 = int(len(Gen_elist[v]))
                num_jnt_node_deg[k1][k2] += 1
                num_jnt_node_deg[k2][k1] += 1

    sum_num_jnt_node_deg = 0
    for e in Gen_E:
        s = len(e)
        sum_num_jnt_node_deg += s*(s-1)
        #print(sum_num_jnt_node_deg)
    #print(num_jnt_node_deg)
    dist = 0
    norm = 0

    for k1 in node_degrees:
        for k2 in node_degrees:
            dist += math.fabs(float(target_num_jnt_node_deg[k1][k2])/target_sum_num_jnt_node_deg - float(num_jnt_node_deg[k1][k2])/sum_num_jnt_node_deg)
            norm += float(target_num_jnt_node_deg[k1][k2])/target_sum_num_jnt_node_deg

    print("Initial L1 distance between the target and present P_v(k, k'): ", float(dist)/norm)

    bipartite_edges = []
    for v in Gen_V:
        for e_i in Gen_elist[v]:
            bipartite_edges.append([v, e_i])
    
    randG_node_degree = {}
    for v in Gen_V:
        randG_node_degree[v] = int(len(Gen_elist[v]))

    B_M = len(bipartite_edges)
    
    R = 10*B_M

    for r in range(0, R):

        e1 = random.randrange(0, B_M)
        e2 = random.randrange(0, B_M)
        [v1, e_i] = bipartite_edges[e1]
        [v2, e_j] = bipartite_edges[e2]
        while(v1 == v2 or e_i == e_j):
            e1 = random.randrange(0, B_M)
            e2 = random.randrange(0, B_M)
            [v1, e_i] = bipartite_edges[e1]
            [v2, e_j] = bipartite_edges[e2]

        k1 = randG_node_degree[v1]
        k2 = randG_node_degree[v2]
        
        num_jnt_node_deg_to_add = defaultdict(lambda: defaultdict(int))
            
        for w in Gen_E[e_i]:
            l = randG_node_degree[w]
            num_jnt_node_deg_to_add[k1][l] -= 1
            num_jnt_node_deg_to_add[l][k1] -= 1
        num_jnt_node_deg_to_add[k1][k1] += 2
    
        for w in Gen_E[e_j]:
            l = randG_node_degree[w]
            num_jnt_node_deg_to_add[k1][l] += 1
            num_jnt_node_deg_to_add[l][k1] += 1
    
        for w in Gen_E[e_j]:
            l = randG_node_degree[w]
            num_jnt_node_deg_to_add[k2][l] -= 1
            num_jnt_node_deg_to_add[l][k2] -= 1
        num_jnt_node_deg_to_add[k2][k2] += 2
        num_jnt_node_deg_to_add[k1][k2] -= 1
        num_jnt_node_deg_to_add[k2][k1] -= 1
    
        for w in Gen_E[e_i]:
            l = randG_node_degree[w]
            num_jnt_node_deg_to_add[k2][l] += 1
            num_jnt_node_deg_to_add[l][k2] += 1
        num_jnt_node_deg_to_add[k1][k2] -= 1
        num_jnt_node_deg_to_add[k2][k1] -= 1
    
        rewired_dist = dist
        for k in num_jnt_node_deg_to_add:
            for l in num_jnt_node_deg_to_add:
                x = math.fabs(float(target_num_jnt_node_deg[k][l])/target_sum_num_jnt_node_deg - float(num_jnt_node_deg[k][l] + num_jnt_node_deg_to_add[k][l])/sum_num_jnt_node_deg)
                y = math.fabs(float(target_num_jnt_node_deg[k][l])/target_sum_num_jnt_node_deg - float(num_jnt_node_deg[k][l])/sum_num_jnt_node_deg)
                rewired_dist += x - y
    
        delta_dist = rewired_dist - dist
        
        if delta_dist >= 0:
            continue

        bipartite_edges[e1] = [v1, e_j]
        bipartite_edges[e2] = [v2, e_i]
        #print('First')
        Gen_E, Gen_elist = remove_node_from_hyperedge(Gen_E, Gen_elist, v1, e_i)
        #print('Secode')
        Gen_E, Gen_elist = add_node_to_hyperedge(Gen_E, Gen_elist, v1, e_j)
        #print('Third')
        Gen_E, Gen_elist = remove_node_from_hyperedge(Gen_E, Gen_elist, v2, e_j)
        #print('Fourth')
        Gen_E, Gen_elist = add_node_to_hyperedge(Gen_E, Gen_elist, v2, e_i)
    
        for k in num_jnt_node_deg_to_add:
            for l in num_jnt_node_deg_to_add:
                num_jnt_node_deg[k][l] += num_jnt_node_deg_to_add[k][l]
    
        dist = rewired_dist
    
    print("Final L1 distance between target and current P_v(k, k'): ", float(dist)/norm, "\n")

    #print(Gen_elist)    
    Gen_G = nx.Graph()
    for key, value in Gen_elist.items():
        Gen_G.add_node(key, bipartite=1)
        for e in value:
            Gen_G.add_node(e, bipartite= 0)
            Gen_G.add_edge(key, e)
    Gen_G = HyperGraph(Gen_G)
    return Gen_G


class generated_graph():
    def __init__(self,G):
        self.G = HyperGraph(G)
        self.e_distribution, self.v_distribution = self.G.count_degree_distributation()
        
        self.m = sum(self.e_distribution.values())    
        self.n = sum(self.v_distribution.values())
        
        self.e_average_degree, self.v_average_degree = self.G.average_degree()
        
        # 创建一个空的二部图
        self.bipartite_graph = nx.Graph()
        
        
    def randomizing_d_v_zero_d_e_zero(self):
        
        # Given a hypergraph, return a randomized hypergraph with (d_v, d_e) = (0, 0)，keep average degree

        # 添加左侧的节点，并设置度分布，即超边集合
        e_nodes_list = range(0, self.m)

        for node in e_nodes_list:
            self.bipartite_graph.add_node(node, bipartite=0)  # 0表示属于左侧
            self.bipartite_graph.nodes[node]['degree'] = self.e_average_degree

        # 添加右侧的节点，并设置度分布，即节点集合
        v_nodes_list = range(self.m, self.m+self.n)

        for node in v_nodes_list:
            self.bipartite_graph.add_node(node, bipartite=1)  # 1表示属于右侧
            self.bipartite_graph.nodes[node]['degree'] = self.v_average_degree
            
        
        for left_node in e_nodes_list:
            # 获取左侧节点的度
            left_degree = self.bipartite_graph.nodes[left_node]['degree']
            
            # 随机选择右侧节点连接
            tmp_right = [] #度满足连接条件的右侧节点
            for node in v_nodes_list:
                if self.bipartite_graph.nodes[node]['degree'] > 0:
                    
                    tmp_right.append(node)
                    
            if len(tmp_right) >= left_degree:
                
                tmp_right.sort(reverse=True)
                right_nodes_selected = tmp_right[:left_degree]
                
            else:
                return self.randomizing_d_v_zero_d_e_zero()
            #right_nodes_selected = np.random.choice(tmp_right, left_degree, replace=False)
            
            # 添加边
            for right_node in right_nodes_selected:
                self.bipartite_graph.add_edge(left_node, right_node)
                self.bipartite_graph.nodes[right_node]['degree'] -= 1
        print("Successfully generated a randomized hypergraph with (d_v, d_e) = (0, 0).\n")
        return self.bipartite_graph
    
        
    def randomizing_d_v_one_d_e_one(self):
    	# Given a hypergraph, return a randomized hypergraph with (d_v, d_e) = (0, 0)，keep degree distribution
        
        # 添加左侧的节点，并设置度分布，即超边集合
        e_nodes_list = range(0, self.m)
        
        # print(len(e_nodes_list))
        
        e_degrees = []
        for key, value in self.e_distribution.items():
            for i in range(value):
                e_degrees.append(key)
        
        
        e_degrees.sort(reverse=True)
    
        for node, degree in zip(e_nodes_list, e_degrees):
            self.bipartite_graph.add_node(node, bipartite=0)  # 0表示属于左侧
            self.bipartite_graph.nodes[node]['degree'] = degree
    
        # 添加右侧的节点，并设置度分布，即节点集合
        v_nodes_list = range(self.m, self.m + self.n)
        
        #print(len(v_nodes_list))
        v_degrees = []
        for key, value in self.v_distribution.items():
            for i in range(value):
                v_degrees.append(key)
        
        v_degrees.sort()
        
        for node, degree in zip(v_nodes_list, v_degrees):
            self.bipartite_graph.add_node(node, bipartite=1)  # 1表示属于右侧
            self.bipartite_graph.nodes[node]['degree'] = degree
            
        
        for left_node in e_nodes_list:
            # 获取左侧节点的度
            left_degree = self.bipartite_graph.nodes[left_node]['degree']
            
            # 随机选择右侧节点连接
            tmp_right = [] #度满足连接条件的右侧节点
            for node in v_nodes_list:
                if self.bipartite_graph.nodes[node]['degree'] > 0:
                    
                    tmp_right.append(node)
                    
            if len(tmp_right) >= left_degree:
                
                tmp_right.sort(reverse=True)
                right_nodes_selected = tmp_right[:left_degree]
                
            else:
                return self.randomizing_d_v_one_d_e_one()
                
            #right_nodes_selected = np.random.choice(tmp_right, left_degree, replace=False)
            
            # 添加边
            for right_node in right_nodes_selected:
                self.bipartite_graph.add_edge(left_node, right_node)
                self.bipartite_graph.nodes[right_node]['degree'] -= 1
                
        print("Successfully generated a randomized hypergraph with (d_v, d_e) = (1, 1).\n")
        nodes  = [node for node in self.bipartite_graph.nodes if self.bipartite_graph.nodes[node]['bipartite'] == 1]
        
        edges = [node for node in self.bipartite_graph.nodes if self.bipartite_graph.nodes[node]['bipartite'] == 0]
        
        self.bipartite_graph = HyperGraph(self.bipartite_graph)
        return self.bipartite_graph
    
    
    def randomizing_d_v_two_d_e_zero(self):
        # Given a hypergraph, return a randomized hypergraph with (d_v, d_e) = (2, 0).
        self.bipartite_graph = self.randomizing_d_v_one_d_e_zero()

        #self.bipartite_graph = HyperGraph(self.bipartite_graph)
        
        self.bipartite_graph = targeting_rewiring_d_v_two(self.G, self.bipartite_graph)    	
    	
        print("Successfully generated a randomized hypergraph with (d_v, d_e) = (2, 0).\n")
    	
        return self.bipartite_graph


    def randomizing_d_v_two_d_e_one(self):
        # Given a hypergraph, return a randomized hypergraph with (d_v, d_e) = (2, 1).
        self.bipartite_graph = self.randomizing_d_v_one_d_e_one()
        #self.bipartite_graph = HyperGraph(self.bipartite_graph)
        self.bipartite_graph = targeting_rewiring_d_v_two(self.G, self.bipartite_graph)
        #self.bipartite_graph = HyperGraph(self.bipartite_graph)

        print("Successfully generated a randomized hypergraph with (d_v, d_e) = (2, 1).\n")
        
        return self.bipartite_graph


def jaccard_similarity(dict1, dict2):
    
    pair = 0
    for key, value in dict1.items():
        pair += value
    
    res = 0
        
    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    intersection = set1.intersection(set2)
    
    for key in intersection:
        value1 = dict1[key]
        value2 = dict2[key]
        res += min(value1,value2)
        
    similarity = res / pair
    
    return similarity

def expaction_codegree(Distribution):
        
    
    S = sum(Distribution.values())
    res = 0
    for key, value in Distribution.items():
        res = key * value/S
    return res

Gen_G = generated_graph(Graph).randomizing_d_v_two_d_e_one()

G_codegree = Ini_Graph.count_codegree_distribute()
G_expaction_codegree = expaction_codegree(G_codegree)
G_four_cycle = sum(G_codegree.values())-G_codegree[0]-G_codegree[1]

Gen_codegree = Gen_G.count_codegree_distribute()
Gen_expaction_codegree = expaction_codegree(Gen_codegree)
Gen_four_cycle = sum(Gen_codegree.values()) - Gen_codegree[0] - Gen_codegree[1]

print('the number of 4_cycle and expaction codegree in IniG is {}|{}, and in GenG is {}|{}.'.format(G_four_cycle, G_expaction_codegree, Gen_four_cycle, Gen_expaction_codegree))

s = jaccard_similarity(G_codegree,Gen_codegree)
print('The similarity of IniG and GenG is : ', s)

G_six_cycle_num, G_infected_nodes = Ini_Graph.find_six_cycles()

Gen_six_cycle_num, Gen_infected_nodes = Gen_G.find_six_cycles()
print('the number of 6_cycle and nodes in IniG is {}| {} , and in GenG is {}| {}.'.format(G_six_cycle_num, len(G_infected_nodes), Gen_six_cycle_num, len(Gen_infected_nodes)))



##传播
def initialize_infected(right_nodes, initial_infect):
    infected = set(random.sample(right_nodes, initial_infect))
    return infected

def simulate_si_model(graph, infected):
    Ini_V = graph.graph_nodes()[0]
    Ini_E = graph.graph_nodes()[2]
    
    infected_time = dict()  # 存储每个感染节点被感染的时间步长
    Adj = {} #统计每个节点与那些节点在一条边中
    for node in Ini_V:
        Adj[node] = set()
        
    for edge in Ini_E:
        tmp = copy.deepcopy(edge)
        for i in range(len(edge)):
            node = edge[i]
            tmq = set(tmp)
            tmq.remove(node)
            if node in Adj.keys():  
                Adj[node] = Adj[node].union(tmq) 
            else:
                Adj[node] = tmq
        
    
    time_step = 0
    while True:
        new_infected = set()
        for node in infected:
            neighbors = list(Adj[node])
            for neighbor in neighbors:
                if neighbor not in infected:
                    new_infected.add(neighbor)
                    if neighbor not in infected_time:
                        infected_time[neighbor] = time_step

        if len(new_infected) == 0:
            break

        infected.update(new_infected)
        time_step += 1

    return infected_time


def simulate_sir_model(graph, infected_nodes, infection_rate, recovery_rate, num_iterations, str_graph):
    # 统计评价指标
    max_infected_count = 0  # 最大感染人数
    infected_duration = 0  # 感染持续时间
    infected_num = {} # 更新节点数量
    
    # 初始化节点状态
    Ini_V = graph.graph_nodes()[0]
    Ini_E = graph.graph_nodes()[2]
    E_list = graph.graph_nodes()[3]#v 属于哪些边

    # 初始化节点状态
    #right_nodes = {node for node, attr in graph.nodes(data=True) if attr['bipartite'] == 1}
    node_states = {node: 'S' for node in Ini_V}
    
    # 设置初始感染节点
    for node in infected_nodes:
        node_states[node] = 'I'

    # 迭代模拟传播过程
    for iteration in range(num_iterations):

        infected_count = sum(node_state == 'I' for node_state in node_states.values())
        infected_num[iteration] = infected_count

        # 更新最大感染人数
        if infected_count > max_infected_count:
            max_infected_count = infected_count

        # 如果已经没有感染节点，则记录感染持续时间
        if infected_count == 0 and infected_duration == 0:
            infected_duration = iteration

        for node in Ini_V:
            if node_states[node] == 'I':
                num_e = len(E_list[node])
                node_e = list(E_list[node])
                selceted_e = np.random.choice(node_e, 1, replace=False)
                # 处理感染者节点
                for neighbor in Ini_E[selceted_e[0]]: 
                    if node_states[neighbor] == 'S':
                        # 根据感染率确定是否传播
                        if random.random() < infection_rate:
                            node_states[neighbor] = 'I'
                # 根据康复率确定是否康复
                if random.random() < recovery_rate:
                    node_states[node] = 'R'


    # 计算传播速率
    #fig = repr(graph)

    transmission_rate = max_infected_count / num_iterations
    # 绘制感染节点变化图
    '''
    plt.plot(list(infected_num.keys()), list(infected_num.values()))
    plt.xlabel('Iteration')
    plt.ylabel('Infected Nodes')
    plt.title('Number of Infected Nodes over Iterations')
    plt.show()
    '''
    max_value = max(infected_num.values())  # 获取字典中的最大值
    max_keys = min([key for key, value in infected_num.items() if value == max_value])  # 获取最大值对应的键
    
    # 打印评估结果
    print(f'The {str_graph} max infected step is: {max_keys} iterations, Maximum Infected Count is: {max_infected_count}')  
    print(f'The {str_graph} Epidemic Duration: {infected_duration} iterations')
    print(f'The {str_graph} Transmission Rate: {transmission_rate}')

    return node_states



'''
# 参数设置SI
initial_infect = 5  # 初始感染数量

# 初始化二部图和感染节点
right_nodes = [node for node in Graph.nodes if Graph.nodes[node]['bipartite'] == 1]
num_right_nodes = int(len(right_nodes))
                      
#infected = initialize_infected(list(G_infected_nodes), initial_infect)
#infected_G = initialize_infected(list(Gen_infected_nodes), initial_infect)


infected = G_infected_nodes
infected_G = Gen_infected_nodes
#infected_G = copy.deepcopy(infected)

# 模拟SI模型并统计感染时间步长
infected_time_G = simulate_si_model(Ini_Graph, infected)

# 计算平均感染时间
avg_infection_time = sum(infected_time_G.values()) / len(infected_time_G)
print("Infection time for G is {} and average time is {}.".format(max(infected_time_G.values()), avg_infection_time))

# 模拟SI模型并统计感染时间步长
infected_time_G = simulate_si_model(Gen_G, infected_G)

# 计算平均感染时间
avg_infection_time = sum(infected_time_G.values()) / len(infected_time_G)
print("Infection time for Gen is {} and average time is {}.".format(max(infected_time_G.values()), avg_infection_time))
'''

#模拟SIR模型
# 定义SIR模型参数
infected_nodes = np.random.choice(range(m_edges, m_edges+n_nodes), 1, replace=False)
infection_rate = 0.2
recovery_rate = 0.1
num_iterations = 100

# 模拟SIR模型
def SIR_model(G_list):
    graph_str = ['Ini_G', 'Gen_G']
    for i in range(len(G_list)):
        final_node_states = simulate_sir_model(G_list[i], infected_nodes, infection_rate, recovery_rate, num_iterations, graph_str[i])
    return

SIR_model([Ini_Graph,Gen_G])
