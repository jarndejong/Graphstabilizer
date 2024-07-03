# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:57:39 2021

@author: Jarnd
"""

import networkx as nx
from graph_operations import _local_complementation, X_measurement, Z_measurement, Y_measurement
from matplotlib import pyplot as plt
n = 15

#%% ############################################### Local complementation route
L = nx.Graph()

## Add the nodes to the graph
for i in range(n):
    L.add_node(i)

## Add the edges to the graph
for i in range(n-1):
    L.add_edge(i,i+1)
    
# L.add_edge(0,n-1)    

# nx.draw_circular(L)

## Aply the local complementations
for i in range(1, n - 1):
    L = _local_complementation(L, i)
    
plt.figure()
nx.draw_circular(L, with_labels = True)

## Remove nodes
# remove_nodes = list(range(1,n-1,2))
remove_nodes = list(range(2,n-1,2))
# remove_nodes = [1,3,5, 7]
# remove_nodes = [2,4]
# remove_nodes = [0,3,6]

L.remove_nodes_from(remove_nodes)
plt.figure()
nx.draw_circular(L, with_labels = True)

#%% #################################### Pauli measurement route
P = nx.Graph()

for i in range(n):
    P.add_node(i)

## Add the edges to the graph
for i in range(n-1):
    P.add_edge(i,i+1)

## Set nodes to measure
measure_nodes = list(range(2, n-1, 2))
measure_nodes = list(range(1,n-1,2))


plt.figure()
nx.draw_circular(P, with_labels = True)

for measure_node in measure_nodes:
    P = X_measurement(P, measure_node)
plt.figure()
nx.draw_circular(P, with_labels = True)

# P = Y_measurement(P, 4)
# plt.figure()
# nx.draw_circular(P, with_labels = True)