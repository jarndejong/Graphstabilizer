# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:56:42 2021

@author: Jarnd
"""
import networkx as nx
from copy import deepcopy
import itertools as it
def _local_complementation(graph, vertex):
        '''
        Performs a local complementation on a vertex in graph.
        Args:
            vertex (int): Index of vertex.
        '''
        LCgraph = deepcopy(graph)
        neighborhood = LCgraph.neighbors(vertex)
        neighborhood_edges = it.combinations(neighborhood, 2)
        for v1, v2 in neighborhood_edges:
            if LCgraph.has_edge(v1, v2):
                LCgraph.remove_edge(v1, v2)
            else:
                LCgraph.add_edge(v1, v2, weight=1)
        return LCgraph

def Z_measurement(graph, node):
    '''
    Perform a Z measurement on the graph at node n.
    A Z measurement constitutes of the deletion of node n.
    '''
    graph.remove_node(node)
    return graph
    
def Y_measurement(graph, node):
    '''
    Perform a Y measurement on the graph at node n.
    A Y measurement constitutes of a local complementation on node n, followed by its deletion, which is a Z measurement.
    '''
    graph = _local_complementation(graph, node)
    graph.remove_node(node)
    return graph
    
    
def X_measurement(graph, node):
    '''
    Perform an X measurement on the graph at node n.
    An X measurement constitutes of:
        a local complementation on any neighbour of the node
        a local complementation on the node itself
        deletion of the node itself
        a local complementation on the original neighbour
    
    Which is the same as:
        a local complementation on a random neightbour
        a Y measurement
        a local complementaion on the same neighbour
    '''
    try:
        random_neighbour = list(graph.edges(node))[0][1]
        graph = _local_complementation(graph, random_neighbour)
        
        graph = Y_measurement(graph, node)
    
        graph = _local_complementation(graph, random_neighbour)
    except:
        graph = Y_measurement(graph, node)
    
    
    
    return graph