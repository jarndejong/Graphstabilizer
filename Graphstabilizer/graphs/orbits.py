#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:35:42 2024

@author: jarn
"""

from Graphstabilizer.states import Graphstate
from Graphstabilizer.graphs.elementary import get_AdjacencyMatrix_from_identifier, get_AdjacencyMatrix_from_string, AdjacencyMatrix, get_AdjacencyMatrix_from_edgelist


## Computational functions
def generate_orbit_graphs_identifiers(G, printing = False):
    '''
    Generate the entire LC orbit for a given Graphstate G. Returns a set of the identifiers.
    '''    
    all_identifiers = set([G.identifier])
    
    previous_identifiers = set([G.identifier])
    
    level = 0
    while True:
        
        level +=1
        if printing:
            print(f"\t\tAt level {level}")
        new_identifiers = set()
        for identifier in previous_identifiers:
            
            for node in range(G.nr_nodes):
                adj = get_AdjacencyMatrix_from_identifier(identifier)
                Gnew = Graphstate(adj)
                Gnew.local_complement(node)
            
                if Gnew.identifier not in all_identifiers:
                    all_identifiers.add(Gnew.identifier)
                    new_identifiers.add(Gnew.identifier)
    
        
        if len(new_identifiers) == 0:
            break
        
        previous_identifiers = set()
        previous_identifiers.symmetric_difference_update(new_identifiers)
        
        if printing:
            print(f"\t\t\t{len(previous_identifiers)} in layer.")
    
    return all_identifiers

def generate_orbit_graphs_identifiers_from_identifier(identifier):
    '''
    Generate the entire LC orbit for a given Graphstate identifier. Returns the identifiers.
    '''
    return generate_orbit_graphs_identifiers(Graphstate(get_AdjacencyMatrix_from_identifier(identifier)))
# def generate_orbit_graphs_identifiers_quicker(G):
#     '''
#     Generate the entire orbit for a given Graphstate G. Returns the identifiers.
#     Works under the assumption that locally complementing a node should happen at most once for every node.
#     '''
#     all_identifiers = set([G.identifier])
    
#     previous_identifiers = set([G.identifier, set()])
    
#     while True:
#         new_identifiers = set()
#         for identifier, already_seen_nodes in previous_identifiers:
        
#             for node in set(range(G.nr_nodes)) - already_seen_nodes:
#                 adj = get_AdjacencyMatrix_from_identifier(identifier)
#                 Gnew = Graphstate(adj)
#                 Gnew.local_complement(node)
            
#                 if Gnew.identifier not in all_identifiers:
#                     all_identifiers.add(Gnew.identifier)
#                     new_identifiers.add(Gnew.identifier)
    
        
#         if len(new_identifiers) == 0:
#             break
        
#         previous_identifiers = set()
#         previous_identifiers.symmetric_difference_update(new_identifiers)
    
#     return all_identifiers

def generate_orbit_graphs(G):
    '''
    Generate the entire LC orbit for a given Graphstate G. 
    Returns the graphs as a list. The initial graph will be the first in the list.
    '''    
    return [G] + [Graphstate(get_AdjacencyMatrix_from_identifier(identifier)) for identifier in generate_orbit_graphs_identifiers(G) if identifier != G.identifier]

def generate_orbit_graphs_from_identifier(identifier):
    '''
    Generate the entire LC orbit for a given graphstate G, given as an identifier. 
    Returns the graphs as a list. The initial identifier will be the graph first in the list.
    '''
    return generate_orbit_graphs(Graphstate(get_AdjacencyMatrix_from_identifier(identifier)))


def get_total_nr_graphs(nr_qubits):
    '''
    Returns the total number of (connected) graphs with the given number of qubits.
    '''
    raise NotImplementedError
    if nr_qubits == 2:
        return 1
    return int(2**((nr_qubits*(nr_qubits - 1))/2)) - nr_qubits*(get_total_nr_graphs(nr_qubits - 1))

#%% GENERAL CONNECTED GRAPHS
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_all_graphs(nr_nodes):
    '''
    Generate all simple graphs of a given number of nodes.
    '''
    for nr, edgelist in enumerate(_powerset(combinations(range(nr_nodes), 2))):
        yield nr, Graphstate(get_AdjacencyMatrix_from_edgelist(nr_nodes, edgelist))

def generate_all_connected_graphs_naive(nr_nodes):
    '''
    Generate all simple connected graphs of a given number of nodes
    '''
    for nr, graphstate in generate_all_graphs(nr_nodes):
        if graphstate.is_connected:
            yield nr, graphstate
