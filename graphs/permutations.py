#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 17:14:21 2023

@author: jarn
"""
from Graphstabilizer.states import Graphstate
from Graphstabilizer.graphs.elementary import get_AdjacencyMatrix_from_edgelist, get_AdjacencyMatrix_from_identifier

from itertools import permutations

def obtain_isomorphic_graphs(graphstate):
    '''
    Obtain all the graphs isomorphic to the given graphstate.
    There are between 1 and n! such graphs.
    Returns a list of all unique isomorphic graphs.
    '''
    nr_nodes = graphstate.nr_nodes
    edgelist = graphstate.get_edgelist()
    
    isomorphic_identifiers = set()
    
    for perm in permutations(range(0, nr_nodes)):
        new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in edgelist]
        
        AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
        
        isomorphic_identifiers.add(AM.identifier)
    
        
    return [Graphstate(get_AdjacencyMatrix_from_identifier(identifier)) for identifier in isomorphic_identifiers]
    
def generate_isomorphic_graphs(graphstate):
    '''
    Generate all the isomorphic graphs to the graphstate, or to the list of graph states.
    If a list is provided, 
    the generator yields for a given permutation applied on the entire list, and then loops through all permutations.
    Returns ALL permutations, even if they are equal to a previous one.
    '''
    
    if not isinstance(graphstate, (list, tuple)):
        return generate_isomorphic_graphs([graphstate])
    
    nr_nodes = graphstate[0].nr_nodes

    
    for perm in permutations(range(0, nr_nodes)):
        new_graphstates = []
        for graph in graphstate:
            new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in graph.get_edgelist()]
        
            AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
            
            new_graphstates.append(Graphstate(AM))
        
        if len(new_graphstates) == 0:
            yield new_graphstates[0]
        
        yield new_graphstates
        
    