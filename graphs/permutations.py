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
    return [Graphstate(get_AdjacencyMatrix_from_identifier(identifier)) for identifier in obtain_isomorphic_graph_identifiers(graphstate)]

def obtain_isomorphic_graph_identifiers(graphstate):
    '''
    Obtain the identifiers as all the graphs isomorphic to the given graphstate.
    There are between 1 and n! such graphs.
    Returns a set of the identifiers of all unique isomorphic graphs.
    '''
    nr_nodes = graphstate.nr_nodes
    edgelist = graphstate.get_edgelist()
    
    isomorphic_identifiers = set()
    
    for perm in permutations(range(0, nr_nodes)):
        new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in edgelist]
        
        AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
        
        isomorphic_identifiers.add(AM.identifier)
    
    
    return isomorphic_identifiers

def obtain_isomorphic_graph_identifiers_and_permutations(graphstate):
    '''
    Obtain the identifiers as all the graphs isomorphic to the given graphstate.
    There are between 1 and n! such graphs.
    Returns a id. identifierestopermutations, permutationstoidentifiers where:
        id is a set containing the unique isomorphic graphs (as identifiers)
        identifierestopermutations is a dictionary with keys those graph identifiers, and as values the associated permutation
        permutationstoidentifiers is a dictionary with keys those permutations, and as values those graph identifiers
    '''
    nr_nodes = graphstate.nr_nodes
    edgelist = graphstate.get_edgelist()
    
    isomorphic_identifiers = set()
    
    identifierestopermutations = {}
    permutationstoidentifiers = {}
    
    for perm in permutations(range(0, nr_nodes)):
        new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in edgelist]
        
        AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
        
        if AM.identifier not in isomorphic_identifiers:
            identifierestopermutations[AM.identifier] = perm
            permutationstoidentifiers[perm] = AM.identifier
            isomorphic_identifiers.add(AM.identifier)
    
    
    return isomorphic_identifiers, identifierestopermutations, permutationstoidentifiers

def obtain_automorphism_permutations(graphstate):
    '''
    '''
    nr_nodes = graphstate.nr_nodes
    edgelist = graphstate.get_edgelist()
    
    automorphisms = []
    
    for perm in permutations(range(0, nr_nodes)):
        new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in edgelist]
        
        AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
        
        if AM.identifier == graphstate.identifier:
           automorphisms.append(perm)
          
    return automorphisms

def obtain_all_permuted_graphs_identifiers(graphstate):
    '''
    Obtain the identifiers of all the graphs that you get when permuting the nodes of the given graph state.
    This might be more than all the isomorphic graphs, as different permutations can give the same graph;
    Returns a list of all these permutations.
    '''
    nr_nodes = graphstate.nr_nodes
    edgelist = graphstate.get_edgelist()
    
    isomorphic_identifiers = list()
    
    for perm in permutations(range(0, nr_nodes)):
        new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in edgelist]
        
        AM = get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)
        
        isomorphic_identifiers.extend(AM.identifier)
    
    
    return isomorphic_identifiers


def permute_graph_nodes(graphstate, permutation):
    '''
    This function returns a graphstate with the same properties as the given graphstate, but with the nodes permuted according to the given permutation.
    permutation should be a list of nr_qubits elements, where every node is given exactly once.
    
    
    return the new graphstate
    '''
    nr_nodes = graphstate.nr_nodes
    
    assert set(permutation) == set(range(nr_nodes)), f"Permutation given by {permutation} does not contain the indices {list(range(nr_nodes))}"
    
    new_edgelist = [(permutation[edge[0]], permutation[edge[1]]) for edge in graphstate.get_edgelist()]
    
    return Graphstate(get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist))
    

# TODO: This function is failing. Don't know why yet.
# def generate_isomorphic_graphs(graphstate):
#     '''
#     Generate all the isomorphic graphs to the graphstate, or to the list of graph states.
#     If a list is provided, 
#     the generator yields for a given permutation applied on the entire list, and then loops through all permutations.
#     Returns ALL permutations, even if they are equal to a previous one.
#     '''
    
#     if not isinstance(graphstate, (list, tuple)):
#         print('rerun')
#         print([graphstate])
#         return generate_isomorphic_graphs([graphstate])
    
#     nr_nodes = graphstate[0].nr_nodes
#     print(nr_nodes)

    
#     for perm in permutations(range(0, nr_nodes)):
#         new_graphstates = []
#         for graph in graphstate:
#             new_edgelist = [(perm[edge[0]], perm[edge[1]]) for edge in graph.get_edgelist()]
        
            
#             new_graphstates.append(Graphstate(get_AdjacencyMatrix_from_edgelist(nr_nodes, new_edgelist)))
        
#         if len(new_graphstates) == 0:
#             yield new_graphstates[0]
        
#         yield new_graphstates
        
    