# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:43:28 2021

@author: Jarnd
"""

## Global imports
from numpy import delete, where

## Local imports
from Quantumtools.graphs.elementary import local_complementation

#%% ## Measurements
### non-demolition
def Z_measurement_no_deletion(adjacency_matrix, nodes):
    '''
    Measure qubits at the same time in the Z basis.
    A Z measurement is the same as deleting all edges adjecent to the node,
        which is the same as setting its row and column to 0.
    '''
    adjecency_matrix[nodes,:] = adjecency_matrix[:,nodes] = 0
    
    return adjecency_matrix

def Y_measurement_no_deletion(adjecency_matrix, nodes):
    '''
    Measure qubits in the Y basis.
    A Y measurement is the same as a local complementation on the node, followed by a Z measurement.
    '''
    ## We can't do all of them at the same time unfortunately
    for node in nodes:
        adjecency_matrix = local_complementation(adjecency_matrix, node)
        adjecency_matrix = Z_measurement_no_deletion(adjecency_matrix, node)
    ## Return the adjecency matrix
    return adjecency_matrix

def X_measurement_no_deletion(adjecency_matrix, nodes):
    '''
    Measure qubits in the X basis.
    An X measurement is the same as a local complementation on a random neighbour of the node,
        followed by a Y measurement, followed by a complementation on the random neighbour agian.
    Nodes needs to be a list
    '''
    ## Loop through all nodes if there's more then 1, perform recursion
    if len(nodes) > 1:
        'Multiple nodes!'
        for node in nodes:
            adjecency_matrix = X_measurement_no_deletion(adjecency_matrix, [node])
    ## Only works if there is actually a neighbour
    try:
        # Just get the first neighbour
        random_neighbour = where(adjecency_matrix[nodes,:] == 1)[1][0]
        # Apply the complementation on the random neighbour
        adjecency_matrix = local_complementation(adjecency_matrix, random_neighbour)
        
        # Perform the Y measurement
        adjecency_matrix = Y_measurement_no_deletion(adjecency_matrix, nodes)
        
        # Apply the complementation on the random neighbour again
        adjecency_matrix = local_complementation(adjecency_matrix, random_neighbour)
    ## If there's no neighbour, just perform the Y measurement
    except:
        adjecency_matrix = Y_measurement_no_deletion(adjecency_matrix, nodes)
    
    # Return the updated adjecency matrix
    return adjecency_matrix

def perform_multiple_measurements_no_deletion(adjecency_matrix, measurement_dict):
    '''
    Perform all the measurements in the measurement dict.
    The measurement dict should be a dictionary 
    '''
    nodes = measurement_dict.keys()    
    # Loop through every element in nodes
    for measurement_node in nodes:
        # Get the basis
        measurement_basis = measurement_dict[measurement_node]
        if measurement_basis == 'Z':
            adjecency_matrix = Z_measurement_no_deletion(adjecency_matrix, [measurement_node])
        if measurement_basis == 'Y':
            adjecency_matrix = Y_measurement_no_deletion(adjecency_matrix, [measurement_node])
        elif measurement_basis == 'X':
            adjecency_matrix = X_measurement_no_deletion(adjecency_matrix, [measurement_node])
    
    # Return the adjecency matrix
    return adjecency_matrix

### demolition
def Z_measurement(adjecency_matrix, node):
    '''
    Perform a Z measurement on the graph at node n.
    A Z measurement is the same as deleting the node, which is thesame as deleting the column and row.
    '''
    return delete_node(adjecency_matrix, node)
    
def Y_measurement(adjecency_matrix, node):
    '''
    Perform a Y measurement on the graph at node n.
    A Y measurement constitutes of a local complementation on node n, followed by its deletion, which is a Z measurement.
    '''
    adjecency_matrix = local_complementation(adjecency_matrix, node)
    adjecency_matrix = Z_measurement(adjecency_matrix, node)
    return adjecency_matrix
    
    
def X_measurement(adjecency_matrix, node):
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
        random_neighbour = where(adjecency_matrix[node,:] == 1)[1][0]
        adjecency_matrix = local_complementation(adjecency_matrix, random_neighbour)
        
        adjecency_matrix = Y_measurement(adjecency_matrix, node)
    
        adjecency_matrix = local_complementation(adjecency_matrix, random_neighbour)
    except:
        adjecency_matrix = Y_measurement(adjecency_matrix, node)
    
    return adjecency_matrix


def perform_multiple_measurements(adjecency_matrix, measurement_dict):
    '''
    Perform all the measurements in the measurement dict.
    The measurement dict should be a dictionary 
    '''
    # Get a sorted reversed list of the to be measured nodes
    nodes = list(map(int,measurement_dict.keys()))
    nodes.sort()
    nodes.reverse()
    
    # Loop through every element in nodes
    for measurement_node in nodes:
        # Get the basis
        measurement_basis = measurement_dict[measurement_node]
        if measurement_basis == 'Z':
            adjecency_matrix = Z_measurement(adjecency_matrix, measurement_node)
        if measurement_basis == 'Y':
            adjecency_matrix = Y_measurement(adjecency_matrix, measurement_node)
        elif measurement_basis == 'X':
            adjecency_matrix = X_measurement(adjecency_matrix, measurement_node)
    
    # Return the adjecency matrix
    return adjecency_matrix

def perform_multiple_X_measurements(adjecency_matrix, measurement_dict):
    '''
    Perform all the measurements in the measurement dict or list in the X basis.
    The measurement dict can be a list or dict
    '''
    if type(measurement_dict) == list:
        # Get a sorted reversed list of the to be measured nodes
        nodes = list(map(int,measurement_dict.keys()))
        nodes.sort()
        nodes.reverse()
        for node in measurement_dict:
            adjecenxy_matrix = X_measurement(adjecency_matrix, node)
    elif type(measurement_dict) == dict:
        # Get a sorted reversed list of the to be measured nodes
        nodes = list(map(int,measurement_dict.keys()))
        nodes.sort()
        nodes.reverse()
        
        # Loop through every element in nodes
        for node in nodes:
            adjecency_matrix = X_measurement(adjecency_matrix, node)
            
    # Return the adjecency matrix
    return adjecency_matrix

#%% ## Deletions
