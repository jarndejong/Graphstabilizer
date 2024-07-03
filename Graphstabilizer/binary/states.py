# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:46:01 2021

@author: Jarnd
"""


## Global imports

## Local imports
from graph.graphs import get_lin_cluster_adjecency, get_ring_adjecency, get_GHZ_fully_conn_adjecency

from binary.stabilizer import get_generator_matrix_from_adjecency, get_generators_from_generator_matrix

#%% Generator matrices
def get_lin_cluster_generator_matrix(nr_qubits):
    '''
    Returns the generator matrix for the n-qubit linear cluster state.
    '''
    return get_generator_matrix_from_adjecency(get_lin_cluster_adjecency(nr_qubits))

def get_ring_generator_matrix(nr_qubits):
    '''
    Returns the generator matrix for the n-qubit ring graph.
    '''
    return get_generator_matrix_from_adjecency(get_ring_adjecency(nr_qubits))

def get_GHZ_fully_conn_generator_matrix(nr_qubits):
    '''
    Returns the generator matrix for the n-qubit fully connected GHZ graph state.
    '''
    return get_generator_matrix_from_adjecency(get_GHZ_fully_conn_adjecency(nr_qubits))

#%% Generator lists
def get_lin_cluster_generator_list(nr_qubits):
    '''
    Returns a list of generators for the n-qubit linear cluster state.
    '''
    return get_generators_from_generator_matrix(get_lin_cluster_generator_matrix(nr_qubits))

def get_ring_generator_list(nr_qubits):
    '''
    Returns a list of generators for the n-qubit ring graph.
    '''
    return get_generators_from_generator_matrix(get_ring_generator_matrix(nr_qubits))

def get_GHZ_fully_conn_generator_list(nr_qubits):
    '''
    Returns a list of generators for the n-qubit fully connected GHZ graph state.
    '''
    return get_generators_from_generator_matrix(get_GHZ_fully_conn_generator_matrix(nr_qubits))