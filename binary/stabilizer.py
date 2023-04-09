# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:06:03 2021

@author: Jarnd
"""
## Global imports
from itertools import product as iterproduct

from numpy import zeros_like, zeros, matrix, concatenate, eye, diagonal, diagflat
from numpy.linalg import inv

## Local imports
from binary.Paulistrings import bit_to_string
from binary.linAlg import _swap_rows, get_echelon

#%% Helper/base functions
def return_stabilizer(generators, with_strings = False):
    '''
    Return the entire stabilizer set from a list of generators as binary vectors.
    If with_strings = True, also return the stabilizers as Pauli strings
    '''
    # Init the stabilizers list for the binary vectors and the strings
    stab = []
    stab_paulis = []
    
    # Loop through every possible combination of the generators
    for iteration in iterproduct((0,1), repeat = len(generators)):
        
        # Init the stabilizer element as a zeros-vector
        base = zeros_like(generators[0], dtype = 'int')
        
        # Add every generator to the base as long as it's in this iteration
        for index, generator in enumerate(generators):
            if iteration[index] == 1:
                base += generator
        
        # Append the new vector mod 2
        stab.append(base % 2)
        
        # Append the decoded bitstring
        stab_paulis.append(bit_to_string(base % 2))
    
    if with_strings:
        return stab, stab_paulis
    else:
        return stab

def shift_qubits(generators_list, k = 1, direction = 'right'):
    '''
    Shift the qubits of the generators k positions to the right or left, with the last qubit becoming the first one, or first becoming last.
    '''
    ## Check if k is 0
    if k == 0:
        return generators_list
    ## Recursive for higher k
    if k > 1:
        generators_list =  shift_qubits(generators_list, k - 1, direction)
    
    ## Get n
    n = int(len(generators_list[0])/2)
    
    ## Init the shifter matrix
    shifter = eye(n, k = -1, dtype = 'int')
    
    shifter[0,-1] = 1
    
    ## Flip if direction is left
    if direction == 'left':
        shifter = shifter.T
    
    ## Get the full shifter matrix
    full_shifter = zeros((2*n, 2*n), dtype = 'int')
    full_shifter[0:n,0:n] = shifter
    full_shifter[n:,n:] = shifter
    
    ## Apply to every generator and return
    return [full_shifter @ generator for generator in generators_list]

#%% Transformations between generator lists and matrices
def get_generator_matrix_from_generators(generators):
    '''
    Concatenate all generators into single generator matrix.
    Provide the generators as a list of bitvectors.
    Returns the generator matrix a numpy matrix.
    '''
    ## Set dimensions
    n = int(len(generators[0])/2)
    
    ## Init
    G = zeros((2*n,n), dtype = 'int')
    
    ## Put every generator in the matrix
    for i, g in enumerate(generators):
        G[:,i:i+1] = g
    
    ## Return
    return matrix(G)

def get_generators_from_generator_matrix(generator_matrix):
    '''
    Obtain a list of the generators from the generator matrix.
    Provide the generator matrix as a numpy matrix.
    Returns the generators as a list of bitvectors.
    '''
    return [generator_matrix[:,i] for i in range(generator_matrix.shape[1])]

#%% Transformations between adjecency matrices and generator matrices of the according graph states
def get_generator_matrix_from_adjecency(adjecency):
    '''
    Get the generator matrix for a graph state with an adjecency matrix for the accompanying graph.
    
    The generator matrix is [I A.T].T
    '''
    ## Get n
    n = adjecency.shape[0]
    
    # Return the matrix
    return concatenate((eye(n, dtype = 'int'), adjecency), axis = 0)

def get_adjecency_from_generator_matrix(generator_matrix):
    '''
    First, checks if the provided generator matrix indeed is a valid graph state (i.e. check if the upper nxn block is the identity).
    Second, checks if the lower block has zero diagonal.
    Then, returns the lower block, which corresponds to the adjecency matrix.
    '''
    ## Get n
    n = generator_matrix.shape[1]
    
    ## Check if upper block is equal to identity
    assert all(generator_matrix[0:n,:] == eye(n)), 'Warning: this generator matrix does not have an upper identity block (e.g. all X)'
    
    ## Check if lower block has zero diagonal
    assert all(diagonal(generator_matrix[n:,:] == 0)), 'Warning: this generator matrix does not have a lower block with zero diagonal'
    
    # Finally, return lwoer block
    return generator_matrix[n:,:]

#%% Other graph-related functions
def get_LC_equivalent_graph(generator_matrix):
    '''
    Get a local clifford-equivalent graph to the state specified by the generator matrix.
    Returns the adjecency matrix of the graph.
    '''
    ## Get n
    n = generator_matrix.shape[1]
    
    ## Get the X part of the generator matrix // X is the upper nxn block ~ pass it immediately to next function
    ## Get its pivot rows immediately and from it the non-pivot rows
    _, piv_rows = get_echelon(generator_matrix[:n,:n].T)
    non_piv_rows = [i for i in range(n) if i not in piv_rows]
    
    ## Apply Hadamards to every non-pivot row qubit of G; this makes the X block invertible
    ## Loop through every said qubit
    for row in non_piv_rows:
        generator_matrix = perform_Hadamard_on_generator_matrix(generator_matrix, row)
    
    ## Compute the inverse of the new X block
    iXt = inv(generator_matrix[:n,:n]).astype(dtype = int)
        
    ## Compute the Z part after the transformation, which is the lower nxn block times the iXt transformation
    Zt = generator_matrix[n:, :n] @ iXt
    
    ## Return the Z part, with zero diagonal
    return (Zt - diagflat(Zt.diagonal())) % 2

#%% (clifford) operations of stabilizers
def perform_Hadamard_on_generator_matrix(generator_matrix, qubit_index):
    '''
    Performs a Hadamard operation on the qubit indexed by qubit_index. 
    The Hadamard operations conjugates X and Z, so in the stabilizer framework it swaps the X- and Z-row for the target qubit.
    In our case, that's equavalent to swapping the rows at qubit_index and qubit_index + n, where n is the total number of qubits.
    '''
    ## Get n
    n = int(generator_matrix.shape[1])
    
    ## Swap the rows and return
    return _swap_rows(generator_matrix, qubit_index, qubit_index + n)