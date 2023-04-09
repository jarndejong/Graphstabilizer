# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:05:11 2021

@author: Jarnd
"""

## Global imports
# from itertools import product as iterproduct

from numpy import asarray, zeros

## Local imports

#%% Decoding (bit -> string)
def bit_to_string(bitstring):
    '''
    Obtain the string representation of a n-qubit Pauli from its binary representation as a 2n binary vector.
    
    '''
    bitstring = asarray(bitstring)
    
    if len(bitstring) == 1:
        bitstring = bitstring[0]
    # Get n, which is half of the length of the 2n bitstring
    n = int(len(bitstring)/2)
    
    # Init a 'lookup' table
    lookup = ['I','X','Z','Y']
    
    # Init the string
    basename = ''
    
    # Loop through every qubit
    for qubit in range(n):
        # Calculate the corresponding index for the lookup table
        index = bitstring[qubit] + 2*bitstring[qubit+n]
        # Add the Pauli to the string
        basename += lookup[index[0]]
    
    # Return the Pauli
    return basename

def obtain_all_Pauli_strings_from_matrix(matrix):
    '''
    Obtain the string representations of every row in a matrix, returned as a list of strings.
    '''
    ## Init the list
    Pauli_list = []
    
    ## Loop through every row
    for row_nr in range(matrix.shape[0]):
        # Append the returned string
        Pauli_list.append(bit_to_string(matrix[row_nr,:]))
    ## Return
    return Pauli_list

#%% Encoding (string -> bit)
def string_to_bit(Pauli):
    '''
    Obtain the 2n-binary vector representing the Pauli, which is input as a string.
    '''
    ## Calculate the nr of qubits
    n = len(Pauli)
    
    ## Init the length-2n Pauli
    bitstring = zeros((2*n,1), dtype = int)
    
    ## Loop through every single-qubit Pauli; add a one to the bitstring at the proper indices
    for index, sP in enumerate(Pauli):
        if sP == 'X':
            bitstring[index,0] = 1
        elif sP == 'Y':
            bitstring[index,0] = bitstring[index+n, 0] = 1
        elif sP == 'Z':
            bitstring[index+n,0] = 1
    
    ## Return the complete bitstring
    return bitstring

#%% Helper functions
def nqubitPauli_from_single(sP, index, total, as_binary = False):
    '''
    Create an n-qubit Pauli string of a single qubit Pauli, with the single qubit Pauli at the i-th index, and I elsewhere.
    If as_binary is True, return as a bitvector rather than a string
    '''
    # Create the list of single-qubit Paulis per qubit
    P = ['I']*total
    
    # Put in the non-trivial Pauli
    P[index] = sP
    
    # Check for return type
    if as_binary:
        # Return as binary version of joined string
        return string_to_bit(''.join(P))
    else:
        # Return as joined string
        return ''.join(P)