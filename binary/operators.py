# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:17:52 2021

@author: Jarnd
"""
## Global imports
# from itertools import product as iterproduct

# from numpy import zeros_like

## Local imports
from binary.Paulistrings import string_to_bit, nqubitPauli_from_single

#%% String representation
def get_X_strings_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one X operators as Pauli strings, with the X on the indices (one operator per index) and I elsewhere.
    '''
    ## Init the list of operators
    ops = []
    
    # Loop through every index
    for operator in indices:
        # Append the operator as a full n-qubit Pauli string
        ops.append(nqubitPauli_from_single('X', operator, nr_qubits))
        
    # Return the entire list
    return ops

def get_Y_strings_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one Y operators as Pauli strings, with the Y on the indices (one operator per index) and I elsewhere.
    '''
    ## Init the list of operators
    ops = []
    
    # Loop through every index
    for operator in indices:
        # Append the operator as a full n-qubit Pauli string
        ops.append(nqubitPauli_from_single('Y', operator, nr_qubits))
        
    # Return the entire list
    return ops

def get_Z_strings_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one Z operators as Pauli strings, with the Z on the indices (one operator per index) and I elsewhere.
    '''
    ## Init the list of operators
    ops = []
    
    # Loop through every index
    for operator in indices:
        # Append the operator as a full n-qubit Pauli string
        ops.append(nqubitPauli_from_single('Z', operator, nr_qubits))
        
    # Return the entire list
    return ops

def get_single_Pauli_strings_operators(indices, single_Paulis, nr_qubits):
    '''
    Get a list of n-qubit weight one Pauli operators as Pauli strings, with the (non-trivial) Pauli on the indices (one operator per index) and I elsewhere.
    Very useful to get a list of single-qubit measurement operators.
    '''
    ## Init the list of operators
    ops = []
    
    # Loop through every index
    for operator, singlePauli in zip(indices, single_Paulis):
        # Append the operator as a full n-qubit Pauli string
        # Check for the single Pauli and use correct function accordingly
        if singlePauli == 'X':
            ops.append(nqubitPauli_from_single('X', operator, nr_qubits))
        elif singlePauli == 'Y':
            ops.append(nqubitPauli_from_single('Y', operator, nr_qubits))
        elif singlePauli == 'Z':
            ops.append(nqubitPauli_from_single('Z', operator, nr_qubits))
        elif singlePauli == 'I':
            pass
        else:
            raise ValueError(f'Invalid character provided; cannot parse {singlePauli} into a single qubit Pauli.')
    # Return the entire list
    return ops

#%% Bitvector representation
def get_X_bitvector_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one X operators as bitvectors, with the X on the indices (one per operator index) and I elsewhere
    '''
    ## Get the Pauli strings first
    ops = get_X_strings_operators(indices, nr_qubits)
    
    ## Return converted to bitstrings
    return [string_to_bit(operator_string) for operator_string in ops]

def get_Y_bitvector_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one X operators as bitvectors, with the X on the indices (one per operator index) and I elsewhere
    '''
    ## Get the Pauli strings first
    ops = get_Y_strings_operators(indices, nr_qubits)
    
    ## Return converted to bitstrings
    return [string_to_bit(operator_string) for operator_string in ops]

def get_Z_bitvector_operators(indices, nr_qubits):
    '''
    Get a list of n-qubit weight-one X operators as bitvectors, with the X on the indices (one per operator index) and I elsewhere
    '''
    ## Get the Pauli strings first
    ops = get_Z_strings_operators(indices, nr_qubits)
    
    ## Return converted to bitstrings
    return [string_to_bit(operator_string) for operator_string in ops]

def get_single_Pauli_bitvector_operators(indices, single_Paulis, nr_qubits):
    '''
    Get a list of n-qubit weight one Pauli operators as bitvectors, with the (non-trivial) Pauli on the indices (one operator per index) and I elsewhere.
    Very useful to get a list of single-qubit measurement operators.
    '''   
    ## Get the Pauli strings first
    ops = get_single_Pauli_strings_operators(indices, single_Paulis, nr_qubits)

    ## Return converted to bitstrings
    return [string_to_bit(operator_string) for operator_string in ops]

#%% Operators (binary representation)
def get_basis_for_indices(indices, nr_qubits):
    '''
    Get a basis for the binary subspace spanned over the qubits at the indices provided.
    Returns a weight-one n-qubit X and Z operator at every qubit listed in the indices, in the form of a binary vector.
    '''   
    return get_X_bitvector_operators(indices, nr_qubits) + get_Z_bitvector_operators(indices, nr_qubits)