# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:45:58 2020

@author: Jarnd
"""

## Global imports
from numpy import zeros, eye, matrix, where

from copy import deepcopy

## Local imports


#%% Elementary functions
def get_simplectic_P(nr_qubits):
    '''
    Get the P matrix for the simplectic inner product.
    '''
    return matrix(eye(2*nr_qubits, k = nr_qubits, dtype = int) + eye(2*nr_qubits, k = -nr_qubits, dtype = int))

def bin_product(A, B):
    '''
    Compute the binary (mod-2) inner product of two bitmatrixes A and B
    '''
    return A.T @ B % 2

def inner_product_simplectic(A,B):
    '''
    Compute the simplectic inner product of bitvectors A and B
    '''
    # Get nr of qubits for simplectic P
    
    nr_qubits = int(B.shape[0]/2)
    # Get the simplectic P
    P = get_simplectic_P(nr_qubits)
    
    # Return
    return bin_product(A, P @ B)

#%% Echelon form functions
def get_echelon(M_in):
    '''
    Returns the echelon form of a binary matrix M. Also returns the indices of the pivot columns.
    '''
    # Determine some data
    nr_rows, nr_columns = M_in.shape
    
    # Set some parameters and init the output matrix
    M = deepcopy(M_in)   
    pivot_columns = []
    
    # We will start looping through all rows within a column and check for pivots at indices after the previous pivot row index; 
    # we thus init at index -1 to start searching from row 0
    pivot_row = -1
    
    
    # Loop through every column searching for a pivot position. In every column, we look from index (pivot_row + 1) untill the end for non-zero elements.
    for col in range(nr_columns):
        # Get the indices of all non-zero elements, starting from the row of the previous pivot row add one.
        # A new pivot can only be at a lower row than the previous pivot
        nonzero_indices = [i for i in range(pivot_row + 1, nr_rows) if M[i, col] == 1]
        
        # Check if there is a pivot in this column
        if len(nonzero_indices) >= 1:
            # Now there is a non-zero element in the erst of the column - so a pivot.
            # Append the column index to the indices
            pivot_columns.append(col)
            
            # Get the new pivot row index
            pivot_row_new = nonzero_indices[0]
            
            # Delete all ones after the pivot row in this column
            for index in nonzero_indices[1:]:
                M[index, :] = (M[index,:] + M[pivot_row_new,:]) % 2
            
            # Check if the new pivot row is one after the previous pivot_row, swap otherwise
            if pivot_row_new - pivot_row >= 2:
                M = _swap_rows(M, pivot_row_new, pivot_row+1)
                        
            # Update the (previous) pivot_row at the end of the loop with the new value, which is one more
            pivot_row = pivot_row+1
        
    # Return the data
    return M, pivot_columns
    

def get_reduced_echelon(M_in, pivot_columns):
    '''
    Get the reduced echelon form of the matrix M_in. The pivot column indices should also be provided.
    This regards a matrix over the finite binary field.
    '''
    # Get some prelim data
    nr_rows, nr_columns = M_in.shape
    M = deepcopy(M_in)
    
    # Loop through every pivot
    for pivot_nr in range(len(pivot_columns)):

        # For the pivot column, get all the rows which also have entry one within the pivot column. 
        # Get it in reversed order.
        indices_1 = [i for i in range(nr_rows-1, -1, -1) if M[i, pivot_columns[pivot_nr]] == 1]

        # Loop through all these rows (expect for the first one, which is the pivot itself) and set them all to zero by adding the pivot row
        for index in indices_1[1:]:
            M[index, :] = (M[index, :] + M[indices_1[0],:]) % 2
    
    # Return the reduced echelon form matrix
    return M

#%% Higher level functions
def get_nullspace_basis(M, get_pivs = False):
    '''
    Get a basis for the null space of a matrix M. 
    This is done by first getting the echelon & reduced echelon form of M, from which the nullspace is relatively easily obtained.
    Returns the basis as a list of vectors. If get_pivs is True, also returns the pivot column indices.
    '''
    # Get the echelon and reduced echelon form.
    M_ech, pivot_columns = get_echelon(M)
    M_red = get_reduced_echelon(M_ech, pivot_columns)
    
    # Get the basis from the reduced echelon form
    basis = _get_nullspace_basis(M_red, pivot_columns)
    
    # Reduce the basis, and the pivot columns if get_pivs is True
    if get_pivs:
        return basis, pivot_columns
    else:
        return basis




#%% Helper functions
def _swap_rows(M, index_1, index_2):
    '''
    Swap the rows at index_1 and index_2 of matrix M 
    '''
    
    # Init an identity matrix that we then transform to a permute matrix
    P = eye(M.shape[0], dtype = 'int')
    
    # Set the diagonal entries of index_1 and index_2 to 0
    P[index_1, index_1] = P[index_2, index_2] = 0
    # Set the off-diagonal entries of index_1 and index_2 to 1
    P[index_1, index_2] = P[index_2, index_1] = 1
    
    # Compute the permutation and return
    return P@M

def _get_nullspace_basis(M, pivot_columns):
    '''
    Get a basis of the nullspace of M, when M is in reduces echelon form. Also provide the pivot columns.
    '''
    # Some useful data
    nr_rows, nr_columns = M.shape
    
    # These are the column indices that are not pivots
    free_columns = [i for i in range(nr_columns) if i not in pivot_columns]
    
    # Init the bases
    basis = []
    
    # Now loop through every free column; and make a new vector for the basis
    for column_index in free_columns:
        # Init the vector as the all-zeros vector
        vector = zeros((nr_columns,1), dtype = 'int')
        
        # The free parameter index itself is always 1
        vector[column_index] = 1
        
        # Now we add 1's for every non-free parameter that is dependend on this particular free parameter.
        # Any time there is a one in this particular free column, 
        # it indicates that the pivot in that row is dependend on the free parameter.
        # Thus, we get all the indices for which the free column is 1 - which indicates some rows.
        # Then, we look which pivot corresponds to these rows - these pivot (columns) are parameters that
        # depend on this particular free parameter, so we add 1 there.
        
        # First, get the (row) indices where this free column is nonzero
        row_indices_1 = [i for i in range(nr_rows) if M[i,column_index] == 1]
        
        # For all these row indices, check the corresponding full row and find the first 1 in there
        # Add a one on this index to the vector
        for row_index in row_indices_1:
            vector[where(M[row_index, :] == 1)[0]] = 1
        
        # # Anytime the free parameter also shows up in the rest of the pivot column, it should be part of the basis vector as well.
        # # However, if the free parameter (column) is smaller than the total number of rows, it might delete over itself.
        # # To circumvent this, check if the current column index is lower or equal than the total nr of rows. 
        # # If so, only copy the part of the column that is above the current column index
        # # Check if current column index is equal or lower than total nr of rows
        # if column_index <= nr_rows:
        #     vector[0:column_index] = M[0:column_index,column_index:column_index+1]
        # else:
        #     vector[0:nr_rows] = M[:,column_index:column_index+1]
        
        # Append the vector to the basis
        basis.append(vector)
        
    # Return the basis
    return basis
        
def _get_pivots(M_in):
    '''
    Compute and return the rows and columns of the pivots for a matrix M.
    Returns a list of rows and a list of column
    '''
    # Determine some data
    nr_rows, nr_columns = M_in.shape
    
    # Set some parameters and init the output matrix
    M = deepcopy(M_in)
    pivot_rows = []
    pivot_columns = []
    
    # We will start looping through all rows within a column and check for pivots at indices after the previous pivot row index; 
    # we thus init at index -1 to start searching from row 0
    pivot_row = -1
    
    # Loop through every column searching for a pivot position. In every column, we look from index (pivot_row + 1) untill the end for non-zero elements.
    for col in range(nr_columns):
        # Get the indices of all non-zero elements, starting from the row of the previous pivot row add one.
        # A new pivot can only be at a lower row than the previous pivot
        nonzero_indices = [i for i in range(pivot_row + 1, nr_rows) if M[i, col] == 1]
        
        # Check if there is a pivot in this column
        if len(nonzero_indices) >= 1:
            # Now there is a non-zero element in the erst of the column - so a pivot.
            # Append the column index to the indices
            pivot_columns.append(col)
            
            # Get the new pivot row index
            pivot_row_new = nonzero_indices[0]
            
            # Delete all ones after the pivot row in this column
            for index in nonzero_indices[1:]:
                M[index, :] = (M[index,:] + M[pivot_row_new,:]) % 2
            
            # Check if the new pivot row is one after the previous pivot_row, swap otherwise
            if pivot_row_new - pivot_row >= 2:
                M = _swap_rows(M, pivot_row_new, pivot_row+1)
            
            # Append the new pivot row to the pivot_rows list - we've swapped it to be the one immediately after the previous row
            pivot_rows.append(pivot_row_new)
            
            # Update the (previous) pivot_row at the end of the loop with the new value, which is one more
            pivot_row = pivot_row+1
        
    # Return the data
    return pivot_rows, pivot_columns

def _remove_measured_nodes(pauli, L):
    if type(pauli) == str:
        return ''.join([pauli[i] for i in L])
    else:
        nodes = [i for i in L]
        nodes.extend([i + int(pauli.size/2) for i in nodes])
        return pauli[0,nodes]
