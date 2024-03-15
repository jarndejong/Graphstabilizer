# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:58:41 2021

@author: Jarnd
"""

## Global imports
from numpy import array as _array, matrix as _matrix, diag as _diag, diagflat as _diagflat
from numpy import ones_like as _ones_like, zeros as _zeros, zeros_like as _zeros_like


from numpy import r_, c_, nditer as _nditer

from numpy import sum as _sum, all as _all, logical_or as nplogical_or, delete as _delete,  outer as _outer, round as _round

from numpy.linalg import eigvalsh as _eigvalsh


# from numpy import matrix as _matrix, _array as _array, logical_and as nplogical_and,
# from numpy import where as npwhere

from networkx import adjacency_matrix as nxadjacency_matrix, Graph as nxGraph


## Local imports

#%% Class of adjacency matrices
class AdjacencyMatrix:
    '''
    Simple class of adjacency matrix that checks if its a valid adjacency matrix.
    The input graphy can either be a networkx graph or a square numpy array or matrix.
    '''
    def __init__(self, graph):
        # from Graphstabilizer.checkers.graphs import check_is_AdjacencyMatrixinstance, check_is_networkxinstance

        # Check if matrix is already AdjacencyMatrix object
        if str(type(graph)) == 'Graphstabilizer.graphs.elementary.AdjacencyMatrix':
            self.matrix = graph.matrix

        
        # Check if matrix is a nx graph
        elif type(graph) is nxGraph:
            self.matrix = AdjacencyMatrix(nxadjacency_matrix(graph, dtype = 'int64').todense()).matrix

        else:
            # Check if numpy matrix or ndarray
            if not (type(graph) is _matrix or type(graph) is _array):
                raise TypeError(f"Input adjecency matrix {graph} has wrong type {type(graph)}, it must be numpy.matrix or numpy.ndarray.")

            # Check is it's a square 2d matrix or array
            if graph.shape != (graph.shape[0],graph.shape[0]):
                raise ValueError(f"Input adjecency matrix of shape {graph.shape} is not square.")
            # Check if datatypes are int64 or int32
            if not (str(graph.dtype) == 'int64') or (str(graph.dtype) == 'int32'):
                raise TypeError(f"Input adjecency matrix {graph} has wrong datatype {graph.dtype}, it must be 'int64' or 'int32'.")
            # Check if symmetric
            if not _all(graph.T == graph):
                raise ValueError(f"Input adjacency matrix {graph} is not symmetric.")
            # Check if diagonal is zeros
            if not all(_diag(graph) == _zeros_like(_diag(graph))):
                raise ValueError(f"Input adjacency matrix has non-zero diagonal: {_diag(graph)}.")
            # Check if all entries are zero or 1
            if not _all(nplogical_or(graph == _zeros_like(graph),graph == _ones_like(graph))):
                raise ValueError(f"Input adjacency matrix has non-binary entries: {graph}.")

            self.matrix = graph
    
    def __str__(self):
        return str(self.matrix)
    
    
    ## Properties
    @property
    def shape(self):
        return self.matrix.shape
    
    @property
    def size(self):
        return self.shape[0]
    
    @property
    def identifier(self):
        '''
        Get the identifier for the graph, which is just the binary adjacency matrix as a length-n^2 bitstring
        '''
        return ''.join([str(val) for val in _nditer(self.matrix)])
    
    @property
    def Laplacian(self):
        '''
        Get the Laplacian given the adjacencymatrix.
        '''
        return _diag(_array(_sum(self.matrix, axis = 0))[0]) - self.matrix
    
    @property
    def is_connected(self):
        '''
        Return true if the adjacency matrix represents a connected graph.
        Uses the algebraic connectivity:
            https://en.wikipedia.org/wiki/Algebraic_connectivity
        '''
        return _round(_eigvalsh(self.Laplacian)[1], decimals = 10) > 0
    
    
    
    ## Operational methods (delete and create edges and nodes)
    def add_edge(self, node1, node2):
        '''
        Add an edge between node1 and node2 if it is not already there.
        '''
        self.matrix[node1, node2], self.matrix[node2, node1] = 1, 1
    
    def remove_edge(self, node1, node2):
        '''
        Remove the edge between node1 and node2 if it was there.
        '''
        self.matrix[node1, node2], self.matrix[node2, node1] = 0, 0
    
    def flip_edge(self, node1, node2):
        '''
        Flip the edge between node1 and node2.
        '''
        self.matrix[node1, node2], self.matrix[node2, node1] = 1 - self.matrix[node1, node2], 1 - self.matrix[node2, node1]
    
    def delete_node(self, node):
        '''
        Deletes a node from the matrix.
        '''
        from Graphstabilizer.checkers.elementary import check_is_node_index
        
        check_is_node_index(self.size, node)
        
        self.matrix = _delete(_delete(self.matrix, node, 0), node, 1)
        # self.shape = self.matrix.shape
        # self.size = self.shape[0]
        
    def add_node(self, index):
        '''
        Add a node to the matrix.
        '''
        col_added = c_[self.matrix[:,0:index], _zeros((self.size,1), dtype = 'int'), self.matrix[:, index:]]
        self.matrix = r_[col_added[0:index,:], _zeros((1, self.size+1), dtype = 'int'), col_added[index:,:]]
    
        
    
    ## Return info    
    def get_matrix(self):
        '''
        Return the adjacency matrix as a numpy ndarray or matrix.
        '''
        return self.matrix
    


#%% Local complementations
def local_complementation(A: AdjacencyMatrix, node: int):
    ''' 
    Returns the graph that is locally complemented on the given node. 
    Input should be an adjacencymatrix object A, and a node index to perform the complementation on.
    Returns the adjecency matrix object for a graph which is locally complemented on the given node.    
    '''
    ## Get the column for the node to be complemented. Then take the outer product.
    old_adjacency = A.get_matrix()
    node_column = old_adjacency[:, node]
    neighbours = _outer(node_column, node_column.T)
    
    # Add the locally complemented update graph and do mod 2
    new_adjacency = (old_adjacency + neighbours) % 2
    
    # Delete the diagonal
    new_adjacency = new_adjacency - _diagflat(new_adjacency.diagonal())
    
    # Create new instance of AdjacencyMatrix
    return AdjacencyMatrix(new_adjacency)

#%%
def get_AdjacencyMatrix_from_edgelist(nr_nodes: int, edge_list: list):
    '''
    Obtain an adjacency matrix for the graph with the given number of nodes and the given list of edges.
    '''
    from Graphstabilizer.checkers.elementary import check_is_naturalnr
    from Graphstabilizer.checkers.elementary import check_is_node_index
    
    check_is_naturalnr(nr_nodes)
    
    adjmatrix = _matrix(_zeros(shape = (nr_nodes, nr_nodes), dtype = 'int64'))
    
    # Loop through every entry in the edge list
    for edge in edge_list:
        assert (type(edge) == list) or (type(edge) == tuple), f"Warning, edge {edge} is not of type list or tuple, but {type(edge)}."
        assert len(edge) == 2, f"Warning, edge {edge} has {len(edge)} entries instead of 2."
        for node in edge:
            check_is_node_index(nr_nodes, node)
        adjmatrix[edge[0],edge[1]] = 1
        adjmatrix[edge[1],edge[0]] = 1
    
    return AdjacencyMatrix(adjmatrix)

def get_AdjacencyMatrix_from_string(adjstr : str):
    '''
    Load an AdjacencyMatrix from a string. The string can be retrieved e.g. from a txt file.
    The string should loook like:
        
        [[0,...,1],[1,0,...,0],....,[...]]
    
    So a nested list.
    '''
    from numpy import matrix
    adj = matrix(adjstr)
    adj = adj.reshape((int(adj.size**(1/2)), int(adj.size**(1/2))))
    
    return AdjacencyMatrix(adj)

def get_AdjacencyMatrix_from_identifier(identifier: str):
    '''
    Load an AdjacencyMatrix from a identifier string. 
    This identifier string looks like the string from get_AdjacencyMAtrix_from_string with all commas and brackets removed.
    It's thus a n^2-length list.
    '''
    from numpy import matrix
    adj = matrix([entry for entry in identifier], dtype = int)
    adj = adj.reshape((int(adj.size**(1/2)), int(adj.size**(1/2))))
    
    return AdjacencyMatrix(adj)

# def apply_complementations(state, local_complementations_list):
#     '''
#     Apply local complementations on the state for each node in the list of complementations.
#     state should be the adjecency matrix of a graph (state), and the list com complementations should be an iterable containing node indices.
#     '''
#     for node in local_complementations_list:
#         state = local_complementation(state, node)
    
#     return state

#%% Other Clifford operators
# def apply_CZ(adjecency_matrix, ctrl, targ):
#     '''
#     Apply a CZ gate to the adjecency matrix between the control and target qubit (these can of course be interchanged).
#     This adds a 1 to the correcponsing positions in the adjecency matrix
#     '''
#     # Make the adder
#     B = _zeros_like(adjecency_matrix, dtype = int)
#     B[ctrl , targ] = B[targ , ctrl] = 1
    
#     ## Return with adder
#     return (adjecency_matrix + B) % 2

#%% Helper functions
# def apply_CZ_or_LC_from_list_for_bipartite(state, operations_list):
#     '''
#     Apply the operations in the operations_list in order to the state.
#     The valid operations in the list are Lxa, Lxb or Cx, 
#         where x is any of the node, 
#         where the state has 2x qubits,
#         where Lxa is a local complementation on the left qubit in the node (i.e. node 2x)
#         where Lxb is a local complementation on the right qubit in the node (i.e. node 2x + 1)
#         where Cx is a CZ gate between the qubits in a node (i.e. between 2x and 2x+1)
#     '''
#     ## Loop through every operation
#     for operation in operations_list:
#         if operation[0] == 'C':
#             # Operation is now a CZ gate between 2x and 2x + 1; format of operation is Cxxx with xxx the node
#             state = apply_CZ(state, 2 * int(operation[1:]), 2 * int(operation[1:]) + 1)
#         elif operation[-1] == 'a':
#             # Operation is now Lxxxa with xxx the node; its a local complementation on 2xxx
#             state = local_complementation(state, 2 * int(operation[1:-1]))
#         else:
#             # Operation is now Lxxxb with xxx the node; its a local complementation on 2xxx + 1
#             state = local_complementation(state, 2 * int(operation[1:-1]) + 1)
    
#     return state