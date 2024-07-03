# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:58:49 2021

@author: Jarnd
"""

## Global imports
from numpy import matrix as _npmatrix, zeros as _npzeros, ones as _npones, eye as _npeye


## Local imports
from Graphstabilizer.graphs.elementary import AdjacencyMatrix as _AdjacencyMatrix, get_AdjacencyMatrix_from_edgelist as _get_AdjacencyMatrix_from_edgelist

    
#%%
def empty_graph(nr_nodes):
    '''
    Get the adjacency matrix for an empty graph, i.e. the all-zeros matrix.
    '''
    adjmatrix = _npmatrix(_npzeros((nr_nodes,nr_nodes), dtype = int))
    return _AdjacencyMatrix(adjmatrix)

def empty(nr_nodes):
    '''
    See empty_graph
    '''
    return empty_graph(nr_nodes)

# def twod_cluster(r, c = None):
#     '''
#     Get the adjacency matrix of a 2d cluster state with r rows and c columns. 
#     If no c is provided, a square matrix is returned (i.e. c=r).
#     Returns an adjacencymatrix object.
#     '''
#     if c is None:
#         c = r
    
#     # Create a networkx graph
#     nxG = _nxgrid_2d_graph(r,c)
    
#     return _AdjacencyMatrix(nxG)   
def twod_cluster(r, c = None):
    '''
    Get the adjacency matrix of a 2d cluster state with r rows and c columns.
    If no c is provided, a square matrix is returned (i.e. c = r).
    Returns an adjacencymatrix object
    '''
    if c is None:
        c = r
    
    # Create an empty graph first
    Adj = empty_graph(r * c)
    
    # Create horizontal lines
    for row in range(r):
        for column in range(c - 1):
            Adj.add_edge(row*c + column, row*c + column + 1)
    
    # Create vertical lines
    for column in range(c):
        for row in range(r - 1):
            Adj.add_edge(row*c + column, (row+1)*c + column)
    
    return Adj

def oned_cluster(nr_nodes):
    '''
    See linear_cluster()
    '''
    return linear_cluster(nr_nodes)

def linear_cluster(nr_nodes):
    '''
    Get the adjacency matrix of a linear cluster state of nr_qubits qubits.
    '''
    adjmatrix = _npmatrix(_npeye(nr_nodes, k = 1, dtype = int) + _npeye(nr_nodes, k = -1, dtype = int))
    
    return _AdjacencyMatrix(adjmatrix)

def line(nr_nodes):
    '''
    Get the adjacency matrix of a line graph with nr_qubits nodes. Same as linear_cluster.
    '''
    return linear_cluster(nr_nodes)
    
def ring(nr_nodes):
    '''
    Get the adjacency matrix of a ring graph with nr_nodes nodes.
    '''
    adjmatrix = linear_cluster(nr_nodes)
    
    adjmatrix.matrix[0,nr_nodes - 1] = adjmatrix.matrix[nr_nodes - 1, 0] = 1
    
    return adjmatrix

def circle(nr_nodes):
    '''
    See ring
    '''
    return ring(nr_nodes)

def tree(nr_layers, branch_factor = 2):
    '''
    Get the adjacency matrix for a tree with nr_layers and a given branching factor. 
    The lowest layer will have 
    '''
    raise ValueError("This function imports networkx, should be removed")
    from networkx import balanced_tree
    
    
    nxG = balanced_tree(branch_factor, nr_layers - 1)

    return _AdjacencyMatrix(nxG)

def radial_out(nr_rays, nodes_per_ray):
    '''
    Get the adjacency matrix for a radial star graph, 
    where there are nr_rays outward rays from the origin, and every ray has nodes_per_ray nodes on it.
    The origin also contains a node that is not counted in the ray but is connected to the first node in every ray.
    '''
    
    edges = []
    
    # Loop through every ray
    nr_qubits = 1
    for ray_nr in range(nr_rays):
        # Loop through all the nodes in this ray. for the k-th ray (starting at 0), 
        # the first node is indexed by k*nodes_per_ray + 1, and the last is (k+1)*nodes_per_ray
        nodes_this_ray = [x for x in range(ray_nr*nodes_per_ray + 1, (ray_nr+1)*nodes_per_ray + 1)]
        edges.append((0,nodes_this_ray[0]))
        for node_nr in range(nodes_per_ray - 1):
            edges.append((nodes_this_ray[node_nr],nodes_this_ray[node_nr+1]))
            
        nr_qubits = max([max(edge[0], edge[1]) for edge in edges]) + 1
    
    return _get_AdjacencyMatrix_from_edgelist(nr_qubits, edges)


def star(nr_nodes):
    '''
    Get the star graph, where the first node is connected to all other nodes, 
    and all other nodes are connected to only the first node.
    '''
    return radial_out(nr_nodes - 1, 1)

def complete(nr_nodes):
    '''
    Get the complete graph on nr_nodes nodes.
    '''
    adjmatrix = _npmatrix(_npones(shape = (nr_nodes, nr_nodes), dtype = 'int') - _npeye(nr_nodes, dtype = 'int'))
    
    return _AdjacencyMatrix(adjmatrix)

def GHZ(nr_nodes):
    '''
    See complete graph.
    '''
    return complete(nr_nodes)

# def get_ring_adjecency(nr_qubits):
#     '''
#     Get the adjecency matrix of a ring cluster state of nr_qubits qubits.
#     '''
#     linear = get_lin_cluster_adjecency(nr_qubits)
    
#     ## Get extra CZ gate for the two end nodes
#     linear[0,nr_qubits-1] = linear[nr_qubits - 1, 0] = 1
    
#     # Return
#     return linear

# def get_connected_ring_adjecency(nr_qubits):
#     '''
#     Get the adjecency matrix of a ring graph state with two opposing nodes connected. 
#     Only works for even nr_qubits. Node 0 is connected to node nr_qubits/2.
#     '''
#     ## Assert nr_qubits is even
#     assert nr_qubits % 2 == 0, f"{nr_qubits} is not even, which I think makes it odd."
    
#     ## Get the ring
#     ring = get_ring_adjecency(nr_qubits)
    
#     ## Connect node 0 with node n/2
#     ring[0,int(nr_qubits/2)] = ring[int(nr_qubits/2),0] = 1
    
#     # Return
#     return ring

# def get_GHZ_fully_conn_adjecency(nr_qubits):
#     '''    
#     Get the adjecency matrix of a GHZ state of nr_qubits qubits, with the fully connected graph.
#     '''
#     return matrix(ones(nr_qubits) - eye(nr_qubits), dtype = int)

# def get_bipartite_connected_line(nr_nodes):
#     '''
#     Get the adjecency matrix of a network where there are nr_nodes nodes, 
#     where each node has 2 qubits, and everyone shares an entangled pair with each other.
#     That is, the network is a line graph where every other edge is not there.
#     '''
#     # Init the empty graph
#     A = zeros((2*nr_nodes, 2*nr_nodes),dtype = int)
    
#     # Add an edge between every node 2n+1 and 2n+2
#     for node in range(nr_nodes - 1):
#         A[2*node + 1, 2*node + 2] = A[2*node + 2, 2*node + 1] = 1
    
#     # Return the adjecency matrix
#     return A

# def get_bipartite_connected_ring(nr_nodes):
#     '''
#     Get the adjecency matrix of the bipartite connected line as described above, but now also connect the first and last node.
#     '''
#     # Get the connected line
#     A = get_bipartite_connected_line(nr_nodes)
    
#     # Get the edge between the first and last qubit
#     A[0 , -1] = A[-1 , 0] = 1
    
#     # Return
#     return A

