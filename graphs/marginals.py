#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:40:25 2023

@author: jarn
"""
# Global imports
from numpy.linalg import matrix_rank
from numpy import ix_, diag, zeros

from numpy import ndarray

from itertools import combinations_with_replacement, permutations, combinations

# Local imports
from Graphstabilizer.states import Graphstate
from Graphstabilizer.graphs.elementary import get_AdjacencyMatrix_from_edgelist, AdjacencyMatrix
from Graphstabilizer.graphs.graphstyles import metagraphstyle
from Graphstabilizer.binary.linAlg import get_rank

from Graphstabilizer.checkers.elementary import check_is_node_index, check_is_Boolvar


def marginal_rank(graphstate: Graphstate, M: tuple):
    '''
    For a given graph state G, its marginal on the set M is tr_C[G], with C the complement of M. 
    
    This function returns the rank of the reduced density matrix, which is a power of two, 
    starting from 2^0 (for pure reduced states) until 2^Ms with Ms the size of the marginal, for completely mixed marginals.
    
    The rank of the marginal has to do with the number of elements in the reduced stabilizer, by an inverse proportion.
    
    The number of elements in the reduced stabilizer is 2^b_m, where b_m is the dimension of the nullspace of the matrix A_{M,C}, 
    i.e. the lower left block of the blockmatrix A, the adjacency matrix:
    
    A = [ A_{M,M} | A_{C,M} ]
        [ A_{M,C} | A_{C,C} ]
        
    Where it is implied w.l.o.g. that M are the first nodes. (The function itself actually doesn't assume this)
    The function calculates the nullity as abs(M) - rank(A_{M,C}) by the rank-nullity theorem
    
    The rank of the marginal is then exactly 2**(abs(M) - b_m) = 2**rank(A_{M,C})                                                          
                                                              
    '''
    Mc = tuple(a for a in range(graphstate.nr_qubits) if a not in M)

    # rank = matrix_rank(graphstate.adj.get_matrix()[ix_(Mc, M)])
    rank = get_rank((graphstate.adj.get_matrix()[ix_(Mc, M)]))
    
    return 2**(rank)

def marginal_nullspace_rank_after_Z(graphstate: Graphstate, M: tuple, k: int):
    '''
    Obtain the rank of the marginal state 
    '''
    
    Mc = tuple(a for a in range(graphstate.nr_qubits) if a not in M + (k,))
    
    # rank = matrix_rank(graphstate.adj.get_matrix()[ix_(Mc, M)])
    rank = get_rank((graphstate.adj.get_matrix()[ix_(Mc, M)]))
    
    return 2**(rank)


#%% 3-body meta-graph
class MetaGraphThree:
    '''
    Initialize a 3-body MetaGraph. It can be initialized from a given 
    '''
    def __init__(self, G: Graphstate  = None, N = None, M: tuple = (0,1,2)):
        # If G is None we try to init from the N list
        if G is None:
            if N is None:
                raise ValueError("Provide either a Graphstate or a list of metaneighbours")
            # Now N is a list
            assert len(N) == 10, f"Metaneighbourlist {N} has {len(N)} entries which is not 10."
            
            # The marginal is by default now [0,1,2]
            self.Marginal = M
            
            
        
        # Now G is a Graphstate so we initialize from this Graphstate
        else:
            assert len(M) == 3, f"Warning, provided marginal {M} has {len(M)} entries instead of 3."
            for node in M:
                assert check_is_node_index(size = G.nr_qubits, node = node, raiseorFalse = 'false'), f"Warning, marginal {M} with entry {node} is not contained in graph G with {G.nr_qubits} nodes."
            self.Metagraph, self.Metaneighours = self.init_from_Graphstate(G, M)
            self.Marginal = M
        
        
        self.Origingraphstate = G
        
    #%% Meta-operations
    def map_metaneighbours_zero(self):
        '''
        Remove all connections between outside metaneighbours that might have been created.
        '''
        A = self.Metagraph.adj.get_matrix()
        A[ix_(range(3,10),range(3,10))] = zeros((7,7), dtype = 'int')
        self.Metagraph = Graphstate(graph = AdjacencyMatrix(A), 
                                    node_labels = self.Metagraph.node_labels, 
                                    node_positions = self.Metagraph.node_positions, 
                                    graphstyle = self.Metagraph.graphstyle)
    
    def local_complementation_internal(self, node):
        '''
        Perform a local complementation on the given internal node.
        '''
        assert node in [0,1,2], f"Warning, not an internal node given. Node provided is {node} but should be 0,1 or 2"
        
        # Perform local complementation
        self.Metagraph.local_complement(node)
        
        # Map out edges to zero
        self.map_metaneighbours_zero()
        
        # Return a new Metagraph
        NewMeta =  MetaGraphThree(G = self.Metagraph, M = self.Marginal)
        
        self.Metagraph = NewMeta.Metagraph
        
    #%% Counting functions
    def get_number_of_populated_metaneighbours(self) -> int:
        '''
        Obtain the number of metaneighbours that are populated, i.e. that are connected to the marginal.
        '''
        nr_metaneighbours = 0
        
        for single in [3,4,5]:
            nr_metaneighbours += len(self.Metagraph.get_neighbourhood(single))
        for double in [6,7,8]:
            nr_metaneighbours += len(self.Metagraph.get_neighbourhood(double))/2
        nr_metaneighbours += len(self.Metagraph.get_neighbourhood(9))/3
        
        assert nr_metaneighbours == int(nr_metaneighbours), f"{nr_metaneighbours} is not an int."
        
        return int(nr_metaneighbours)
        
        
    #%% Init functions    
    def init_from_metaneighbourlist(self, N: list):
        '''
        Obtain the 3-body Metagraph for a given list of metaneighbourhoods. This list is:
            N = [a, b, c, Na, Nb, Nc, Nab, Nbc, Nac, Nabc]
            
            Where 
                a, b, c are the internal neighbourhoods of node a, b and c as list of indexes
                Na, Nb, Nc are 
        '''
        raise NotImplementedError()
    
    def init_from_Graphstate(self, G: Graphstate, M: list):
        '''
        Obtain the 3-body Metagraph for a given Graphstate on the marginal M containing 3 nodes of G.
        '''
        # Imports
        from Graphstabilizer.graphs.metagraphs.tri import potential_single_edges, potential_double_edges, potential_triple_edges, labels, pos
        
        
        
        # Sort M in ascending order and get the complement C
        if type(M) == tuple:
            M = list(M)
        M.sort()
        Mc = tuple(a for a in range(G.nr_qubits) if a not in M)
            
        
        
        # Initialize the meta neighbourhoods
        # a, b, c = [], [], []
        Na, Nb, Nc, Nab, Nac, Nbc, Nabc, Ne = [], [], [], [], [], [], [], []
        
        # Fill the meta neighbourhoods
        # First the inner nodes
        
        # Then the outer nodes
        for node in Mc:
            # row = G.adj.get_matrix()[ix_([node], M)]
            
            
            
            n_string = ''.join([str(G.adj.get_matrix()[ix_([node], M)][0,i]) for i in range(3)])
            
            
            if n_string == '000':
                Ne.append(node)
            elif n_string == '100':
                Na.append(node)
            elif n_string == '010':
                Nb.append(node)
            elif n_string == '001':
                Nc.append(node)
            elif n_string == '110':
                Nab.append(node)
            elif n_string == '101':
                Nac.append(node)
            elif n_string == '011':
                Nbc.append(node)
            elif n_string == '111':
                Nabc.append(node)
        
        # Obtain the edges of the metagraph
        edgelist = []
        # First the inner edges
        a_neighbourhood = G.get_neighbourhood(M[0])
        b_neighbourhood = G.get_neighbourhood(M[1])
        
        if M[1] in a_neighbourhood:
            edgelist.append((0,1))
        if M[2] in a_neighbourhood:
            edgelist.append((0,2))
        if M[2] in b_neighbourhood:
            edgelist.append((1,2))
    
        
        # The the outer edges
        if len(Na) >= 1:
            edgelist.append(potential_single_edges[0])
        if len(Nb) >= 1:
            edgelist.append(potential_single_edges[1])
        if len(Nc) >= 1:
            edgelist.append(potential_single_edges[2])
        if len(Nab) >= 1:
            edgelist.extend(potential_double_edges[0])
        if len(Nbc) >= 1:
            edgelist.extend(potential_double_edges[1])
        if len(Nac) >= 1:
            edgelist.extend(potential_double_edges[2])
        if len(Nabc) >= 1:
            edgelist.extend(potential_triple_edges[0])
        
        # Create the metagraph
        adj = get_AdjacencyMatrix_from_edgelist(10,edgelist)
        
        G = Graphstate(graph = adj, node_labels = labels, node_positions = pos, graphstyle = metagraphstyle)
        
        return G, {'Na' : Na, 'Nb' : Nb, 'Nc' : Nc, 'Nab' : Nab, 'Nac' : Nac, 'Nbc' : Nbc, 'Nabc' : Nabc, }
    
#%% Four body metagraph
class MetaGraphFour:
    '''
    Initialize a 4-body MetaGraph. It can be initialized from a given 
    '''
    def __init__(self, G: Graphstate | None = None, N: list | None = None, M: tuple = (0,1,2,3)):
        # If G is None we try to init from the N list
        if G is None:
            if N is None:
                raise ValueError("Provide either a Graphstate or a list of metaneighbours")
            # Now N is a list
            assert len(N) == 19, f"Metaneighbourlist {N} has {len(N)} entries which is not 19."
            
            # The marginal is by default now [0,1,2,3]
            self.Marginal = M
            
            
        
        # Now G is a Graphstate so we initialize from this Graphstate
        else:
            assert len(M) == 4, f"Warning, provided marginal {M} has {len(M)} entries instead of 4."
            for node in M:
                assert check_is_node_index(size = G.nr_qubits, node = node, raiseorFalse = 'false'), f"Warning, marginal {M} with entry {node} is not contained in graph G with {G.nr_qubits} nodes."
            self.Metagraph, self.Metaneighours = self.init_from_Graphstate(G, M)
            self.Marginal = M
        
        
        self.Origingraphstate = G
        
    #%% Meta-operations
    def map_metaneighbours_zero(self):
        '''
        Remove all connections between outside metaneighbours that might have been created.
        '''
        A = self.Metagraph.adj.get_matrix()
        A[ix_(range(4,19),range(4,19))] = zeros((15,15), dtype = 'int')
        self.Metagraph = Graphstate(graph = AdjacencyMatrix(A), node_labels = self.Metagraph.node_labels, node_positions = self.Metagraph.node_positions, graphstyle = self.Metagraphstyle.graphstyle)
    
    def local_complementation_internal(self, node):
        '''
        Perform a local complementation on the given internal node.
        '''
        assert node in [0,1,2,3], f"Warning, not an internal node given. Node provided is {node} but should be 0,1,2 or 3"
        
        # Perform local complementation
        self.Metagraph.local_complement(node)
        
        # Map out edges to zero
        self.map_metaneighbours_zero()
        
        # Return a new Metagraph
        NewMeta =  MetaGraphThree(G = self.Metagraph, M = self.Marginal)
        
        self.Metagraph = NewMeta.Metagraph
        
    
    #%% Init functions    
    def init_from_metaneighbourlist(self, N: list):
        '''
        Obtain the 4-body Metagraph for a given list of metaneighbourhoods. This list is:
            N = [a, b, c, d, Na, Nb, Nc, Nd, Nab, Nbc, Nac, Nabc]
            
            Where 
                a, b, c are the internal neighbourhoods of node a, b and c as list of indexes
                Na, Nb, Nc are 
        '''
        raise NotImplementedError()
    
    def init_from_Graphstate(self, G: Graphstate, M: list):
        '''
        Obtain the 3-body Metagraph for a given Graphstate on the marginal M containing 3 nodes of G.
        '''
        # Imports
        from Graphstabilizer.graphs.metagraphs.tri import potential_single_edges, potential_double_edges, potential_triple_edges, labels, pos
        
        
        
        # Sort M in ascending order and get the complement C
        M.sort()
        Mc = tuple(a for a in range(G.nr_qubits) if a not in M)
            
        
        
        # Initialize the meta neighbourhoods
        Na, Nb, Nc, Nab, Nac, Nbc, Nabc, Ne = [], [], [], [], [], [], [], []
        
        # Fill the meta neighbourhoods
        # First the inner nodes
        
        # Then the outer nodes
        for node in Mc:
            
            n_string = ''.join([str(G.adj.get_matrix()[ix_([node], M)][0,i]) for i in range(3)])
            
            
            if n_string == '000':
                Ne.append(node)
            elif n_string == '100':
                Na.append(node)
            elif n_string == '010':
                Nb.append(node)
            elif n_string == '001':
                Nc.append(node)
            elif n_string == '110':
                Nab.append(node)
            elif n_string == '101':
                Nac.append(node)
            elif n_string == '011':
                Nbc.append(node)
            elif n_string == '111':
                Nabc.append(node)
        
        # Obtain the edges of the metagraph
        edgelist = []
        # First the inner edges
        a_neighbourhood = G.get_neighbourhood(M[0])
        b_neighbourhood = G.get_neighbourhood(M[1])
        
        if M[1] in a_neighbourhood:
            edgelist.append((0,1))
        if M[2] in a_neighbourhood:
            edgelist.append((0,2))
        if M[2] in b_neighbourhood:
            edgelist.append((1,2))
    
        
        # The the outer edges
        if len(Na) >= 1:
            edgelist.append(potential_single_edges[0])
        if len(Nb) >= 1:
            edgelist.append(potential_single_edges[1])
        if len(Nc) >= 1:
            edgelist.append(potential_single_edges[2])
        if len(Nab) >= 1:
            edgelist.extend(potential_double_edges[0])
        if len(Nbc) >= 1:
            edgelist.extend(potential_double_edges[1])
        if len(Nac) >= 1:
            edgelist.extend(potential_double_edges[2])
        if len(Nabc) >= 1:
            edgelist.extend(potential_triple_edges[0])
        
        # Create the metagraph
        adj = get_AdjacencyMatrix_from_edgelist(10,edgelist)
        
        G = Graphstate(graph = adj, node_labels = labels, node_positions = pos, graphstyle = metagraphstyle)
        
        return G, {'Na' : Na, 'Nb' : Nb, 'Nc' : Nc, 'Nab' : Nab, 'Nac' : Nac, 'Nbc' : Nbc, 'Nabc' : Nabc, }
    

#%% Tensors
def marginal_tensor(graphstate: Graphstate, marginalsize: int, inv: str  = 'marginal_rank') -> ndarray:
    '''
    For a marginal size m, obtain the m-fold tensor with n 
    '''
    
    assert marginalsize < graphstate.nr_qubits, f"There are no {marginalsize}-marginals for a state with {graphstate.nr_qubits} qubits."
    
    T = zeros(shape = (graphstate.nr_qubits,)*marginalsize, dtype = 'int')
    
    
    # Loop through every possible marginal selection
    for marginal in combinations_with_replacement(iterable = range(graphstate.nr_qubits), r = marginalsize):
    # for marginal in combinations(iterable = range(graphstate.nr_qubits), r = marginalsize):

        # For this marginal, calculate the desired property
        if inv == 'marginal_rank':
            tensor_entry = marginal_rank(graphstate, marginal)
        elif inv == 'metaneighbourhood_size':
            if marginalsize == 3:
                meta = MetaGraphThree(graphstate, marginal)
                tensor_entry = meta.get_number_of_populated_metaneighbours()
        
        # Enter this in all the entries in the tensor associated with this marginal
        # These are exactly all the permutations of the marginal node list
        for index in permutations(iterable = marginal):
            T[index] = tensor_entry
    
    # Return the tensor
    return T
