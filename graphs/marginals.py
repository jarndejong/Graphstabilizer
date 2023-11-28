#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:40:25 2023

@author: jarn
"""
# Global imports
# from numpy.linalg import matrix_rank
from numpy import ix_, zeros, ones

from numpy import ndarray, log2

from itertools import combinations_with_replacement, permutations, product, combinations

# Local imports
from Graphstabilizer.states import Graphstate
from Graphstabilizer.graphs.elementary import get_AdjacencyMatrix_from_edgelist, AdjacencyMatrix
from Graphstabilizer.graphs.graphstyles import GraphStyle, metagraphstyle
from Graphstabilizer.binary.linAlg import get_rank

from Graphstabilizer.graphs.drawing import prepare_graphstatedrawing, draw_nodes, draw_edges

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
    def __init__(self, M:tuple = (0,1,2), G: Graphstate  = None, N = None, identifier = None):
        # If G is None we try to init from the N list
        self.Marginal = M
        
        if G is None:
            if N is None:
                if identifier is None:
                    raise ValueError("Provide either a Graphstate, the neighbourhoods of marginal nodes, or an identifier,")
                elif identifier is not None:
                    self.Metagraph, self.Graphstyle,  = self.init_from_identifier(identifier)
            # Now N is a list
            elif N is not None:
                self.Metagraph, self.Graphstyle,  = self.init_from_marginal_neighbours(N)
            
            # The marginal is by default now [0,1,2]
            
            
            
        
        # Now G is a Graphstate so we initialize from this Graphstate
        else:
            assert len(M) == 3, f"Warning, provided marginal {M} has {len(M)} entries instead of 3."
            for node in M:
                assert check_is_node_index(size = G.nr_qubits, node = node, raiseorFalse = 'false'), f"Warning, marginal {M} with entry {node} is not contained in graph G with {G.nr_qubits} nodes."
            self.Metagraph, self.Graphstyle, self.Metaneighours_as_dict = self.init_from_Graphstate(G, M)
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
    
    #%% Info functions
    @property
    def metaneighbours_as_list(self):
        '''
        '''
        if self.Origingraphstate is None:
            raise ValueError("No original graphstate, so no original metaneighbours.")
        
        return [self.Metaneighours_as_dict['Na'],
                self.Metaneighours_as_dict['Nb'],
                self.Metaneighours_as_dict['Nc'],
                self.Metaneighours_as_dict['Nab'],
                self.Metaneighours_as_dict['Nbc'],
                self.Metaneighours_as_dict['Nac'],
                self.Metaneighours_as_dict['Nabc']]
        
    @property
    def metagraph_rank(self):
        '''
        The rank of the marginal, i.e. the rank of the metagraph in the original graph.
        '''
        return marginal_rank(self.Metagraph, M = (0,1,2))
    
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
        
    #%% Drawing functions
    def draw(self, axis = None, fig = None):
        '''
        Draw the metagraph in the given axis. If no axis is given, create a new figure and axis.
        Returns the figure and axis.
        '''
        if axis is None:
            fig, axis = prepare_graphstatedrawing(self.Graphstyle)
        
        draw_nodes(Graphstyle = self.Graphstyle, axis = axis)
        
        draw_edges(Graphstate = self.Metagraph, Graphstyle = self.Graphstyle, axis = axis)
        
        return fig, axis
        
    #%% Init functions    
    def init_from_marginal_neighbours(self, N: list):
        '''
        Obtain the 3-body Metagraph for a given list of neighbours of the marginal,
        i.e. Na, Nb and Nc, where these are the complete neighbourhoods.
        '''
        a_N = set(N[0])
        b_N = set(N[1])
        c_N = set(N[2])
        
        ab = 1 in a_N
        bc = 2 in b_N
        ac = 2 in a_N
        
        Na = a_N - b_N - c_N
        Nb = b_N.difference(a_N, c_N)
        Nc = c_N.difference(a_N, b_N)
        
        Nab = a_N & b_N - c_N
        Nbc = b_N & c_N - a_N
        Nac = a_N & c_N - b_N
        
        Nabc = a_N & b_N & c_N
        
        # First the inner edges
        identifier = ''.join(['1' if edge else '0' for edge in [ab,bc,ac]])
        
        # Then the outer edges
        identifier += ''.join(['1' if len(neigh) > 0 else '0' for neigh in [Na, Nb, Nc, Nab, Nbc, Nac, Nabc]])
        
        return self.init_from_identifier(identifier)
        
    
    def init_from_identifier(self, identifier: str):
        '''
        Obtain the 3-body marginal Metagraph from an identifier. See Metagraph.identifier for information.
        '''
        # Import
        from Graphstabilizer.graphs.metagraphs.tri import labels, pos, potential_inner_edges, potential_single_edges, potential_double_edges, potential_triple_edges
        
        # Create an edgelist and fill based on the identifier
        edgelist = []
        
        # First the inner edges
        for i in range(3):
            # Inner edges
            if identifier[i] == '1':
                edgelist.append(potential_inner_edges[i])
            # Single edges
            if identifier[i+3] == '1':
                edgelist.append(potential_single_edges[i])
            # Double edges
            if identifier[i+6] == '1':
                edgelist.extend(potential_double_edges[i])
        
        # Triple edge
        if identifier[-1] == '1':
            edgelist.extend(potential_triple_edges[0])
    
        
        
        adj = get_AdjacencyMatrix_from_edgelist(10,edgelist)
        
        G = Graphstate(graph = adj)
        graphstyle = GraphStyle(nr_nodes = 10, template = metagraphstyle, node_labels = labels, node_positions = pos)
        return G, graphstyle
        
    
    def init_from_Graphstate(self, G: Graphstate, M: list):
        '''
        Obtain the 3-body Metagraph for a given Graphstate on the marginal M containing 3 nodes of G.
        '''
        # Imports
        from Graphstabilizer.graphs.metagraphs.tri import potential_single_edges, potential_double_edges, potential_triple_edges, pos
        
        
        
        # Sort M in ascending order and get the complement C
        if type(M) == tuple:
            M = list(M)
        # M.sort()
        Mc = tuple(a for a in range(G.nr_qubits) if a not in M)
            
        labels = [f'${M[0]}$',
                  f'${M[1]}$',
                  f'${M[2]}$',
                  f'$N_{{{M[0]}}}$',
                  f'$N_{{{M[1]}}}$',
                  f'$N_{{{M[2]}}}$',
                  f'$N_{{{M[0]},{M[1]}}}$',
                  f'$N_{{{M[1]},{M[2]}}}$',
                  f'$N_{{{M[0]},{M[2]}}}$',
                  f'$N_{{{M[0]},{M[1]},{M[2]}}}$',
  ]
        
        # Initialize the meta neighbourhoods
        # a, b, c = [], [], []
        Na, Nb, Nc, Nab, Nac, Nbc, Nabc, Ne = [], [], [], [], [], [], [], []
        
        # Fill the meta neighbourhoods
        # First the inner nodes
        
        
        # Then the outer nodes
        for node in Mc:
            # row = G.adj.get_matrix()[ix_([node], M)]
            
            in_a = False
            in_b = False
            in_c = False
            
            
            if node in G.get_neighbourhood(M[0]):
                in_a = True
            if node in G.get_neighbourhood(M[1]):
                in_b = True
            if node in G.get_neighbourhood(M[2]):
                in_c = True
            
            
            
            if not in_a and not in_b and not in_c:
                Ne.append(node)
            elif in_a and not in_b and not in_c:
                Na.append(node)
            elif not in_a and in_b and not in_c:
                Nb.append(node)
            elif not in_a and not in_b and in_c:
                Nc.append(node)
            elif in_a and in_b and not in_c:
                Nab.append(node)
            elif in_a and not in_b and in_c:
                Nac.append(node)
            elif not in_a and in_b and in_c:
                Nbc.append(node)
            elif in_a and in_b and in_c:
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
    
        
        # Then the outer edges
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
        
        G = Graphstate(graph = adj)
        graphstyle = GraphStyle(nr_nodes = 10, template = metagraphstyle, node_labels = labels, node_positions = pos)
        return G, graphstyle, {'Na' : Na, 'Nb' : Nb, 'Nc' : Nc, 'Nab' : Nab, 'Nac' : Nac, 'Nbc' : Nbc, 'Nabc' : Nabc, }
    
    #%% Info functions
    @property
    def identifier(self):
        '''
        Get the identifier for the metagraph. This is a 10-bit string, with every entry being 0 or 1 if that (meta)-edge is connected to or not.
        This starts with the internal edges, and subsequently the 7 outside neighbors are added.
        '''
        
        ## First the internal edges
        identifier = ['1' if edge else '0' for edge in [self.Metagraph.contains_edge(0,1), self.Metagraph.contains_edge(2,1), self.Metagraph.contains_edge(0,2)]]
        
        ## Then the external edges
        identifier.extend(['1' if len(self.Metagraph.get_neighbourhood(index)) >= 1 else '0' for index in range(3,10)])
        
        return ''.join(identifier)
    
#%% Four body metagraph
class MetaGraphFour:
    '''
    Initialize a 4-body MetaGraph. It can be initialized from a given 
    '''
    def __init__(self, G, N, M: tuple = (0,1,2,3)):
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
                meta = MetaGraphThree(G = graphstate, M = marginal)
                tensor_entry = meta.get_number_of_populated_metaneighbours()
        
        # Enter this in all the entries in the tensor associated with this marginal
        # These are exactly all the permutations of the marginal node list
        for index in permutations(iterable = marginal):
            T[index] = tensor_entry
    
    # Return the tensor
    return T

def metagraph_tensor(graphstate: Graphstate, metagraphgroups, indexing) -> ndarray:
    '''
    Calculate the tensor for a metagraph
    '''
    T = ones(shape = (graphstate.nr_qubits,)*3, dtype = 'int')
    # Loop through every possible marginal selection
    # for marginal in product(range(graphstate.nr_qubits), repeat = 3):
    #     if marginal[0] == marginal[1]:
    #         continue
    #     elif marginal[0] == marginal[2]:
    #         continue
    #     elif marginal[1] == marginal[2]:
    #         continue
    for M in combinations_with_replacement(range(graphstate.nr_qubits), r = 3):
        if len(set(M)) == 1:
            
            T[M] = 0
            # print(T)
            # break
        elif len(set(M)) == 2:
            for marginal in permutations(M):
                T[M] = int(log2(marginal_rank(graphstate, marginal)))
        else:
            for marginal in permutations(M):
                # Compute the metagraph and the identifier
                meta = MetaGraphThree(G = graphstate, M = marginal)
                identifier = meta.identifier
                
                # Loop through every group and check if the metagraph is in this group
                for groupnr, group in enumerate(metagraphgroups):
                    if identifier in group:
                        # Now add the resulting number in the tensor
                        T[marginal] = indexing[groupnr]
                        break
    
    # Return the tensor
    return T
            