## Global imports
# from networkx import from_numpy_array as nxfrom_numpy_array
# from networkx.classes.graph import Graph as nxgraph

from Graphstabilizer.checkers.elementary import check_is_Boolvar, check_is_node_index

from Graphstabilizer.binary.Paulistrings import bit_to_string
from Graphstabilizer.graphs.elementary import bitvector_from_neighbourhood, get_AdjacencyMatrix_from_edgelist
# from Graphstabilizer.graphs.graphstyles import blackonwhite
#%% Graph state class
class Graphstate:
    '''
    Class representing a graph state.
    
    Initialize a graph state from an adjacency matrix, a stabilizer state or a edgelist.
    '''
    ### Init and string functions
    # from Graphstabilizer.checkers.elementary import check_is_naturalnr, check_is_node_index, check_is_Boolvar
    # from Graphstabilizer.checkers.Paulis import check_is_Paulistr
    # from Graphstabilizer.checkers.graphs import check_is_AdjacencyMatrixinstance, check_is_networkxinstance
    
    
    def __init__(self, graph = None):
        from Graphstabilizer.graphs.elementary import AdjacencyMatrix    
        # Set adjacency_matrix to none, possibly overriding it later.
        self.adj = None
        
        
        # Handling of input graph
        if type(graph) is AdjacencyMatrix:
            self.init_from_adjacency_matrix(graph)
        elif type(graph) is StabilizerState:
            self.init_from_stabilizer_state(graph)
        elif type(graph) is list:
            self.init_from_edgelist(graph)
        else: raise ValueError(f"Please provide either an adjacency matrix, a stabilizerstate or an edgelist. This was provided instead: {type(graph)}")
        
        
    
    def __str__(self):
        return f'Graph state of {self.nr_qubits} qubits.'
    
    ### Initialization methods
    def init_from_adjacency_matrix(self, adjacencymatrix):
        '''
        This function will set the adjecency matrix if there is not one yet. 
        The adjacency matrix should be an instance of the AdjacencyMatrix class.
        '''
        if self.adj is not None:
            raise ValueError("Object instance already has adjacency matrix")
        from Graphstabilizer.checkers.graphs import check_is_AdjacencyMatrixinstance    
        check_is_AdjacencyMatrixinstance(adjacencymatrix)

        
        # Init adjacency matrix
        self.adj = adjacencymatrix
        self.nr_qubits = adjacencymatrix.size
    
    def init_from_networkx_graph(self, graph):
        '''
        This function will set the adjecency matrix if there is not one yet from a networkx graph.
        Only the adjecency matrix of the networkx graph is used.
        '''
        # Check validity of input
        from Graphstabilizer.checkers.graphs import check_is_networkxinstance
        from Graphstabilizer.states import AdjacencyMatrix
        check_is_networkxinstance(graph)
        
        # Init adjacency matrix
        adjacencymatrix = AdjacencyMatrix(graph)
         
        self.init_from_adjacency_matrix(self, adjacencymatrix)
        
    
    def init_from_stabilizer_state(self, stabstate):
        '''
        Placeholder for function to initialize a graph state from a stabilizer state.
        '''
        pass
    
    def init_from_edgelist(self, edgelist):
        '''
        Init from an edgelist.
        '''
        nr_qubits = max([qub for edge in edgelist for qub in edge])
        
        self.init_from_adjacency_matrix(get_AdjacencyMatrix_from_edgelist(nr_qubits, edgelist))
    
    #%% Graph operation methods
    def local_complement(self, node: int):
        '''
        Locally complement the graph (state) on the given node. Updates the object instance with the new graph.
        '''
        # Check validity of input
        from Graphstabilizer.graphs.elementary import local_complementation
        node_index = self.__handle_nodeindex_param(node)
        
        self.adj = local_complementation(self.adj, node_index)
    
    def add_edge(self, node1, node2):
        '''
        Add an edge between the two given nodes. Performs nothing if the edge is already there.
        '''
        # Check validity of input
        node1_index = self.__handle_nodeindex_param(node1)
        node2_index = self.__handle_nodeindex_param(node2)
        
        self.adj.add_edge(node1_index, node2_index)
    
    def remove_edge(self, node1, node2):
        '''
        Remove an edge between the two given nodes. Performs nothing if the edge wasn't there.
        '''
        # Check validity of input
        node1_index = self.__handle_nodeindex_param(node1)
        node2_index = self.__handle_nodeindex_param(node2)
        
        self.adj.remove_edge(node1_index, node2_index)
    
    def flip_edge(self, node1, node2):
        '''
        Remove an edge between the two given nodes. Performs nothing if the edge wasn't there.
        '''
        # Check validity of input
        node1_index = self.__handle_nodeindex_param(node1)
        node2_index = self.__handle_nodeindex_param(node2)
        
        self.adj.flip_edge(node1_index, node2_index)
    
    def CZ(self, node1, node2):
        '''
        Perform a CZ gate between the two given nodes. Equivalent to self.flip_edge(node1, node2)
        '''
        self.flip_edge(node1, node2)
        
        
    #%% Info functions    
    def get_neighbourhood(self, node):
        '''
        Return a list of the neighbours of a node.
        Returns empty list if the node has no neighbours.
        '''
        # Check validity of input
        node_index = self.__handle_nodeindex_param(node)
        
        adj_column = self.adj.matrix[:,node_index]
        
        return [index for index in range(self.nr_qubits) if adj_column[index] == 1]
    
    def get_edges(self):
        '''
        Returns a list of the edges, where the edges are tuples of indices.
        '''
        return [(i,j) for i in range(self.nr_qubits) for j in range(i,self.nr_qubits) if self.adj.matrix[i,j] == 1]
    
    def get_edgelist(self):
        '''
        Returns a list of the edges. Same as self.get_edges()
        '''
        return self.get_edges()
    
    @property
    def nr_edges(self):
        '''
        Returns the number of edges in the graph.
        '''
        return len(self.get_edges())
    
    @property
    def nr_nodes(self):
        '''
        Returns the number of nodes in the graph, which is the same as the nr of qubits in the graph state.
        '''
        return self.nr_qubits
    
    @property
    def size(self):
        '''
        Returns the size of the graph, which is the same as the nr of qubits in the graph state.
        '''
        return self.nr_qubits
    
    def contains_edge(self, node1, node2):
        '''
        Returns True if there is an edge between the two nodes, returns false otherwise.
        '''
        # Check validity of input
        node1_index = self.__handle_nodeindex_param(node1)
        node2_index = self.__handle_nodeindex_param(node2)
        
        # Check if node 1 is in neighbourhood of node2 which is unambiguous.
        return node1_index in self.get_neighbourhood(node2_index)
    
    @property
    def identifier(self):
        '''
        Get the identifier for the graph, which is just the binary adjacency matrix as a length-n^2 bitstring
        '''
        return self.adj.identifier
    
    @property
    def is_connected(self):
        '''
        Returns true if the graphstate represents a connected graph.
        See self.adj.is_connected for more details.
        '''
        return self.adj.is_connected
    
    #%% Generators
    @property
    def generators(self):
        '''
        Return the canonical generators as bitvectors.
        '''
        gens = []
        for i in range(self.size):
            gens.append(bitvector_from_neighbourhood(self.size, i, self.get_neighbourhood(i)))
        
        return gens
    
    @property
    def generators_as_string(self):
        '''
        Return the canonical generators as Pauli strings.
        '''
        return [bit_to_string(gen) for gen in self.generators]

    #%% Drawing methods
    
    
    ### Transformations to other objects
    # def to_networkxGraph(self):
    #     '''
    #     Return a networkxGraph from the graph associated with the graph state. Attributes also brought over are:
    #         - None
    #     '''
    #     return nxfrom_numpy_array(self.adj.matrix)
    
    #%% Node additions
    def add_node(self, index = None, return_new = False):
        '''
        Introduce a node to the graphstate at the given index.
        No connections are added.
        '''
        if index is None:
            index = self.size

            
        if return_new:
            from copy import deepcopy
            G = deepcopy(self)
            
            G.add_node(index, return_new = False)
            return G
        
        self.adj.add_node(index)
        self.nr_qubits = self.adj.size
    
    def introduce_connected_node(self, connections = 'all', index = None, return_new = False):
        '''
        Introduce a node to the graphstate, with a given connections type.
        The node is added at the given index, if no is provided will be added as the last node.
        Connections is either a list of nodes to connect to, or 'all', for all nodes.
        
        if return_new is True (default False), return a new graphstate instance instead of updating the current one.
        '''
        if return_new:
            from copy import deepcopy
            G = deepcopy(self)
            G.introduce_connected_node(connections = connections, index = index, return_new = False)
            return G
        
        
        if index is None:
            index = self.nr_nodes
        
        if connections == 'all':
            connections = list(range(self.nr_nodes))
        elif connections == 'none':
            connections = []
        
        # Add the node
        self.add_node(index)
        
        print(connections)
        
        # Perform connections
        for node in connections:
            self.add_edge(index, node)
    
    def add_graphstate(self, G2):
        '''
        Add another graphstate to the current graphstate, by adding the nodes and connections of the other graphstate on top of the current graph state.
        Does not return a new graph state
        '''
        if isinstance(G2, (list, tuple)):
            for G in G2:
                self.add_graphstate(G)
        else:
            for n in range(G2.size):
                self.add_node()
            
            for e in G2.get_edgelist():
                self.add_edge(self.size - G2.size + e[0], self.size - G2.size + e[1])
    
    def merge_graphstates(self, G2):
        '''
        Merge two graphstates by combining the nodes andkeeping the edges of the original graphs intact.
        Return a new Graphstate object.
        '''
        from copy import deepcopy
        G = deepcopy(self)
        
        G.add_graphstate(G2)
        
        return G
        
    #%% Measurements and node deletions
    def delete_node(self, node):
        '''
        Delete a node from a graph (state), by deleting both the row and the column from the adjacency matrix.
        Node should be either an integer that resembles the node indexed by that number, or a valid node label as a string.
        
        '''
        node_index = self.__handle_nodeindex_param(node)

        # Return the adjecency matrix with the column and row deleted
        self.adj.delete_node(node_index)
        self.nr_qubits = self.adj.size
    
    def merge_nodes(self, merged_nodes, leftover_node = None):
        '''
        Merge the nodes in merged_nodes, so they become one node, 
        connected to any of the other nodes that were connected to at least one node in the original collection.
        Leftover_index is the node that stays, so that all other nodes will be deleted. if leftover_index is None (default), the first node will be used.
        '''
        merged_nodes = list(merged_nodes)
        merged_nodes.sort()
        
        # Choose the node that will be left over if none is chosen.
        if leftover_node is None:
            leftover_node = merged_nodes[0]
        
        # Check if the leftover node is actually in the set that will be merged
        if leftover_node not in merged_nodes:
            raise ValueError(f"Leftover node {leftover_node} not in merged nodes collection {merged_nodes}")
        
        # Get the nodes to connect the leftover/merged node to. 
        # These are all the nodes connected to at least one node in the merged set.
        neighbourhood = []
        for node in merged_nodes:
            neighbourhood.extend(self.get_neighbourhood(node))
        
        # Create the edges between the leftover nodes and the other nodes. Only create edges towards outside.
        for node in neighbourhood:
            if node not in merged_nodes:
                self.add_edge(leftover_node, node)
        
        # Delete the other nodes in the merged nodes from the graph
        for node in merged_nodes[:-len(merged_nodes):-1]:
            self.delete_node(node)
    
    def condense_nodes(self, condensed_nodes, leftover_node = None):
        '''
        See self.merge_nodes()
        '''
        self.merge_nodes(merged_nodes = condensed_nodes, leftover_node = leftover_node)
    
    def single_measurement(self, basis: str, node: int, deletion = True):
        '''
        Perform measurement on graph state in a Pauli basis.
        Input:
            basis: str of Pauli basis, i.e. 'I', 'X', 'Y' or 'Z'.
            node: int of the node index to be measured.
            deletion: delete the measured node or not; if not deleted is set to an isolated node. (Default True)
        If 'I' is given as a basis, no measurement is performed, regardless of deletion is True or False the node is not deleted.
        '''
        # from Graphstabilizer.checkers.elementary import check_is_naturalnr, check_is_node_index, check_is_Boolvar
        from Graphstabilizer.checkers.Paulis import check_is_Paulistr
        # Check inputs
        check_is_Paulistr(basis)
        check_is_Boolvar(deletion)
        
        node_index = self.__handle_nodeindex_param(node)
        
        # If basis is 'I' perform no measurement
        if basis == 'I':
            pass
        elif basis == 'Z':
            self.Z_measurement(node = node_index, deletion = deletion)
        elif basis == 'Y':
            self.Y_measurement(node = node_index, deletion = deletion)
        elif basis == 'X':
            self.X_measurement(node = node_index, deletion = deletion)
        
    
    def Z_measurement(self, node, deletion = True):
        '''
        Measure qubits at the same time in the Z basis.
        A Z measurement is the same as deleting all edges adjecent to the node,
            which is the same as setting its row and column to 0.
        '''
        node_index = self.__handle_nodeindex_param(node)

        
        self.adj.matrix[node_index,:] = self.adj.matrix[:,node_index] = 0
        
        if deletion:
            self.delete_node(node_index)
    
    def Y_measurement(self, node, deletion = True):
        '''
        Measure the node in the Y basis.
        A Y measurement is the same as a local complementation on the node, followed by a Z measurement.
        '''
        node_index = self.__handle_nodeindex_param(node)
        
        self.local_complement(node_index)
        self.Z_measurement(node = node_index, deletion = deletion)
        
    def X_measurement(self, node, deletion = True):
        '''
        Measure the node in the X basis.
        A X measurement is the same as a local complementation on a random neighbour of the node,
        followed by a Y measurement on the node, followed by a local complementation on the same neighbour.
        '''
        node_index = self.__handle_nodeindex_param(node)
        
        neighbours = self.get_neighbourhood(node_index)
        
        if len(neighbours) >= 1:
            self.local_complement(neighbours[0])
            self.Y_measurement(node_index, deletion = False)
            self.local_complement(neighbours[0])
            if deletion:
                self.delete_node(node_index)
        else:
            self.Y_measurement(node_index, deletion)
            
    def multiple_measurements(self, measurement_setting: list, deletion = True):
        '''
        Perform multiple measurements on the graph state in Pauli bases.
        Measurements are performed in order of the list.
        The measurement setting list is a list of tuples (node, basis)
        '''
        assert len(measurement_setting) <= self.nr_nodes, "Warning, too many nodes provided to measure."
        
        sorted_measurements = sorted(measurement_setting, key = lambda x: x[0], reverse = True)
        
        for measured_node, basis in sorted_measurements:
            self.single_measurement(basis = basis, node = measured_node, deletion = deletion)
    
    def __handle_nodeindex_param(self, node):
        '''
        Handle a node parameter. This is either an index for the node, or the associated node label. Returns the index.
        '''
        
        if type(node) == int:
            check_is_node_index(self.nr_qubits, node)
            node_index = node
        elif type(node) == str:
            node_index = self.retrieve_node_index(node)
        else:
            raise TypeError(f"Can't find the node labeled {node} because it's not a string or int.")
        return node_index
    
    #%% Returning other graphs
    def complementary_graph(self):
        '''
        Return a new graphstate which is the complementary graph of the given graphstate. 
        The complementary graph of a graph G is the graph with edgeset exactly opposite from G.
        '''
        from itertools import combinations
        new_edgeset = [k for k in combinations(range(self.size), 2) if not self.contains_edge(k[0], k[1])]
        
        return Graphstate(get_AdjacencyMatrix_from_edgelist(self.nr_qubits, new_edgeset))


#%% Stabiliserstate class
class StabilizerState:
    pass