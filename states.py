## Global imports
# from networkx import from_numpy_array as nxfrom_numpy_array
# from networkx.classes.graph import Graph as nxgraph

from Graphstabilizer.checkers.elementary import check_is_Boolvar, check_is_node_index

# from Graphstabilizer.graphs.graphstyles import blackonwhite
#%% Graph state class
class Graphstate:
    '''
    Class representing a graph state.
    
    Initialize a graph state from an adjacency matrix, a stabilizer state or a networkXgraph.
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
        # elif type(graph) is nxgraph:
        #     self.init_from_networkx_graph(graph)
        else: raise ValueError(f"Please provide either an adjacency matrix, a STabilizerState or a networkXgraph. This was provided instead: {type(graph)}")
        
        
    
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
    
    def contains_edge(self, node1, node2):
        '''
        Returns True if there is an edge between the two nodes, returns false otherwise.
        '''
        # Check validity of input
        node1_index = self.__handle_nodeindex_param(node1)
        node2_index = self.__handle_nodeindex_param(node2)
        return node1_index in self.get_neighbourhood(node2_index)
    
    @property
    def identifier(self):
        '''
        Get the identifier for the graph, which is just the binary adjacency matrix as a length-n^2 bitstring
        '''
        return self.adj.identifier
    
    #%% Drawing methods
    
    
    ### Transformations to other objects
    # def to_networkxGraph(self):
    #     '''
    #     Return a networkxGraph from the graph associated with the graph state. Attributes also brought over are:
    #         - None
    #     '''
    #     return nxfrom_numpy_array(self.adj.matrix)
    
    
    
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
        

    
    def single_measurement(self, basis: str, node: int, deletion = True):
        '''
        Perform measurement on graph state in a Pauli basis.
        Input:
            basis: str of Pauli basis, i.e. 'I', 'X', 'Y' or 'Z'.
            node: int of the node index to be measured.
            deletion: delete the measured node or not; if not deleted is set to an isolated node. (Default True)
        If 'I' is given as a basis, no measurement is performed, irregardless of deletion is True or False the node is not deleted.
        '''
        # from Graphstabilizer.checkers.elementary import check_is_naturalnr, check_is_node_index, check_is_Boolvar
        from Graphstabilizer.checkers.Paulis import check_is_Paulistr
        # Check inputs
        check_is_Paulistr(basis)
        check_is_Boolvar(deletion)
        
        node_index = self.__handle_nodeindex_param(node)
        
        # If basis is 'I' perform no measurement
        if basis == 'I':
            return
        elif basis == 'Z':
            self.Z_measurement(node = node_index, deletion = deletion)
        elif basis == 'Y':
            self.Y_measurement(node = node_index, deletion = deletion)
        elif basis == 'X':
            self.X_measurement(node = node_index, deletion = deletion)
        return
    
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
class StabilizerState:
    pass