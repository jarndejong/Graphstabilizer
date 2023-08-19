## Global imports
from networkx import from_numpy_array as nxfrom_numpy_array
from networkx.classes.graph import Graph as nxgraph

from Graphstabilizer.checkers.elementary import check_is_node_index, check_is_Boolvar
from Graphstabilizer.checkers.graphs import check_are_nodelabels, check_are_nodepositions

from Graphstabilizer.graphs.graphstyles import blackonwhite
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
    
    
    def __init__(self, graph = None, node_labels = None, node_positions = None, graphstyle = None):
        from Graphstabilizer.graphs.elementary import AdjacencyMatrix    
        # Set adjacency_matrix to none, possibly overriding it later.
        self.adj = None
        
        
        # Handling of input graph
        if type(graph) is AdjacencyMatrix:
            self.init_from_adjacency_matrix(graph)
        elif type(graph) is StabilizerState:
            self.init_from_stabilizer_state(graph)
        elif type(graph) is nxgraph:
            self.init_from_networkx_graph(graph)
        else: raise ValueError(f"Please provide either an adjacency matrix, a STabilizerState or a networkXgraph. This was provided instead: {type(graph)}")
        
        # Handling of node labels
        if not node_labels is None:
            check_are_nodelabels(self.nr_qubits, node_labels)
        self.node_labels = node_labels
        
        # Handling of node positions
        if not node_positions is None:
            check_are_nodepositions(self.nr_qubits, node_positions)
        self.node_positions = node_positions
        
        # Set the graphstyle, if none is passed, set the standard blackonwhite
        if graphstyle == None:
            graphstyle = blackonwhite
        self.graphstyle = graphstyle
    
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
        if self.node_labels is not None:
            return [(self.node_labels[i],self.node_labels[j]) for i in range(self.nr_qubits) for j in range(i,self.nr_qubits) if self.adj.matrix[i,j] == 1]
        else:
            return [(i,j) for i in range(self.nr_qubits) for j in range(i,self.nr_qubits) if self.adj.matrix[i,j] == 1]
    
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
    
    
    #%% Node label methods
    def retrieve_node_index(self, nodelabel):
        '''
        Retrieve the (current) index of the node associated with the node label.
        '''
        return self.node_labels.index(nodelabel)
    
    
    #%% Drawing methods
    def labels_to_graphx_format(self):
        '''
        Return a dictionary of the current node labels, with keys the index of the element in the list.
        '''
        labels_dict = {}
        for index, label in enumerate(self.node_labels):
            labels_dict[index] = label
        return labels_dict
    
    ### Transformations to other objects
    def to_networkxGraph(self):
        '''
        Return a networkxGraph from the graph associated with the graph state. Attributes also brought over are:
            - None
        '''
        return nxfrom_numpy_array(self.adj.matrix)
    
    def get_extreme_coordinates(self):
        '''
        Get the outermost coordinates of the nodes in both x and y direction. Returns two lists:
            x_lim: [min, max]
            y_lin: [min, max]
        '''
        x, y = zip(*self.node_positions)
        return [min(x), max(x)], [min(y), max(y)]
    
    def get_extreme_node_indices(self) -> [list, list]:
        '''
        Get the indices of the outermost nodes in both x and y direction. Returns two lists:
            x_index: [xminindex, xmaxindex]
            y_index: [yminindex, ymaxindex]
            
            where xminindex is the index of the nodes with the lowest x coordinate
        '''
        x, y = zip(*self.node_positions)
        return [x.index(min(x)), x.index(max(x))], [y.index(min(y)), y.index(max(y))]
    
    def get_node_positions_and_radii(self, selection) -> list:
        '''
        Get the positions for a given selection of nodes.
        Always returns a list, even if only one node given.
        '''
        node_positions = []
        
        for node in selection:
            check_is_node_index(self, node)
            node_positions.append()
    
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
        
        if self.node_labels is not None:
            self.node_labels.pop(node_index)
        if self.node_positions is not None:
            self.node_positions.pop(node_index)
    
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
    
    #### Internal functions
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