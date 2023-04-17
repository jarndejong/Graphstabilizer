from Graphstabilizer.graphs.elementary import AdjacencyMatrix
from networkx.classes.graph import Graph as nxgraph

def check_is_AdjacencyMatrixinstance(adj, raiseorFalse = 'raise'):
    '''
    Checks if the input is an instance of the AdjacencyMatrix class.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if not type(adj) is AdjacencyMatrix:
            raise TypeError(f"Input is not AdjacencyMatrix object but {type(adj)}.")
        else: return
    
    elif raiseorFalse == 'false':
        if not type(adj) is AdjacencyMatrix:
            return False
        else: return True
    
def check_is_networkxinstance(graph, raiseorFalse = 'raise'):
    '''
    Checks if the input is a networkx simple graph.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if type(graph) is not networkx.classes.graph.Graph:
            raise TypeError(f"Input graph is not type networkx.classes.graph.Graph but {type(graph)}.")
        else: return
    
    elif raiseorFalse == 'false':
        if type(graph) is not networkx.classes.graph.Graph:
            return False
        else: return True

def check_are_nodelabels(graphsize, node_labels, raiseorFalse = 'raise'):
    '''
    Checks if the input are node labels, i.e. a list of strings of proper length.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    
    
    if raiseorFalse == 'raise':
        if type(node_labels) is not list:
            raise TypeError(f"Input node labels is not a list but {type(node_labels)}.")
        else:
            if len(node_labels) != graphsize:
                raise ValueError(f"{len(node_labels)} node labels provided but graph size is {graphsize}.") 
            for index, entry in enumerate(node_labels):
                if type(entry) is not str:
                    raise TypeError(f"Node labels {node_labels} contains a non-str character {entry} at index {index}.")
            return True
    
    elif raiseorFalse == 'false':
        if type(node_labels) is not list:
            return False
        else:
            if len(node_labels) != graphsize:
                raise ValueError(f"{len(node_labels)} node labels provided but graph size is {graphsize}.")
            for entry in node_labels:
                if type(entry) is not str:
                    return False
            return True
            
def check_are_nodepositions(graphsize, node_positions, raiseorFalse = 'raise'):
    '''
    Checks if the input are node positions, i.e. a list of strings of proper length.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if type(node_positions) is not list:
            raise TypeError(f"Input node positions is not a list but {type(node_positions)}.")
        else:
            if len(node_positions) != graphsize:
                raise ValueError(f"{len(node_positions)} node positions provided but graph size is {graphsize}.")
            for index, entry in enumerate(node_positions):
                if not ((type(entry) is tuple) or (type(entry) is list) and (len(entry) == 2)):
                    raise TypeError(f"Node positions {node_positions} contains a non-2tuple entry {entry} at index {index}.")
            return True
    
    elif raiseorFalse == 'false':
        if type(node_labels) is not list:
            return False
        else:
            for entry in node_labels:
                if not ((type(entry) is tuple) or (type(entry) is list) and (len(entry) == 2)):
                    return False
            return True        
