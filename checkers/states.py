from Quantumtools.states import GraphState

def check_is_GraphStateinstance(graph, raiseorFalse = 'raise'):
    '''
    Checks if the input is an instance of the AdjacencyMatrix class.
    
    raiseorFalse is 'raise' or 'false' and decides if the function throws an exception or if it returns false.
    '''
    if not (raiseorFalse == 'raise') or (raiseorFalse == 'false'):
        raise ValueError(f"Input {raiseorFalse} should be either 'raise' or 'false'.")
    
    if raiseorFalse == 'raise':
        if not type(graph) is GraphState:
            raise TypeError(f"Input is not AdjacencyMatrix object but {type(graph)}.")
        else: return
    
    elif raiseorFalse == 'false':
        if not type(graph) is GraphState:
            return False
        else: return True