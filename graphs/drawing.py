# Global imports
from netorkx import draw_nodes

class GraphDrawing:
    def __init__(self, graph):
        '''
        A graph state drawing from a given GraphState.
        '''
        from Quantumtools.checkers.states import check_is_GraphStateinstance
        check_is_GraphStateinstance(graph)
        
        
        
        
        