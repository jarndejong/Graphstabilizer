# Global imports
# from netorkx imporxt draw_nodes

class GraphDrawing:
    def __init__(self, graph):
        '''
        A graph state drawing from a given GraphState.
        '''
        from Quantumtools.checkers.states import check_is_GraphStateinstance
        check_is_GraphStateinstance(graph)
#%%

graphstyle = {
    'with_labels'           : True,
    'node_radius'           : 0.1,
    'node_color'             : 'k',
    'node_edgecolor'         : 'w',
    'node_edgewidth'         : 2,
    'label_color'            : 'w',
    'edgecolor'              : 'w',
    'edgewidth'              : 2,
    }

def prepare_figure(state):
    '''
    Prepare 
    '''

def draw_nodes(state, figure, axis, graphstyle):
    '''
    Draw the nodes
    '''
    
    from plt import Circle
    # Copy the 
    labels = state.node_labels
    if labels is None:
        labels = [f'{i}' for i in range(state.nr_qubits)]
        
    
    for node_index, position in enumerate(state.node_positions):
        circle = Circle(position, radius = graphstyle['node_radius'], facecolor = graphstyle['node_color'], 
                            edgecolor = graphstyle['node_edgecolor'], linewidth = graphstyle['node_edgewidth'])
        
        
        axis.add_patch(circle)
        
        if graphstyle['with_labels']:
            axis.text(x = position[0], y = position[1], s = labels[node_index], 
                      fontsize=16, ha="center", va="center", usetex = True, color = graphstyle['label_color'])



def draw_edges(state, figure, axis, graphstyle):
    '''
    Draw the edges 
    '''
    edges = state.get_edges()

def draw_edge(figure, state, edge, graphstyle):
    '''
    Draw the edge from node to node
    '''



#%% Helper functions

def _calculate_node_edge_dist(node, p1, p2):
    '''
    Calculate the distance from a node to 
    '''

def _calculate_arc_params(pos_1, pos_2, angle = 15, offset = 0, anglecoor = 'deg'):
    '''
    Calculate the parameters needed for a arc drawing (i.e. edge) from a node to a node.
    The arc is a circle segment from pos_1 to pos_2, 
    where the outgoing line makes the given angle (in radians) with the straight line from pos_1 to pos_2.
    The offset is the relative length on both ends that is not drawn.
    Returns:
        [xr,yr], r, [t1,t2]
        where
        xr,yr: the center of the circle
        r: the radius of the circle
        t1, t2: the angles of the beginning and end of the circle arc in radians
    '''
    # Import necessary math functions
    from math import copysign, tan, atan, sin, sqrt, pi
    if angle == 0:
        angle = 0.001
    elif angle < 0:
        return _calculate_arc_params(pos_2, pos_1, angle = -angle, offset = offset, anglecoor = anglecoor)
    
    
    conv = 180/pi
    # Convert to radian if necessary
    if anglecoor == 'deg':
        angle = angle/conv
    elif anglecoor != 'rad':
        raise ValueError(f"Warning, {anglecoor} is not a valid coordinate system, pick 'deg' or 'rad'.")
        
    # Unpack the positions as x and y coordinates
    x1, y1 = pos_1
    
    x2, y2 = pos_2
    
    # Calculate the distance between the two points
    d = sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Calculate the radius
    r = d/(2*sin(angle))
    
    # Calculate the midpoints

    xr = (x1 + x2 + 1/tan(angle)*(y2 - y1))/2
    yr = (y1 + y2 + 1/tan(angle)*(x1 - x2))/2
    
    # Calculate the angles
    theta1 = atan((y1 - yr)/(x1 - xr)) + pi/2 - copysign(pi/2,x1 - xr)
    theta2 = atan((y2 - yr)/(x2 - xr)) + pi/2 - copysign(pi/2,x2 - xr)
    
    # Add the offset
    theta1, theta2 = (1-offset) * theta1 + offset*theta2, (1-offset) * theta2 + offset*theta1
    
    if anglecoor == 'deg':
        return [xr, yr], r, [conv*theta2, conv*theta1]
    
    return [xr, yr], r, [theta2, theta1]