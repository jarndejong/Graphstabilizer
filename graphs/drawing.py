# Global imports
# from netorkx imporxt draw_nodes
from matplotlib.patches import Circle, Arc
from matplotlib.pyplot import figure
from numpy import mat
from numpy.linalg import norm
from math import copysign, tan, atan, sin, sqrt, pi

#%%

graphstyle = {
    'with_labels'               : True,
    'node_radius'               : 0.15,
    'node_color'                : 'k',
    'node_edgecolor'            : 'w',
    'node_edgewidth'            : 2,
    'label_color'               : 'w',
    'edge_color'                : 'w',
    'edge_width'                : 2,
    'edge_offset'               : 0.2,
    'edge_fontsize'             : 16,
    'tex_for_labels'            : True,
    'figure_multiplier'         : 2,
    'figure_tightness'          : 0.1,
    'figure_offsetequal'        : True,
    'background_color'          : 'k',
    }
#%% Init functions
def calculate_axes_limits(xminmax, yminmax, node_radius, tightness = 0.05, equaloffset = True):
    '''
    Calculate x and y axes limits for a graph picture from the x and y min and max coordinates.
    The limits are calculated as such:
        For x, let d be the distance from max to min.
        Then the limits are the xmin and xmax minus cq. plus the offset, and similarly for y
        The offset is node_radius + tightness*d
        
    If equaloffset = True, the offset for both x and y is set to the max of the two
    '''
    lims = []
    offsets = []
    for coor in (xminmax, yminmax):
        d = coor[1] - coor[0] + 2*node_radius
        offsets.append(node_radius + tightness*d)
            
    for index, coor in enumerate((xminmax, yminmax)):
        if equaloffset:
            lims.append([coor[0] - max(offsets[0], offsets[1]), coor[1] + max(offsets[0], offsets[1])])
        else:
            lims.append([coor[0] - offsets[index], coor[1] + offsets[index]])
    return lims

def prepare_graphstatedrawing(Graphstate, graphstyle, axislimits = None, figure_multiplier = 1):
    '''
    Prepare a Graphstate figure.
    Return a figure and an axis object
    '''
    # If no axislimits are given, calculate them
    if axislimits is None:
        xminmax, yminmax = Graphstate.get_extreme_coordinates()

        axislimits = calculate_axes_limits(xminmax, yminmax, graphstyle['node_radius'], tightness = graphstyle['figure_tightness'], equaloffset = graphstyle['figure_offsetequal'])
        
    # Unpack the axis limits
    xlim, ylim = axislimits[0], axislimits[1]
    
    fig = figure(figsize = (figure_multiplier*(xlim[1] - xlim[0]), figure_multiplier*(ylim[1] - ylim[0])))

    axis = fig.add_subplot(111)
    print(xlim, ylim)
    
    axis.set(xlim=xlim, ylim=ylim, aspect=1)
    axis.axis('off')
    # axis.set(xlim=xlim, ylim=ylim)

    axis.margins(x=0, y=0)
    
    # fig.bbox = 'tight'
    fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    
    # Set the background color
    fig.patch.set_facecolor(graphstyle['background_color'])
    axis.patch.set_facecolor(graphstyle['background_color'])

    
    return fig, axis

def draw_nodes(Graphstate, axis, graphstyle):
    '''
    Draw the nodes of the Graphstate in the given axis.
    '''
    labels = Graphstate.node_labels
    if labels is None:
        labels = [f'{i}' for i in range(Graphstate.nr_qubits)]
    # Plot every node iteratively
    for node_index, position in enumerate(Graphstate.node_positions):
        # Make a circle at the node position
        circle = Circle(position, radius = graphstyle['node_radius'], 
                        facecolor = graphstyle['node_color'], 
                        edgecolor = graphstyle['node_edgecolor'],
                        linewidth = graphstyle['node_edgewidth'])
        
        # Add the node to the axis
        axis.add_patch(circle)
        
        # Make the label if the graphstyle wants it
        if graphstyle['with_labels']:
            axis.text(x = position[0], y = position[1], 
                      s = labels[node_index], 
                      fontsize = graphstyle['edge_fontsize'],
                      ha="center",
                      va="center_baseline", 
                      usetex = graphstyle['tex_for_labels'], 
                      color = graphstyle['label_color'])

def draw_edges(Graphstate, axis, graphstyle, edgesmap = None):
    '''
    Draw the edges of the Graphstate in the given axis.
    The edgesmap is a list with an edgemap instruction for every edge.
    If no edgesmap is given, the edge will first tried to be drawn straight, and then alternatingly through all 
    An edgemap looks like:
        edgemap =  {'type' : 'straight'}
        or 
        edgemap = {'type'  : 'arc',
                   'params': [angle, anglecoor, direction]}
    '''
    if edgesmap is None:
        # Init an arc direction so that the arc directions can get flipped (from positive to negative angle)
        flip = True
        # Now loop through all edges
        for edge in Graphstate.get_edges():
            flip = draw_edge(Graphstate, axis, graphstyle, edge, arcflip = flip)
    else:
        for edge, edgemap in zip(Graphstate.get_edges(), edgesmap):
            draw_edge(Graphstate, axis, graphstyle, edge, edgemap)
        

def draw_edge(Graphstate, axis, graphstyle, edge, edgemap = None, arcflip = None):
    '''
    Draw the edge for the given Graphstate and fiven edgemap. If edgemap is 
    '''
    # Get indices instead of labels
    if Graphstate.node_labels is not None:
        # Update the edge to have the labels instead of the indices as the entries
        edge = (Graphstate.node_labels.index(edge[0]), Graphstate.node_labels.index(edge[1]))
    
    # Get the vectors of the two nodes of the edge
    x1 = mat(Graphstate.node_positions[edge[0]]).T
    x2 = mat(Graphstate.node_positions[edge[1]]).T
       
    # Calculate help vector, the one from x1 to x2
    p = (x2 - x1)
    
    # Calculate the distance from x1 to x2
    pd = norm(p)
    
    # Normalize p
    phat = p/pd
    
    # Check if an edgemap was fiven
    if edgemap is not None:
        raise ValueError("Warning, passing an edgemap has not yet been implemented.")
    # Now check if a straight edge can be made, or otherwise plot an arc
    else:
        # Check if any nodes intersect with the straight edge    
        if not _do_nodes_intersect_straightedge(Graphstate.node_positions, phat, pd, x1, graphstyle['node_radius']):
            # Now we can plot a straight edge
            # Calculate the two points to draw the line from and to
            # These are the node positions adjusted by the offset given in the graphstyle, i.e. x1 and x2 plys/minus phat for the offset length
            p1,p2 = x1 + (graphstyle['edge_offset'])*phat, x2 - (graphstyle['edge_offset'])*phat
            
            # Plot the line
            axis.plot([p1[0,0], p2[0,0]], [p1[1,0],p2[1,0]], color = graphstyle['edge_color'], linewidth = graphstyle['edge_width'])
            
        # If the previous if statement doesn't hit, we make an arced edge   
        else:
            # Check if there's an arc direction, if not init 
            # The variable is flip; this is the direction that the arc goes (i.e. positive or negative angle)
            # This is flipped any time there is an arced 
            if arcflip is None:
                arcflip = True
            arc = None
            # Loop through all angles to find an arc that doesn't intersect with any node. Also loop through both directions to flip
            from itertools import product
            for angle, direction in product([15,20,25,30,35,45], [arcflip, not arcflip]):
    
                
                # Calculate the circle params of the arc
                [xr, yr], r, [theta1, theta2] = _calculate_arc_params(Graphstate.node_positions[edge[0]],
                                                                      Graphstate.node_positions[edge[1]], angle=((-1)**direction * angle), 
                                                                      offset = graphstyle['edge_offset'])
                
                # Check if no nodes intersect with the arcedge
                if not _do_nodes_intersects_arcedge(Graphstate.node_positions, xr, yr, r, theta1, theta2, graphstyle, anglecoor = 'deg'):
                    # Make the edge with the current circle params of the arc if the if statement passes
                    arc = Arc((xr,yr), width = 2*r, height = 2*r, 
                                        theta1=theta1, theta2=theta2, color = graphstyle['edge_color'], linewidth = graphstyle['edge_width'])
                    # Break out of the loop because we found a valid angle
                    arcflip = not direction
                    break
            
            # If no edge was defined it means we can't find a proper arc that doesn't intersect. Then just do a small one.
            if arc is None:
                print("Warning, no arc could be found that doesn't intersect with at least one node. Resorting to angle = 15 degree.")
                # Calculate the circle params of the arc
                [xr, yr], r, [theta1, theta2] = _calculate_arc_params(Graphstate.node_positions[edge[0]],
                                                                      Graphstate.node_positions[edge[1]], angle=((-1)**arcflip * 15), 
                                                                      offset = graphstyle['edge_offset'])
                arc = Arc((xr,yr), width = 2*r, height = 2*r, 
                                    theta1=theta1, theta2=theta2, color = graphstyle['edge_color'], linewidth = graphstyle['edge_width'])    
            # Plot the edge, flip the arc direction
            arcflip = not arcflip
            axis.add_patch(arc)
    
    # Return the flip direction        
    return arcflip

#%% Edge plots
# def plot_straight_edge(Graphstate, axis, graphsyle, edge)

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
    theta1 = (atan((y1 - yr)/(x1 - xr)) + pi/2 - copysign(pi/2,x1 - xr)) % (2*pi)
    theta2 = (atan((y2 - yr)/(x2 - xr)) + pi/2 - copysign(pi/2,x2 - xr)) % (2*pi)
    
    # Check if the arc goes from theta1 to theta2 or the other way around
    if abs(theta1 - theta2) > pi:
        # The arc is now going in the `wrong` direction, so we have to add the offset in the wrong direction
        # Add the offset
        theta1, theta2 = theta1 - (offset/r), theta2 + (offset/r)
    else:
        theta1, theta2 = theta1 - (offset/r), theta2 + (offset/r)
    
    if anglecoor == 'deg':
        return [xr, yr], r, [conv*theta2, conv*theta1]
    
    return [xr, yr], r, [theta2, theta1]


def _do_nodes_intersect_straightedge(node_positions, phat, pd, x1, node_radius):
    '''
    For every node perform the following
    Calculate if the node with vector e (with origin at x1!) with radius node_radius intersects 
    the line from point x1 to x2. The only thing necessary to provide is the vector p = x2 - x1.
    Note that for any given point at [x1, y1] = x, the vector e = x - x1, the vector from x1 to the point.
    If any node intersects: return True
    If no node intersects: return False
    '''
    #Loop through every node in the node positions
    for node, node_position in enumerate(node_positions):

        e = mat(node_position).T - x1
        # We decompose e into p and p_ort. Then e = <e,p>p + <e,port>port, if p and port are normalized.
        # Let e = a*p + b*port
        # a is the projection of e upon the line through x1 and x2
        # b is th distance from e to the line through x1 and x2 (with negative)
        
        # Calculate a
        a = (e.T @ phat)[0,0]
        # x1 is at projection 0, x2 is at projection pd, so a needs to be between this range for the node to potentially intersect
        inprange = (a > 0.01) and (0.01 < pd - a)
    
    
        # Calculate b
        # First we calculate the projection of E on port, the orthogonal complement of p. This is b*port
        # For any vector e, we have e = <e,p>p + <e,port>port, if p and port are normalized
        # So <e,port>port = e - <e,p>p = e - a*p
        epor = e - a*phat
    
        # If the edge intersects with a node, that means the node position is closer to the line than the node radius.
        # b**2 = epor @ epor.T, so we check this against the square of the node radius. (So that we also only get positive values)
        inportrange = (epor.T @ epor)[0,0] < node_radius**2
        
        
        # If both truth values are True, we have a intersecting node. Then we should return True
        if inprange and inportrange:
            return True
    return False


def _do_nodes_intersects_arcedge(node_positions, xr, yr, r, theta1, theta2, graphstyle, anglecoor = 'deg'):
    '''
    For every node perform the following
    Calculate if the node with vector e (with origin at x1!) with radius node_radius intersects 
    the line from point x1 to x2. The only thing necessary to provide is the vector p = x2 - x1.
    Note that for any given point at [x1, y1] = x, the vector e = x - x1, the vector from x1 to the point.
    If any node intersects: return True
    If no node intersects: return False
    '''
    
    # Loop through all nodes to check if they intersect with the arc
    for node, node_position in enumerate(node_positions):
        # Calculate distance from middle of circle, d
        # This is square root of (x - xr)**2 + (y - yr)**2
        d = sqrt((node_position[0] - xr)**2 + (node_position[1] - yr)**2)
        
        # If the distance of the node to the midpoint is in the range of the arc radius +- the node radius, it might intersect
        if d < r + graphstyle['node_radius'] and d > r - graphstyle['node_radius']:
            # print(f'node {node} has distance {d} from point {xr, yr} for circle with radius {r}')
            # Now check if the point is actually in the slice of the circle made by the arc.
            point_angle = (atan((node_position[1] - yr)/(node_position[0] - xr)) + pi/2 - copysign(pi/2,node_position[0] - xr)) % (2*pi)
            # print(f'the point makes angle {point_angle} while the thetas are {theta1, theta2}')
            # Convert to degree if necessary
            if anglecoor == 'deg':
                point_angle *= (180/pi)
            
            if point_angle > min(theta1, theta2) and point_angle < max(theta1, theta2):
                return True
    return False