# Global imports
# from netorkx imporxt draw_nodes
from matplotlib.patches import Circle, Arc
from matplotlib.pyplot import figure
from numpy import array, matrix, inner, sin, cos, arccos, concatenate, linspace, rad2deg, ceil, isclose
from scipy.spatial import ConvexHull
from numpy.linalg import norm
from math import copysign, tan, atan, sin, sqrt, pi

#%%
import matplotlib


matplotlib.rcParams.update({
    "text.usetex": False,
    # "font.fontname": "Academy Engraved LET"
})
# matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
#%% Init functions
def calculate_axes_limits(Graphstyle):
    '''
    Calculate x and y axes limits for a graph picture from the x and y min and max coordinates.
    The limits are calculated as such:
        For x, let d be the distance from max to min.
        Then the limits are the xmin - xleftoffset and xmax + xrightoffset, where the offset is the radius of the node in question + tightness*d
        
        For y it works similarly
        
    If equaloffset = True, the offset for both x and y is set to the max of the two
    If xtoyratio is not None, the limits will be adjusted so to have the fixed ratio
    '''
    
    
    # Get the coordinates of the extreme nodes
    minmaxes = Graphstyle.get_extreme_coordinates()
    
    # Get the indices of these nodes
    [xleft, xright], [yleft, yright] = Graphstyle.get_extreme_node_indices()
    
    # Check if the node radius is defined for all nodes equally or for every node separately
    nodesstyle = Graphstyle.nodes_style
    
    
    noderadii = [
        [nodesstyle[xleft]['node_radius'], nodesstyle[xright]['node_radius']],
        [nodesstyle[yleft]['node_radius'], nodesstyle[yright]['node_radius']]
                ]
    
    # Calculate the offsets first
    offsets = []
    for coor, coorradii in zip(minmaxes,noderadii):
        d = coor[1] - coor[0] + coorradii[0] + coorradii[1]
        

        offsets.append(
                            (
                            Graphstyle.patch_style['tightness']*coorradii[0] +
                            Graphstyle.patch_style['padding'] +
                            Graphstyle.figure_style['figure_tightness']*d,
                            Graphstyle.patch_style['tightness']*coorradii[1] +
                            Graphstyle.patch_style['padding'] +
                            Graphstyle.figure_style['figure_tightness']*d
                            )
                        )
        
    if Graphstyle.figure_style['figure_offsetequal']:
        offsets = [[max(max(offsets[0]), max(offsets[1]))] * 2] * 2
    lims = [] 
    
    if Graphstyle.figure_style['figure_ratio'] is None:
        for coor, offset in zip(minmaxes, offsets):
            lims.append([coor[0] - offset[0], coor[1] + offset[1]])
    else:
        x0, x1 = minmaxes[0][0] - offsets[0][0], minmaxes[0][1] + offsets[0][1]
        y0, y1 = minmaxes[1][0] - offsets[1][0], minmaxes[1][1] + offsets[1][1]
        
        dx, dy = (x1 - x0), (y1 - y0)
        
        xr, yr = (x0 + x1 )/2, (y0 + y1)/2
        
        current_r = dx/dy
        
        # from numpy import argmax as _argmax
        # longest_side = _argmax((x1 - x0, y1 - y0))
        
        if Graphstyle.figure_style['figure_ratio'] >= current_r:
            x0, x1 = xr - Graphstyle.figure_style['figure_ratio']*dy/2, xr + Graphstyle.figure_style['figure_ratio']*dy/2

        elif Graphstyle.figure_style['figure_ratio'] < current_r:
            y0, y1 = yr - dx/(2*Graphstyle.figure_style['figure_ratio']), dx/(2*Graphstyle.figure_style['figure_ratio'])
        
        lims = [[x0, x1], [y0,y1]]
    
    return lims

#%% Preparation    
def prepare_graphstatedrawing(Graphstyle):
    '''
    Prepare a Graphstate figure.
    Return a figure and an axis object
    '''
    # If no axislimits are given, calculate them
    axislimits = Graphstyle.figure_style['axes_limits']
    
    if axislimits is None:
        axislimits = calculate_axes_limits(Graphstyle)

    # Unpack the axis limits
    xlim, ylim = axislimits[0], axislimits[1]
    
    # Make figure
    fig = figure(figsize = (Graphstyle.figure_style['figure_multiplier']*(xlim[1] - xlim[0]), 
                            Graphstyle.figure_style['figure_multiplier']*(ylim[1] - ylim[0]))
                 )
    
    # Add the axis and set the params
    axis = fig.add_subplot(111)
    
    axis.set(xlim=xlim, ylim=ylim, aspect=1)
    
    axis.axis('off')


    axis.margins(x=0, y=0)
    
    # fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = 0, hspace = 0)
    
    # Set the background color
    fig.patch.set_facecolor(Graphstyle.figure_style['background_color'])
    axis.patch.set_facecolor(Graphstyle.figure_style['background_color'])

    
    # Plot the title
    if Graphstyle.figure_style['with_title']:
        axis.set_title(**Graphstyle.figure_style['figure_title'])

    
    return fig, axis

def prepare_multiple_graphstatedrawing(Graphstyles: list, nr_rows = None, nr_columns = None, figure_multiplier = None, background_color = None, fill_order = 'row', gridspec_mapping = None, nr_plots = None):
    '''
    Prepare a figure with multiple sublots for drawings.
    Returns a figure and a list of axes.
    If gridspec mapping is provided, it is used in the subplot_mosaic routine; please still probide a nr_rows and nr_cols for the figure size. fill_order will be ignored.
    '''
    
    if not isinstance(Graphstyles, list):
        if nr_rows is not None and nr_columns is not None:
            Graphstyles = [Graphstyles]*nr_rows*nr_columns
        elif nr_plots is not None:
            Graphstyles = [Graphstyles]*nr_plots
        else: raise ValueError("When graphstyles is not a list, provide either the nr of plots, or the nr of columns and nr of rows.")
    
    # Determine number of rows and columns
    nr_drawings = len(Graphstyles)
    if nr_rows is None and nr_columns is None:
        if gridspec_mapping is not None:
            raise ValueError("When providing a gridspec mapping, please manually provide at least a nr of rows or nr_columns")
        from math import ceil
        nr_columns = ceil(nr_drawings**(1/2))
        nr_rows = ceil(nr_drawings/nr_columns)
    elif nr_rows is None:
        print(nr_columns)
        from math import ceil
        nr_rows = ceil(nr_drawings/nr_columns)
        print(nr_rows)
    elif nr_columns is None:
        from math import ceil
        nr_columns = ceil(nr_drawings/nr_rows)
    # Determine figure multiplier
    if figure_multiplier is None:
        figure_multiplier = 3
    
    # Determine background color
    if background_color is None:
        background_color = Graphstyles[0].figure_style['background_color']
    
    assert fill_order == 'row' or fill_order == 'column', f"Provide either 'row' or 'column' as a filling order, not {fill_order}."
    
    # Prepare the grid
    from matplotlib.gridspec import GridSpec
    
    gs = GridSpec(nr_rows, nr_columns)
    
    # Prepare figure
    fig = figure(figsize = (figure_multiplier * nr_columns, figure_multiplier * nr_rows))
    
        
    fig.patch.set_facecolor(background_color)
    
    
    
    
    
    
    if gridspec_mapping is None:
        axes = []
        graphstate_index = 0
        if fill_order == 'column':
            for col_nr in range(nr_columns):
                for row_nr in range(nr_rows):
                    axis = fig.add_subplot(gs[row_nr, col_nr])
                    
                    # If no axislimits are given, calculate them
                    axeslimits = Graphstyles[graphstate_index].figure_style['axes_limits']
                    
                    if axeslimits is None:
                        axeslimits = calculate_axes_limits(Graphstyles[graphstate_index])

                    # Unpack the axis limits
                    xlim, ylim = axeslimits[0], axeslimits[1]
                    
                    axis.set(xlim=xlim, ylim=ylim, aspect=1)
                    axis.axis('off')
                    
                    axis.patch.set_facecolor(Graphstyles[graphstate_index].figure_style['background_color'])
                    
                    # Plot the title
                    if Graphstyles[graphstate_index].figure_style['with_title']:
                        axis.set_title(**Graphstyles[graphstate_index].figure_style['figure_title'])
                    
                    axes.append(axis)
                    
                    graphstate_index += 1
                    
                    if graphstate_index == len(Graphstyles):
                        return fig, axes
        if fill_order == 'row':
            
            for row_nr in range(nr_rows):
                for col_nr in range(nr_columns):
                    
                    
                    axis = fig.add_subplot(gs[row_nr, col_nr])
                    
                    # If no axislimits are given, calculate them
                    axeslimits = Graphstyles[graphstate_index].figure_style['axes_limits']
                    
                    if axeslimits is None:
                        axeslimits = calculate_axes_limits(Graphstyles[graphstate_index])

                    # Unpack the axis limits
                    xlim, ylim = axeslimits[0], axeslimits[1]
                    
                    axis.set(xlim=xlim, ylim=ylim, aspect=1)
                    axis.axis('off')
                    
                    axis.patch.set_facecolor(Graphstyles[graphstate_index].figure_style['background_color'])
                    
                    # Plot the title
                    if Graphstyles[graphstate_index].figure_style['with_title']:
                        axis.set_title(**Graphstyles[graphstate_index].figure_style['figure_title'])
                    
                    axes.append(axis)
                    
                    graphstate_index += 1
                    
                    if graphstate_index == len(Graphstyles):
                        return fig, axes
    else:
        axesdict = fig.subplot_mosaic(gridspec_mapping)
        
        
        # This assumes that axes are labelled "A", "B", .... Will fail otherwise
        axes = [axesdict[chr(65 + i)] for i in range(len(axesdict.keys()))]
        
        if len(axes) != len(Graphstyles):
            raise ValueError(f"gridspec_mapping specifies {len(axes)} subplots but {len(Graphstyles)} graphstyles are provided.")
        
        for ax, graphstyle in zip(axes, Graphstyles):
            # If no axislimits are given, calculate them
            axeslimits = graphstyle.figure_style['axes_limits']
            
            if axeslimits is None:
                # raise ValueError("Please provide axeslimits when using a gridspec_mapping!")
                axeslimits = calculate_axes_limits(graphstyle)

            # Unpack the axis limits
            xlim, ylim = axeslimits[0], axeslimits[1]
            print(xlim, ylim)
            ax.set(xlim=xlim, ylim=ylim, aspect=1)
            # ax.set(aspect=1)
            ax.axis('off')
            
            ax.patch.set_facecolor(graphstyle.figure_style['background_color'])
            
                
                    
    return fig, axes
        

#%% Nodes
def draw_nodes(Graphstyle, axis):
    '''
    Draw the nodes of the Graphstate in the given axis.
    The graphstyle is either a dictionary for instructions on how to draw the nodes, or a list of dictionaries on how to draw every node separately.
    '''
    labels = Graphstyle.node_labels
    if labels is None:
        labels = [f'{i}' for i in range(Graphstyle.nr_nodes)]
        
    nodesstyle = Graphstyle.nodes_style

    assert len(nodesstyle) == Graphstyle.nr_nodes, f"Graph style has instructions for {len(nodesstyle)} nodes but graphstayle has {Graphstyle.nr_nodes} nodes."
    
    # Plot every node iteratively
    for node_index, position in enumerate(Graphstyle.node_positions):
        draw_node(position, nodesstyle[node_index], labels[node_index], axis)

def draw_node(position, nodestyle, label, axis):
    '''
    Draw a single node given by the node axis.
    '''
    # Make a circle at the node position
    circle = Circle(position, radius = nodestyle['node_radius'], 
                    facecolor = nodestyle['node_color'], 
                    edgecolor = nodestyle['node_edgecolor'],
                    linewidth = nodestyle['node_edgewidth'],
                    linestyle = nodestyle['node_edgestyle'])
    
    # Add the node to the axis
    axis.add_patch(circle)
    
    # Make the label if the graphstyle wants it
    if nodestyle['with_labels']:
        axis.text(x = position[0] + nodestyle['label_xoffset'], y = position[1] + nodestyle['label_yoffset'], 
                  s = label, 
                  fontsize = nodestyle['label_fontsize'],
                  fontname = nodestyle['label_fontname'],
                  ha="center",
                  va="center_baseline", 
                  usetex = nodestyle['tex_for_labels'], 
                  color = nodestyle['label_color'])
    
#%% Edges
def draw_edges(Graphstate, Graphstyle, axis, edgesmap = None):
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
    edgestyle = Graphstyle.edge_style
    if isinstance(edgestyle, dict):
        if edgesmap is None:
            # Init an arc direction so that the arc directions can get flipped (from positive to negative angle)
            flip = True
            # Now loop through all edges
            for edge in Graphstate.get_edges():
                flip = __draw_edge(Graphstyle, axis, edge, arcflip = flip, max_node_radius=max(Graphstyle.get_nodes_radii()))
        else:
            for edge, edgemap in zip(Graphstate.get_edges(), edgesmap):
                __draw_edge(Graphstyle, axis, edge, edgemap)
                
    elif isinstance(edgestyle, list):
        assert len(edgestyle) == len(Graphstate.get_edges()), f"Graph style has instructions for {len(edgestyle)} edges but graphstate has {len(Graphstate.get_edges())} edges."
        flip = True
        for edge_index, edge in enumerate(Graphstate.get_edges()):
            if edgesmap is None:
                flip = __draw_edge(Graphstyle, axis, edgestyle[edge_index], edge, arcflip = flip)
            else:
                __draw_edge(Graphstyle, axis, edgestyle[edge_index], edge, edgemap[edge_index])

def __draw_edge(Graphstyle, axis, edge, edgemap = None, arcflip = None, max_node_radius = 0.1):
    '''
    Draw the edge for the given Graphstate and fiven edgemap. If edgemap is 
    '''
    edgestyle = Graphstyle.edge_style
    other_nodes = list(range(Graphstyle.nr_nodes))
    other_nodes.pop(max(edge))
    other_nodes.pop(min(edge))
    
    other_nodes_positions = [Graphstyle.node_positions[index] for index in other_nodes]
    other_nodes_radii = Graphstyle.get_nodes_radii(other_nodes)
    
    # Get the radii of the nodes that the edge connects.
    r1, r2 = Graphstyle.get_nodes_radii(edge)
    
    # Get the vectors of the two nodes of the edge
    x1 = matrix(Graphstyle.node_positions[edge[0]]).T
    x2 = matrix(Graphstyle.node_positions[edge[1]]).T
       
    # Calculate help vector, the one from x1 to x2
    p = (x2 - x1)
    
    # Calculate the distance from x1 to x2
    pd = norm(p)
    
    # Normalize p
    phat = p/pd
    
    # Check if an edgemap was fiven
    if edgemap is not None:
        raise NotImplementedError("Warning, passing an edgemap has not yet been implemented.")
    # Now check if a straight edge can be made, or otherwise plot an arc
    else:
        # Check if any nodes intersect with the straight edge
        
        if not _do_nodes_intersect_straightedge(other_nodes_positions, other_nodes_radii, phat, pd, x1, Graphstyle.edge_style['nodeedgetightness'], Graphstyle.edge_style['nodeedgepadding']):
            # Now we can plot a straight edge
            # Calculate the two points to draw the line from and to
            # These are the node positions adjusted by the offset given in the graphstyle, i.e. x1 and x2 plys/minus phat for the offset length
            p1,p2 = x1 + (r1 + edgestyle['edge_offset'])*phat, x2 - (r2 + edgestyle['edge_offset'])*phat
            
            # Plot the line
            axis.plot([p1[0,0], p2[0,0]], [p1[1,0],p2[1,0]], color = edgestyle['edge_color'], linewidth = edgestyle['edge_width'])
            
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
                #
                # print(f"At angle {angle}")
                
                # Calculate the circle params of the arc
                [xr, yr], r, [theta1, theta2] = _calculate_arc_params(Graphstyle.node_positions[edge[0]],
                                                                      Graphstyle.node_positions[edge[1]], angle=((-1)**direction * angle), 
                                                                      offset = edgestyle['edge_offset'], node_radii = (r1,r2), crossing_radius = max_node_radius)
                
                # Check if no nodes intersect with the arcedge
                if not _do_nodes_intersects_arcedge(other_nodes_positions, other_nodes_radii, xr, yr, r, theta1, theta2, anglecoor = 'deg',
                                                    tightness = Graphstyle.edge_style['nodeedgetightness'], padding = Graphstyle.edge_style['nodeedgepadding']):
                    # print(f"Arc does not intersect for {theta1},{theta2} and {max_node_radius} with {xr},{yr} as midpoint of circle {r}")
                    # Make the edge with the current circle params of the arc if the if statement passes
                    arc = Arc((xr,yr), width = 2*r, height = 2*r, 
                                        theta1=theta1, theta2=theta2, color = edgestyle['edge_color'], linewidth = edgestyle['edge_width'])
                    # Break out of the loop because we found a valid angle
                    arcflip = not direction
                    break
            
            # If no edge was defined it means we can't find a proper arc that doesn't intersect. Then just do a small one.
            if arc is None:
                print("Warning, no arc could be found that doesn't intersect with at least one node. Resorting to angle = 15 degree.")
                # Calculate the circle params of the arc
                [xr, yr], r, [theta1, theta2] = _calculate_arc_params(Graphstyle.node_positions[edge[0]],
                                                                      Graphstyle.node_positions[edge[1]], angle=((-1)**arcflip * 15), 
                                                                      offset = edgestyle['edge_offset'], node_radii = (r1,r2), crossing_radius = max_node_radius)
                arc = Arc((xr,yr), width = 2*r, height = 2*r, 
                                    theta1=theta1, theta2=theta2, color = edgestyle['edge_color'], linewidth = edgestyle['edge_width'])    
            # Plot the edge, flip the arc direction
            arcflip = not arcflip
            axis.add_patch(arc)
    
    # Return the flip direction        
    return arcflip

# def _plot_straightedge()

#%% Edge plots
# def plot_straight_edge(Graphstate, axis, graphsyle, edge)

#%% Hulls/collections of nodes
def draw_path_around_nodes(Graphstyle, node_selection: list, axis):
    '''
    Draw a patch around a selection of nodes in the graphstate.
    '''
    # Get the nodes' positions and radii.
    selection_positions = Graphstyle.get_node_positions(node_selection)
    
    selection_radii = Graphstyle.get_nodes_radii(node_selection)
    
    # Obtain the extreme nodes of the hull, i.e. those nodes around which the patch is gonna be drawn.
    extreme_nodes_indices = _get_indices_extreme_nodes_hull(selection_positions, selection_radii)
    
    extreme_nodes_positions = [selection_positions[index] for index in extreme_nodes_indices]
    extreme_nodes_radii = [selection_radii[index] for index in extreme_nodes_indices]
    
    # Draw the contour
    _draw_contour(Graphstyle.patch_style, extreme_nodes_positions, extreme_nodes_radii, axis)
    
    

def _draw_contour(patch_style, node_positions, radii, axis):
    '''
    Draw a contour/hull/patch around the given nodes, in order of the list that they're provided.
    '''

    assert len(node_positions) == len(radii),f"Warning, provided {len(node_positions)} Node positions but {len(radii)} radii."
    
    
    
    # Pre- and append the node positions and radii with their last and first entry, respectively.
    # This facilitates easier looping
    node_positions = [node_positions[-1]] + node_positions + [node_positions[0]]
    radii = [radii[-1]] + radii + [radii[0]]
    

    # Scale and pad the radii with the given parameters
    radii = [patch_style['tightness']*rad + patch_style['padding'] for rad in radii]
    
    # Init a list containing all the points of the contour
    contourpoints = []
    
    # Loop through every node position
    for i in range(1, len(node_positions) - 1):
        # Define the base point, i.e. the location of the node
        x = matrix([[node_positions[i][0]], [node_positions[i][1]]])
        
        # print(f"Current node: {x}, rad = {radii[i]}")
        
        # Get the vectors from the point to the previous and next point
        to_prev = matrix([
            [node_positions[i-1][0] - node_positions[i][0]],
            [node_positions[i-1][1] - node_positions[i][1]]
            ])
        to_next = matrix([
            [node_positions[i+1][0] - node_positions[i][0]],
            [node_positions[i+1][1] - node_positions[i][1]]
            ])
        
        # Get the norms of the vectors
        norm_prev = norm(to_prev)
        norm_next = norm(to_next)
        
        # Calculate the angle between these two vectors
        # We have to calculate the inner product separately; 
        # for vectors in the same direction (i.e. for a contour of 2 points)
        # we can have machine precision have the inner product be slightly above 1 (e.g. 1.00000002)
        # We have to make sure that it falls within the range -1, 1 for the arccos to work properly
        # We do this by taking the minimum of the product and 1, and then the maximum of that and -1
        
        rel_angle = 2*pi - arccos(max(min(inner(to_prev.T, to_next.T)[0,0]/(norm_prev * norm_next), 1), -1))
         
        
        # Get the incoming angle of the arc
        inc_angle = arccos((radii[i] - radii[i-1])/norm_prev)
        
        
        
        # If the previous node is larger then the angle may be flipped w.r.t. pi, whichever is largest
        if radii[i-1] > radii[i]:
            inc_angle = max((pi - inc_angle,inc_angle))
        # If the previous node is smaller if should be the minimum
        else:
            inc_angle = min((pi - inc_angle,inc_angle))
            
        # Get the outcoming angle of the arc
        out_angle = arccos((radii[i+1] - radii[i])/norm_next)
        
        # If the next node is larger, then the angle may be flipped w.r.t. pi, whichever is largest
        if radii[i+1] > radii[i]:
            out_angle = max((pi - out_angle, out_angle))
        else:
            out_angle = min((pi - out_angle, out_angle))
        
        # Define the arc range
        arc_range = linspace(inc_angle, rel_angle - out_angle, endpoint = True, num = int(ceil(rad2deg(rel_angle - out_angle - inc_angle))/5))
        # arc_range = np.linspace(0, rel_angle - out_angle, endpoint = True, num = 50)# num = int(np.ceil(np.rad2deg(rel_angle - out_angle - inc_angle))/5))

        for arc_angle in arc_range:
            rel_vector = x + radii[i]*_rotate_2D(to_prev/norm_prev, arc_angle)
            
            contourpoints.append([rel_vector[0,0],rel_vector[1,0]])
    
    # Append the first position to make a loop
    contourpoints.append(contourpoints[0])
    
    # Unpack and plot
    x, y = list(zip(*contourpoints))
    axis.fill(x,y, 
              alpha = patch_style['face_alpha'],
              facecolor = patch_style['face_color'],
              linewidth = patch_style['edge_width'],
              edgecolor = patch_style['edge_color'],
              linestyle = patch_style['edge_style']
              )
    
    return contourpoints

#%% Helper functions

def _calculate_node_edge_dist(node, p1, p2):
    '''
    Calculate the distance from a node to 
    '''

def _calculate_arc_params(pos_1, pos_2, angle = 15, offset = 0, node_radii = (0,0), crossing_radius = 0.01, anglecoor = 'deg'):
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
        t1, t2: the angles of the beginning and end of the circle arc in radians or degree, as specified by anglecoor
    '''    
    # Import necessary math functions
    if angle == 0:
        angle = 0.001
    elif angle < 0:
        return _calculate_arc_params(pos_2, pos_1, angle = -angle, offset = offset, node_radii = node_radii, crossing_radius = crossing_radius, anglecoor = anglecoor)
    
    
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
    
    
    # Calculate the angle offset to incorporate the radius and the edge offset
    angle_offset = (
        2*sin(node_radii[0]/(2*r)) + 2*sin(offset/(2*r)),
        2*sin(node_radii[1]/(2*r)) + 2*sin(offset/(2*r)),
        )
    
    # Check if the arc goes from theta1 to theta2 or the other way around
    if abs(theta1 - theta2) > pi:
        # The arc is now going in the `wrong` direction, so we have to add the offset in the wrong direction
        # Add the offset
        theta1, theta2 = theta1 - angle_offset[0], theta2 + angle_offset[1]
    else:
        theta1, theta2 = theta1 - angle_offset[0], theta2 + angle_offset[1]
    
    if anglecoor == 'deg':
        return [xr, yr], r, [conv*theta2, conv*theta1]
    
    return [xr, yr], r, [theta2, theta1]


def _do_nodes_intersect_straightedge(nodes_positions, nodes_radii, phat, pd, x1, tightness = 1.1, padding = 0):
    '''
    For every node perform the following
    Calculate if the node with vector e (with origin at x1!) with radius node_radius intersects 
    the line from point x1 to x2. The only thing necessary to provide is the vector p = x2 - x1.
    Note that for any given point at [x1, y1] = x, the vector e = x - x1, the vector from x1 to the point.
    If any node intersects: return True
    If no node intersects: return False
    '''
    #Loop through every node in the node positions
    for node, (node_position, node_radius) in enumerate(zip(nodes_positions, nodes_radii)):

        e = matrix(node_position).T - x1
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
        inportrange = (epor.T @ epor)[0,0] < (tightness*(padding + node_radius))**2
        
        
        # If both truth values are True, we have a intersecting node. Then we should return True
        if inprange and inportrange:
            return True
    return False


def _do_nodes_intersects_arcedge(nodes_positions, nodes_radii, xr, yr, r, theta1, theta2, anglecoor = 'deg', tightness = 1.1, padding = 0):
    '''
    For every node perform the following
    Calculate if the node with vector e (with origin at x1!) with radius node_radius intersects 
    the line from point x1 to x2. The only thing necessary to provide is the vector p = x2 - x1.
    Note that for any given point at [x1, y1] = x, the vector e = x - x1, the vector from x1 to the point.
    If any node intersects: return True
    If no node intersects: return False
    '''
    
    # Loop through all nodes to check if they intersect with the arc
    for node, (node_position, node_radius) in enumerate(zip(nodes_positions, nodes_radii)):
        # Calculate distance from middle of circle, d
        # This is square root of (x - xr)**2 + (y - yr)**2
        d = sqrt((node_position[0] - xr)**2 + (node_position[1] - yr)**2)
        
        # If the distance of the node to the midpoint is in the range of the arc radius +- the node radius, it might intersect
        if abs(d - r) <= tightness*(padding + node_radius):
            # print(f'node {node} has distance {d} from point {xr, yr} for circle with radius {r}')
            # Now check if the point is actually in the slice of the circle made by the arc.
            # If the x coordinates are the same, you divide by zero. To circumvent, manually set the angle to pi/2 or -pi/2
            if isclose(node_position[0], xr):
                if node_position[1] > yr:
                    point_angle = pi/2
                elif node_position[1] < yr:
                    point_angle = -pi/2
            else:
                point_angle = (atan((node_position[1] - yr)/(node_position[0] - xr)) + pi/2 - copysign(pi/2,node_position[0] - xr)) % (2*pi)
            
            # Convert to degree if necessary
            
            
            conv = 1
            
            if anglecoor == 'deg':
                conv = 180/pi
            
            point_angle *= conv
            
            
            if theta1 >= theta2:
                if abs(theta1 - theta2) >= 180:
                    
                    theta2 += 2*pi*conv
            # print(f'the point makes angle {point_angle} while the thetas are {theta1, theta2}')
            if point_angle > min(theta1, theta2) and point_angle < max(theta1, theta2):
                return True
            if point_angle + 2*pi*conv > min(theta1, theta2) and point_angle + 2*pi*conv < max(theta1, theta2):
                return True
    return False

##
def _get_indices_extreme_nodes_hull(positions: list, radii: list):
    '''
    For the nodes with positions and radii given, obtain the extreme points of the convex hull of these nodes.
    Returns the indices of the extreme points, in anti-clockwise order.
    '''
    if isinstance(radii, (float, int)):
        radii = [radii]*len(positions)
    elif isinstance(radii, (list, tuple)):
        assert len(positions) == len(radii), f"{len(positions)} positions given but {len(radii)} radii given."
    
    XY = array(positions)
    
    padded_positions = concatenate((XY + array([(radius, 0) for radius in radii]), XY - array([(radius, 0) for radius in radii]),
                                      XY + array([(0, radius) for radius in radii]), XY - array([(0, radius) for radius in radii])))
    
    
    hull = ConvexHull(padded_positions)
    
    extreme_nodes_indices = []
    
    for index in hull.vertices:
        associated_node_index = index % len(positions)
        if associated_node_index not in extreme_nodes_indices:
            extreme_nodes_indices.append(associated_node_index)
    
    return extreme_nodes_indices

def _rotate_2D(vector, angle):
    '''
    '''
    R = matrix([[cos(angle), -1*sin(angle)],[sin(angle), cos(angle)]])
    return R @ vector