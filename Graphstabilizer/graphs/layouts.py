from numpy import linspace as nplinspace, meshgrid as npmeshgrid
from numpy import cos as npcos, sin as npsin, pi as nppi

def oned_cluster(nr_nodes, spacing = 1):
    '''
    Returns a list of positions for a 1d grid layout, 
    where the y coordinate is 0 and the nodes are all spacing apart in the x axis, starting from x=0
    '''
    return [(spacing * x, 0) for x in range(nr_nodes)]

def line(nr_nodes, spacing = 1):
    return oned_cluster(nr_nodes, spacing)

def twod_cluster(columns, rows, row_sep = 1, col_sep = 1):
    '''
    Returns a list of positions for a 2d grid layout.
    Optionally provide the row separation and column separation between each pair of nodes.
    '''
    x_coordinates = [row_sep * x for x in range(columns)]
    y_coordinates = [col_sep * (rows - 1 - y) for y in range(rows)]
    
    mesh = npmeshgrid(x_coordinates, y_coordinates)
    return list(zip(mesh[0].flatten(), mesh[1].flatten()))

def tree(nr_layers, branch_factor = 2, layer_distance = 1, bottom_layer_sep = 1):
    '''
    Returns a list of positions for a tree with nr_layers and a given branching factor. 
    The lowest layer will have 
    '''
    
    D_b = bottom_layer_sep
    D_l = layer_distance
    
    L = nr_layers
    B = branch_factor
    
    
    coordinates = [(0,0)]
    
    end = (1/2)*D_b*(B**L - 1)
    
    for layernr in range(1,nr_layers):
        
        
        x_coordinates = (layernr/nr_layers)*nplinspace(-end,end,endpoint = True,num = B**layernr)
        
        coordinates.extend([(x,-1*D_l*layernr) for x in x_coordinates])
    
    return coordinates

def circular_tree(nr_layers, branch_factor = 2, layer_distance = 1):
    '''
    Returns a list of positions for a tree with a specific branching factor, 
    but wrapped around in a circle.

    '''
    from networkx import nx_agraph
    from networkx import balanced_tree
    
    
    G = balanced_tree(branch_factor, nr_layers)
    return nx_agraph.graphviz_layout(G, prog="twopi", args="")

def radial_out(nr_rays, nodes_per_ray, node_dist = 1, offset_angle = 0, direction = 'clockwise'):
    '''
    Return a list of positions where all nodes are positioned along rays going outwards from the middle.
    One node is in the middle, and then all nodes are radially going outwards, 
    where every ray is completely filled, and then the next one is started.
    The nodes on one ray are distance node_dist from each other apart.
    The offset_angle is the angle (in radians) that the first ray makes with the x-axis.
    The direction is either 'clockwise' or 'anti-clockwise'.
    '''
    
    angles = nplinspace(offset_angle, 2*nppi + offset_angle, num = nr_rays, endpoint = False)
    if direction == 'clockwise':
        angles = nplinspace(2*nppi + offset_angle, offset_angle, num = nr_rays, endpoint = False)

    
    positions = [(0,0)]
    
    for angle in angles:
        positions.extend([(node_dist*dist*npcos(angle),node_dist*dist*npsin(angle)) for dist in range(1,nodes_per_ray+1)])
    
    return positions

def ellipse(nr_nodes, a = 1, b = 1, offset_angle = 0, direction = 'clockwise'):
    '''
    Return a list of positions on an ellipse with horizontal axis a and vertical axis b.
    The points are calculated as (x,y) = (a*cos(phi), b*sin(phi)), 
    where phi is equally spaced around the circle.
    offset_angle (default 0) sets the angle that the first node makes with the x axis
    if direction == 'clockwise', the points go in clockwise order, otherwise anti-clockwise.
    '''
    angles = nplinspace(offset_angle, 2*nppi + offset_angle, num = nr_nodes, endpoint = False)
    if direction == 'clockwise':
        angles = nplinspace(2*nppi + offset_angle, offset_angle, num = nr_nodes, endpoint = False)
    
    return [(a*npcos(angle), b*npsin(angle)) for angle in angles]
    
def star(nr_nodes, radius = 1, offset_angle = 0, direction = 'clockwise'):
    '''
    Get the star graph, with one node in the middle
    and all other nodes equidistant from each other on a circle with the given radius centered around the middle node.
    '''
    return radial_out(nr_nodes - 1, 1, radius, offset_angle, direction)

# def ring(nr_nodes, radius = 1, offset_angle = 0, direction = 'clockwise'):
#     '''
#     Return a list of positions where all nodes are on a ring or circle around the origin with the given radius. The nodes are all equally distant from each other.
#     The offset_angle is the angle (in radians) that the first node makes with the x-axis.
#     '''
    
#     pos = radial_out(nr_rays = nr_nodes, nodes_per_ray = 1, node_dist = radius, offset_angle = offset_angle, direction = direction)
    
#     return pos[1:]

def ring(nr_nodes, radius = 1, offset_angle = 0, direction = 'clockwise'):
    '''
    Return a list of positions where all nodes are on a ring or circle around the origin with the given radius. The nodes are all equally distant from each other.
    The offset_angle is the angle (in radians) that the first node makes with the x-axis.
    '''
    return ellipse(nr_nodes = nr_nodes, a = radius, b = radius, offset_angle = offset_angle, direction = direction)

def circle(nr_nodes, radius = 1, offset_angle = 0, direction = 'clockwise'):
    '''
    See ring.
    '''
    return ring(nr_nodes, radius, offset_angle, direction)

def cut_ring(nr_nodes, radius = 1, nr_holes = 1, start_of_cut_angle = 0, direction = 'clockwise'):
    '''
    Return a list of positions where all nodes are on a ring or circle around the origin with the given radius.
    '''
    pos = ring(nr_nodes + nr_holes, radius = radius, offset_angle = start_of_cut_angle, direction = direction)
    
    return pos[nr_holes:]


def shift_positions(positions, shift):
    '''
    Shift the given positions given by the shift = (x,y).
    Returns a new list of positions.
    '''
    pos2 = []
    for pos in positions:
        pos2.append((pos[0]+shift[0],pos[1] + shift[1]))
    
    return pos2