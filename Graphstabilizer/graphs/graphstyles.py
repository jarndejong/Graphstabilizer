# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:10:27 2023

@author: jarn
"""
from Graphstabilizer.checkers.graphs import check_are_nodelabels, check_are_nodepositions
from Graphstabilizer.checkers.elementary import check_is_node_index


#%% Templates   
whiteonblack = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.15,
        'node_color'                : 'k',
        'node_edgecolor'            : 'w',
        'node_edgewidth'            : 2,
        'node_edgestyle'            : '-',
        'label_fontname'            : 'AppleMyungjo',
        'label_color'               : 'w',
        'label_fontsize'            : 14,
        'label_xoffset'             : 0,
        'label_yoffset'             : 0,
        'tex_for_labels'            : True,
    },
    'edge_style' :          {
        'edge_color'                : 'w',
        'edge_width'                : 2,
        'edge_offset'               : 0.1,
        'edge_fontsize'             : 16,
        'nodeedgepadding'           : 0.05,
        'nodeedgetightness'         : 1.1,
    },
    'figure_style' :        {
        'figure_multiplier'         : 2,
        'figure_tightness'          : 0.01,
        'figure_offsetequal'        : True,
        'background_color'          : 'k',
        'axes_limits'               : None,
        'figure_ratio'              : None,
        'with_title'                : False,
        'figure_title'              : {
            'label'                 : None,
            'fontdict'              : {'fontsize': 10, 'color': 'w', 'verticalalignment': 'baseline',
                                     # 'fontweight': 'b',
                                     # 'horizontalalignment': loc
                                     },
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.5,
        'face_color'                : '#aee5fc',
        'edge_color'                : '#aee5fc',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0.1,
        'tightness'                 : 1.05
    },
    }



blackonwhite = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.15,
        'node_color'                : 'w',
        'node_edgecolor'            : 'k',
        'node_edgewidth'            : 1,
        'node_edgestyle'            : '-',
        'label_fontname'            : 'AppleMyungjo',
        'label_color'               : 'k',
        'label_fontsize'            : 10,
        'label_xoffset'             : 0,
        'label_yoffset'             : 0,
        'tex_for_labels'            : False,
    },
    'edge_style' :          {
        'edge_color'                : 'k',
        'edge_width'                : 2,
        'edge_offset'               : 0.1,
        'edge_fontsize'             : 16,
        'nodeedgepadding'           : 0.05,
        'nodeedgetightness'         : 1.1,
    },
    'figure_style' :        {
        'figure_multiplier'         : 2,
        'figure_tightness'          : 0.01,
        'figure_offsetequal'        : True,
        'background_color'          : 'w',
        'axes_limits'               : None,
        'figure_ratio'              : None,
        'with_title'                : False,
        'figure_title'              : {
                    'label'                 : None,
                    'fontdict'              : {'fontsize': 10, 'color': 'k', 'verticalalignment': 'baseline',
                                             # 'fontweight': 'b',
                                             # 'horizontalalignment': loc
                                             },
                    'pad'                   : 10,
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.5,
        'face_color'                : 'r',
        'edge_color'                : 'r',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0.05,
        'tightness'                 : 1.2
    },
    }

metawhiteonblack = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.4,
        'node_color'                : 'k',
        'node_edgecolor'            : 'w',
        'node_edgewidth'            : 2,
        'node_edgestyle'            : '-',
        'label_fontname'            : 'AppleMyungjo',
        'label_color'               : 'w',
        'label_fontsize'            : 15,
        'label_xoffset'             : 0,
        'label_yoffset'             : 0,
        'tex_for_labels'            : False,
    },
    'edge_style' :          {
        'edge_color'                : 'w',
        'edge_width'                : 2,
        'edge_offset'               : 0.25,
        'edge_fontsize'             : 16,
        'nodeedgepadding'           : 0.05,
        'nodeedgetightness'         : 1.1,
    },
    'figure_style' :        {
        'figure_multiplier'         : 1,
        'figure_tightness'          : 0.01,
        'figure_offsetequal'        : True,
        'background_color'          : 'k',
        'axes_limits'               : None,
        'figure_ratio'              : 1,
        'with_title'                : False,
        'figure_title'              : {
                    'label'                 : None,
                    'fontdict'              : {'fontsize': 10, 'color': 'w', 'verticalalignment': 'baseline',
                                             # 'fontweight': 'b',
                                             # 'horizontalalignment': loc
                                             },
                    'pad'                   : 10,
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.5,
        'face_color'                : '#aee5fc',
        'edge_color'                : '#aee5fc',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0.2,
        'tightness'                 : 1.05
    },
    }

metablackonwhite = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.4,
        'node_color'                : 'w',
        'node_edgecolor'            : 'k',
        'node_edgewidth'            : 2,
        'node_edgestyle'            : '-',
        'label_fontname'            : 'AppleMyungjo',
        'label_color'               : 'k',
        'label_fontsize'            : 15,
        'label_xoffset'             : 0,
        'label_yoffset'             : 0,
        'tex_for_labels'            : False,
    },
    'edge_style' :          {
        'edge_color'                : 'k',
        'edge_width'                : 3,
        'edge_offset'               : 0.08,
        'edge_fontsize'             : 16,
        'nodeedgepadding'           : 0.05,
        'nodeedgetightness'         : 1.1,
    },
    'figure_style' :        {
        'figure_multiplier'         : 1,
        'figure_tightness'          : 0,
        'figure_offsetequal'        : True,
        'background_color'          : 'k',
        'axes_limits'               : None,
        'figure_ratio'              : 1,
        'with_title'                : False,
        'figure_title'              : {
                    'label'                 : None,
                    'fontdict'              : {'fontsize': 10, 'color': 'k', 'verticalalignment': 'baseline',
                                             # 'fontweight': 'b',
                                             # 'horizontalalignment': loc
                                             },
                    'pad'                   : 10,
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.5,
        'face_color'                : '#aee5fc',
        'edge_color'                : '#aee5fc',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0.1,
        'tightness'                 : 1.05
    },
    }

#%% Class definition
class GraphStyle:
    def __init__(self, nr_nodes, template = None, node_positions = None, node_labels = None, title = None):
        '''
        '''
        nr_nodes = gen_deepcopy(nr_nodes, 1)[0]
        template = gen_deepcopy(template, 1)[0]
        node_positions = gen_deepcopy(node_positions, 1)[0]
        node_labels = gen_deepcopy(node_labels, 1)[0]
        
        self.nr_nodes = nr_nodes
        if template is None:
            template = whiteonblack
        
        if node_positions is None:
            from Graphstabilizer.graphs.layouts import ring as layout
            node_positions = layout(nr_nodes)
            del layout
        if node_labels is None:
            node_labels = [str(i) for i in range(nr_nodes)]
        
        # Handling of node labels
        if not node_labels is None:
            check_are_nodelabels(self.nr_nodes, node_labels)
        
        
        # Handling of node positions
        if not node_positions is None:
            check_are_nodepositions(self.nr_nodes, node_positions)
        
        
        
        self.node_labels = node_labels
        self.node_positions = node_positions
        
        
        
        self.nodes_style = gen_deepcopy(template['nodes_style'], nr_nodes)
        self.edge_style = template['edge_style']
        self.patch_style = template['patch_style']
        self.figure_style = template['figure_style']
        
        
    #%% Nodes
    ## Radius
    def set_node_radius(self, node_index, radius):
        self.nodes_style[node_index]['node_radius'] = radius
    
    def set_nodes_radii(self, node_radii: list, indices: list = None):
        '''
        Set the radii of the nodes given in the node_radii.
        node_radii is either a:
            float: radius to be given to all nodes in indices
            list: list of radii. If no indices provided, must be of length nr_nodes.
        indices, optional:
            if not None, indices must be of same length as node_radii if that is a list.
        '''
        if indices is None:
            indices = list(range(self.nr_nodes))
            
        if type(node_radii) == float:
            node_radii = [node_radii]*len(indices)
            
        assert len(indices) == len(node_radii), f"{len(node_radii)} radii provided but {len(indices)} indices provided."
        
        for node, radius in zip(indices, node_radii):
            assert isinstance(radius, (int, float)), f"Radius {radius} for node {node} is of type {type(radius)}, not int or float."
            self.set_node_radius(node, radius)
    
    ## Colour
    def set_node_color(self, node_index, color):
        self.nodes_style[node_index]['node_color'] = color
    
    def turn_off_node_fill(self, node_index):
        self.node_color(node_index, 'none')
    
    def turn_off_nodes_fill(self):
        self.set_nodes_colors(node_colors = ['none']*self.nr_nodes)

    def set_nodes_colors(self, node_colors: list, indices: list = None):
        '''
        Set the colors of the nodes given in the node_colors.
        Node_colors is either a:
            str: color specification like 'g' or 'b' or '#F0F0F0. Color will be applied to all nodes in indices
            float: color specification as rgb value like (0.5,0.5,0.5). Color will be applied to all nodes in indices
            list: list of color specifications. If no indices provided, must be of length nr_nodes.
        indices, optional:
            if not None, indices must be of same length as node_colors if that is a list.
        '''
        
        if indices is None:
            indices = list(range(self.nr_nodes))

        # If a list of 3 or 4 floats is provided, it's a colour, and not a list of colours.
        # So it should be converted to a tuple
        if isinstance(node_colors, list) and len(node_colors) in [3,4]:
            if isinstance(node_colors[0], (float, int)):
                # node_colors is now a single color, convert to tuple
                node_colors = tuple(node_colors)
        
        # If node_colors is not a list, its a single color and all indices should be mapped to this color. So convert to a list of the same length
        if not isinstance(node_colors, list):
            node_colors = [node_colors]*len(indices)
            
        assert len(node_colors) == len(indices), f"{len(node_colors)} colors provided but {len(indices)} indices provided."
        
        for node, color in zip(indices, node_colors):
            # if the color is a list (again), node_colors was a list of lists. So we can safely convert the color to tuple
            if isinstance(color, list):
                color = tuple(color)
            
            # Now check if the color is a str or tuple.
            assert isinstance(color, (str, tuple)), f"Color {color} for node {node} is of type {type(color)}, not str or tup."

            # Set the node color
            self.set_node_color(node, color)
    # Labels
    def turn_on_labels(self):
        '''
        Turn on all the labels to be printed
        '''
        self.set_nodes_properties('with_labels', True)
    
    def turn_off_labels(self):
        '''
        Turn off all the labels to be printed
        '''
        self.set_nodes_properties('with_labels', False)
    
    def toggle_labels(self):
        '''
        Toggle the labels between being printed or not. Toggles on the first node label being printed or not.
        '''
        if self.nodes_style[0]['with_labels']:
            self.turn_off_labels()
        else:
            self.turn_on_labels()

    ## Other
    def set_node_property(self, node_index: int, property_name: str, property_value):
        '''
        Set the property for the given node to the given value.
        '''
        
        self.nodes_style[node_index][property_name] = property_value
    
    def set_nodes_properties(self, property_name: str, property_values, nodes_indices = None):
        '''
        Set a node property in the graph style for one or more nodes to a given value.

        Parameters
        ----------
        property_name : str
            The property to set.
        property_values : any
            The value to which the property is set. 
            Either a single value that is applied to all nodes, or a list of values of length nodes_indices.
        nodes_indices : int, list or, None. Optional.
            The nodes to which to set the property value. The default is None.
            If None provided, all nodes are set.
        Returns
        -------
        None.

        '''
        if nodes_indices is None:
            nodes_indices = list(range(self.nr_nodes))
        elif nodes_indices is int:
            nodes_indices = [nodes_indices]
            
        if not isinstance(property_values, list):
            property_values = [property_values]*len(nodes_indices)
        
            
        assert len(nodes_indices) == len(property_values), f" There are {len(nodes_indices)} node indices given but {len(property_values)} properties values to set given."
        
        for node, value in zip(nodes_indices, property_values):
            self.set_node_property(node, property_name, value)
    
    ## Removal
    def remove_node(self, node_index: int):
        '''
        Remove node from the graphstyle.
        '''
        self.nodes_style.pop(node_index)
        self.nr_nodes -= 1
    
    ## Info retrieval
    def get_node_radius(self, node_index):
        return self.nodes_style[node_index]['node_radius']
    
    def get_nodes_radii(self, node_indices: list = None):
        if node_indices is None:
            node_indices = range(self.nr_nodes)
        return [self.get_node_radius(node) for node in node_indices]
    
    def labels_to_graphx_format(self):
        '''
        Return a dictionary of the current node labels, with keys the index of the element in the list.
        '''
        labels_dict = {}
        for index, label in enumerate(self.node_labels):
            labels_dict[index] = label
        return labels_dict
    
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
    
    def get_node_positions(self, selection) -> list:
        '''
        Get the positions for a given selection of nodes.
        Always returns a list, even if only one node given.
        '''
        node_positions = []
        
        for node in selection:
            check_is_node_index(self.nr_nodes, node)
            node_positions.append(self.node_positions[node])
        
        return node_positions
    
    
    #%% Edges
    
    #%% Internal functions
    def _handle_nodeindex_param(self, node):
        '''
        Handle a node parameter. This is either an index for the node, or the associated node label. Returns the index.
        '''
        
        if type(node) == int:
            check_is_node_index(self.nr_nodes, node)
            node_index = node
        elif type(node) == str:
            node_index = self.retrieve_node_index(node)
        else:
            raise TypeError(f"Can't find the node labeled {node} because it's not a string or int.")
        return node_index
    
#%% External helper functions
def gen_deepcopy(original, nr_copies):
    from copy import deepcopy
    copies = []
    for nr in range(nr_copies):
        copies.append(deepcopy(original))
    return copies