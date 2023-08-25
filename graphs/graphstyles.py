#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:10:27 2023

@author: jarn
"""
from Graphstabilizer.checkers.graphs import check_are_nodelabels, check_are_nodepositions
from Graphstabilizer.checkers.elementary import check_is_node_index



class GraphStyle:
    def __init__(self, nr_nodes, template = None, node_positions = None, node_labels = None):
        '''
        '''
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
    
    def set_nodes_radii(self, node_radii: list):
        '''
        '''
        assert self.nr_nodes == len(node_radii), f"Graphstyle has {self.nr_nodes} nodes but the given radii list has {len(node_radii)} entries."
        
        for node, radius in enumerate(node_radii):
            assert isinstance(radius, (int, float)), f"Radius {radius} for node {node} is of type {type(radius)}, not int or float."
            self.set_node_radius(node, radius)
    
    ## Colour
    def set_node_color(self, node_index, color):
        self.nodes_style[node_index]['node_color'] = color
    
    def set_nodes_colors(self, node_colors: list):
        '''
        '''
        assert self.nr_nodes == len(node_colors), f"Graphstyle has {self.nr_nodes} nodes but the given colors list has {len(node_colors)} entries."
        
        for node, color in enumerate(node_colors):
            assert isinstance(color, (str, tuple, list)), f"Color {color} for node {node} is of type {type(color)}, not list, typ or str."
            self.set_node_color(node, color)
    
    ## Other
    def set_node_property(self, node_index: int, property_name: str, property_value):
        '''
        Set the property for the given node to the given value.
        '''
        
        self.nodes_style[node_index][property_name] = property_value
    
    def set_nodes_property(self, nodes_indices: list, property_name: str, property_values):
        '''
        '''
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
    
    def get_nodes_radii(self, node_indices: list):
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
    
#%% Templates   
whiteonblack = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.15,
        'node_color'                : 'k',
        'node_edgecolor'            : 'w',
        'node_edgewidth'            : 2,
        'label_color'               : 'w',
        'label_fontsize'            : 14,
        'tex_for_labels'            : True,
    },
    'edge_style' :          {
        'edge_color'                : 'w',
        'edge_width'                : 2,
        'edge_offset'               : 0.2,
        'edge_fontsize'             : 16,
    },
    'figure_style' :        {
        'figure_multiplier'         : 2,
        'figure_tightness'          : 0.01,
        'figure_offsetequal'        : True,
        'background_color'          : 'k',
        'axes_limits'               : None,
        'figure_square'             : False,
        'figure_title'              : {
            'label'                 : None,
            'fontdict'              : {'fontsize': 20, 'color': 'w', 'verticalalignment': 'baseline',
                                     # 'fontweight': 'b',
                                     # 'horizontalalignment': loc
                                     },
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.1,
        'face_color'                : 'r',
        'edge_color'                : 'r',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0,
        'tightness'                 : 1.05
    },
    }



blackonwhite = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.15,
        'node_color'                : 'w',
        'node_edgecolor'            : 'k',
        'node_edgewidth'            : 2,
        'label_color'               : 'k',
        'label_fontsize'            : 14,
        'tex_for_labels'            : True,
    },
    'edge_style' :          {
        'edge_color'                : 'k',
        'edge_width'                : 2,
        'edge_offset'               : 0.2,
        'edge_fontsize'             : 16,
    },
    'figure_style' :        {
        'figure_multiplier'         : 2,
        'figure_tightness'          : 0.01,
        'figure_offsetequal'        : True,
        'background_color'          : 'w',
        'axes_limits'               : None,
        'figure_square'             : False,
        'figure_title'              : {
                    'label'                 : None,
                    'fontdict'              : {'fontsize': 30, 'color': 'k', 'verticalalignment': 'baseline',
                                             # 'fontweight': 'b',
                                             # 'horizontalalignment': loc
                                             },
                    'pad'                   : 10,
                                        },
    },
    'patch_style' :        {
        'face_alpha'                : 0.1,
        'face_color'                : 'r',
        'edge_color'                : 'r',
        'edge_width'                : 2,
        'edge_style'                : '-',
        'padding'                   : 0,
        'tightness'                 : 1.05
    },
    }

metagraphstyle = {
    'nodes_style' :           {
        'with_labels'               : True,
        'node_radius'               : 0.25,
        'node_color'                : 'w',
        'node_edgecolor'            : 'k',
        'node_edgewidth'            : 3,
        'label_color'               : 'k',
        'label_fontsize'            : 14,
        'tex_for_labels'            : True,
    },
    'edge_style' :          {
        'edge_color'                : 'k',
        'edge_width'                : 2,
        'edge_offset'               : 0.35,
        'edge_fontsize'             : 16,
    },
    'figure_style' :        {
        'figure_multiplier'         : 2.5,
        'figure_tightness'          : 0.1,
        'figure_offsetequal'        : True,
        'background_color'          : 'w',
        'axes_limits'               : None,
        'figure_square'             : True,
        'figure_title'              : None,
    },
    'patch_style' :        {
        'face_alpha'                : 0.2,
        'face_color'                : 'r',
        'edge_color'                : 'r',
        'edge_width'                : 0.5,
        'edge_style'                : '-',
        'padding'                   : 0,
        'tightness'                 : 1.05
    },
    }

def gen_deepcopy(original, nr_copies):
    from copy import deepcopy
    copies = []
    for nr in range(nr_copies):
        copies.append(deepcopy(original))
    return copies