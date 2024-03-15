#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:44:38 2024

@author: jarn
"""

import numpy as np



#%% Params
outer_radius = 1.7
inner_radius = 1.4


#%% Edges
potential_inner_edges = [(0,1)]
potential_single_edges = [(0,2),(1,3)]
potential_double_edges = [[(0,4),(1,4)]]


#%% Node positions
# Option 1
pos_M = [(-1*inner_radius, 0), (inner_radius, 0)]
pos_single = [(-1*inner_radius, -outer_radius), (inner_radius, -outer_radius)]
pos_double = [(0, outer_radius)]

# # Option 2
# pos_M = [(-1*inner_radius, 0), (inner_radius, 0)]
# pos_single = [(-1*inner_radius - outer_radius, 0), (inner_radius + outer_radius, 0)]
# pos_double = [(0, outer_radius)]

# # Option 3
# pos_M = [(-1*inner_radius, 0), (inner_radius, 0)]

# pos_single = [(-(1/2)*inner_radius - outer_radius, outer_radius), ((1/2)*inner_radius + outer_radius, outer_radius)]

# pos_double = [(0, -outer_radius)]

pos = []
pos.extend(pos_M)
pos.extend(pos_single)
pos.extend(pos_double)

del pos_M, pos_single, pos_double

#%% labels
node_labels = ['$A$','$B$','$a$','$b$','$ab$']