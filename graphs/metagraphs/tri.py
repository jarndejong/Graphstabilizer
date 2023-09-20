#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:25:55 2023

@author: jarn
"""

import numpy as np



#%% Params
outer_radius = 2.5
inner_radius = 1


#%% Edges
potential_inner_edges = [(0,1),(1,2),(0,2)]
potential_single_edges = [(0,3),(1,4),(2,5)]
potential_double_edges = [[(0,6),(1,6)],[(1,7),(2,7)],[(0,8),(2,8)]]
potential_triple_edges = [[(0,9),(1,9),(2,9)]]

#%% Node positions
pos_M = [(inner_radius*np.cos(angle*2*np.pi/3),inner_radius*np.sin(angle*2*np.pi/3)) for angle in range(3)]

pos_single = [(outer_radius*np.cos(angle*2*np.pi/3),outer_radius*np.sin(angle*2*np.pi/3)) for angle in range(3)]

pos_double = [(outer_radius*np.cos(angle*2*np.pi/3 + np.pi/3),outer_radius*np.sin(angle*2*np.pi/3 + np.pi/3)) for angle in [0,1,2]]

pos = []
pos.extend(pos_M)
pos.extend(pos_single)
pos.extend(pos_double)
pos.extend([(0,0)])

del pos_M, pos_single, pos_double

#%% labels
labels = ['$A$','$B$','$C$','$a$','$b$','$c$','$ab$','$bc$','$ac$','$abc$']