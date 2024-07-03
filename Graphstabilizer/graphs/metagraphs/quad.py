#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:25:55 2023

@author: jarn
"""

import numpy as np


inner_radius = 1
single_layer = 2
double_layer = 3
triple_layer = 5


#%% params
marginal_size = 4
pos_M = [(inner_radius*np.cos(angle*2*np.pi/marginal_size),inner_radius*np.sin(angle*2*np.pi/marginal_size)) for angle in range(marginal_size)]

singlelayersize = 4
doublelayersize = 6
triplelayersize = 4

#%% Potential edges
potential_inner_edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
potential_single_edges = [(0,4),(1,5),(2,6),(3,7)]
potential_edge_pairs = [[(8,0),(8,1)],[(9,0),(9,2)],[(10,1),(10,2)],[(11,2),(11,3)],[(12,1),(12,3)],[(13,0),(13,3)]]
potential_edge_3pairs = [[(14,0),(14,1),(14,2)],[(15,1),(15,2),(15,3)],[(16,0),(16,2),(16,3)],[(17,0),(17,1),(17,3)]]
potential_edge_4pairs = [[(0,18),(1,18),(2,18),(3,18)]]


#%% Node positions
marginal_size = 4
M = tuple(range(marginal_size))
singlelayersize = 4
doublelayersize = 6
triplelayersize = 4

pos_M = [(inner_radius*np.cos(angle*2*np.pi/marginal_size),inner_radius*np.sin(angle*2*np.pi/marginal_size)) for angle in range(marginal_size)]
pos_single = [(single_layer*np.cos(angle*2*np.pi/4),single_layer*np.sin(angle*2*np.pi/marginal_size)) for angle in range(singlelayersize)]
pos_double = [(double_layer*np.cos(angle*2*np.pi/4 + np.pi/4),double_layer*np.sin(angle*2*np.pi/4 + np.pi/4)) for angle in range(4)]
pos_double.insert(-1,(3.5,0))
pos_double.insert(1,(0,3.5))
pos_triple = [(triple_layer*np.cos(angle*2*np.pi/4 + 2*np.pi/4),triple_layer*np.sin(angle*2*np.pi/4 + 2*np.pi/4)) for angle in range(4)]



pos = []
pos.extend(pos_M)
pos.extend(pos_single)
pos.extend(pos_double)
pos.extend(pos_triple)
pos.extend([(0,0)])



#%% Labels
labels = ['$1$','$2$','$3$','$4$']
labels.extend(['$N_{1}$', '$N_{2}$', '$N_{3}$', '$N_{4}$'])
labels.extend(['$N_{12}$', '$N_{13}$', '$N_{23}$', '$N_{34}$', '$N_{24}$', '$N_{14}$'])
labels.extend(['$N_{123}$', '$N_{234}$', '$N_{134}$', '$N_{124}$'])
labels.extend(['$N_{1234}$'])
