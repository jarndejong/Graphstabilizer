import os, sys
sys.path.insert(0, os.path.abspath("."))
import pickle
import numpy as np
import csv #to read in table contents
import glob #to import colours etc.
import os, sys
import networkx as nx
from copy import deepcopy
import itertools as it
from helper_functions import print_to_file, list_folders, list_csv_files, read_graphs, open_graph_as_nx, edges_to_str

# Time the runtime of the script
import time
start_time = time.time()

# Set filepaths
# Output path
out_path = os.path.join("OUTPUT")
# Check path
check_path = os.path.join("CHECK")

# ### Slow mode (find all the orbits and store them in a dictionary) ###
# # Find all available orbits
# folder_list = list_folders(out_path)
# orbit_list = [int(i) for i in folder_list]
# # print(f"There are {len(orbit_list)} orbits.")
# # Create a dictionary to store the orbits
# LC_orbit_dict = {}
#
# # Loop through all orbits
# for orbit in orbit_list:
#     # Create a list to store the graphs in the orbit
#     orbit_graphs = []
#     # Find all graphs in the orbit
#     graph_list = list_csv_files(os.path.join(out_path, str(orbit)))
#     print(f"There are {len(graph_list)} graphs in orbit {orbit}.")
#     # Read in graphs as networkx graphs
#     for graph in graph_list:
#         nx_graph = open_graph_as_nx(out_path, str(orbit), graph, csv_delimiter=";")
#         # Add string of graph edges to the graph
#         edge_str = edges_to_str(nx_graph)
#         orbit_graphs.append(edge_str)
#     # Store the graphs in the orbit
#     LC_orbit_dict[str(orbit)] = orbit_graphs
#
# # Save the dictionary with pickle
# with open(os.path.join(out_path, "LC_orbit_dict.pickle"), "wb") as fout:
#     pickle.dump(LC_orbit_dict, fout)
#     print(f"LC_orbit_dict saved as pickle file.")

### Fast mode (load the orbits from pickle into a dictionary) ###

# Load the dictionary with pickle
with open(os.path.join(out_path, "LC_orbit_dict.pickle"), "rb") as fin:
    LC_orbit_dict = pickle.load(fin)
    print(f"LC_orbit_dict loaded from pickle file.")

# # Print the number of graphs in each orbit
# for key in LC_orbit_dict.keys():
#     print(f"In orbit {key}, we have {len(LC_orbit_dict[key])} graphs.")


# Create dictionary for the evaluation of results
results_dict = {}
for orbit in range(146):
    orbit += 1 # natural counting
    results_dict[str(orbit)] = []

# Loop through all folders that might be in the Check folder
folder_list_check_path = list_folders(check_path)
for folder_path in folder_list_check_path:
    print(f"Checking graphs in folder {folder_path}.")
    check_files = list_csv_files(os.path.join(check_path,folder_path))
    # print(check_files)
    # Loop through all graphs in the folder with folder_path
    for graph_name in check_files:
        # Open the graph as a networkx graph
        try:# Try to open the graph with ; as delimiter
            nx_graph = open_graph_as_nx(check_path, folder_path, graph_name, csv_delimiter=";")
        except: # If that fails, try with , as delimiter
            nx_graph = open_graph_as_nx(check_path, folder_path, graph_name, csv_delimiter=",")
        edge_str_nx_graph = edges_to_str(nx_graph)
        # Calculate the number of noes of the graph nx_graph
        num_nodes = len(nx_graph.nodes())
        # Find the orbit of the graph
        # Loop through all orbits
        for orbit_str in LC_orbit_dict.keys():
            for edge_str_graph in LC_orbit_dict[orbit_str]:
                # print(f"The other graph has edges {edge_str_graph}.")
                if edge_str_nx_graph == edge_str_graph:
                    # print(f"{graph_name} is in orbit {orbit_str}.")
                    results_dict[orbit_str].append(graph_name)
                    break

for key in results_dict.keys():
    print(f"We have identified {len(results_dict[key])} graphs belonging to orbit {key}.")

# Print runtime
print("--- %s seconds ---" % (time.time() - start_time))