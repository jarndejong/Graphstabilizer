import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import csv #to read in table contents
import glob #to import colours etc.
import os, sys
import networkx as nx
from copy import deepcopy
import itertools as it
import scipy

# # Time the runtime of the script
# import time
# start_time = time.time()

# Set filepaths
# Output path
out_path = os.path.join("OUTPUT")
# Check path
check_path = os.path.join("CHECK")

# Function for writing to file
def print_to_file(string, out_path, out_filename):
    string = str(string)
    with open(os.path.join(out_path, out_filename), "w") as fout:
        fout.write(string)

# Function returning a list of all folders in a directory
def list_folders(path):
    folders = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            folders.append(name)
    return folders

# # Test list_folders
# folder_list = list_folders(out_path)
# folder_list = [int(i) for i in folder_list]
# folder_list.sort()
# print(folder_list)
# if folder_list == [i+1 for i in range(146)]:
#     print("All folders present")

# Function returning a list of all .csv files in a directory
def list_csv_files(path):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".csv"):
                csv_files.append(name)
    return csv_files

# # Test list_csv_files
# csv_list = list_csv_files(os.path.join(out_path, "2"))
# print(csv_list)

# Function for reading in graphs from a folder
def read_graphs(primary_folder_path, folder_path, csv_delimiter = ";"):
    '''
    Read in all graphs from a folder.
    :param primary_folder_path: Path to the folder containing the folder containing the graphs
    :param folder_path: Path to the folder containing the graphs
    :param csv_delimiter: Symbol used to separate the columns in the adjacency matrix .csv files
    :return: list of graph strings representing the folder contents
    '''
    print(f"Now looking though folder '{folder_path}'.")
    check_path = primary_folder_path
    current_path = os.path.join(check_path, folder_path)
    # Loop through and import all .csv files in the folder
    check_files = list_csv_files(current_path)
    # print(f"check_files = {check_files}")
    # print(f"I found {check_files}.")
    # open the csv files
    check_graphs = []
    for file in check_files:
        with open(os.path.join(current_path, file), "r") as fin:
            # Split for every ";" or every ","
            reader = csv.reader(fin, delimiter = csv_delimiter)
            check_graphs.append(list(reader))
    # Convert to nice strings (accounting for the fact that there may be an empty column at the end)
    check_graphs = [str(graph) for graph in check_graphs]
    check_graphs = [graph.replace(", ''", "") for graph in check_graphs]
    return check_graphs
    print(f"There are {len(check_files)} files in this folder.")

# Function for opening graph as a networkx graph from a .csv file
def open_graph_as_nx(primary_folder_path, folder_path, graph_name, csv_delimiter = ";"):
    '''
    Open a graph as a networkx graph from a .csv file.
    :param primary_folder_path: Path to the folder containing the folder containing the graphs
    :param folder_path: Path to the folder containing the graphs
    :param graph_name: Name of the graph file to be opened
    :param csv_delimiter: Symbol used to separate the columns in the adjacency matrix .csv files
    :return: networkx graph
    '''
    with open(os.path.join(primary_folder_path, folder_path, graph_name), "r") as fin:  # first graph always named "1.csv"
        # Split for every ";" or every ","
        reader = csv.reader(fin, delimiter=csv_delimiter)
        # Convert to list
        matrix_list = list(reader)
        #
        for row in matrix_list:
            # Delete empty strings
            while "" in row:
                row.remove("")
        # Convert to numpy array
        matrix = np.array(matrix_list, dtype=int)
        # Convert to networkx graph
        orbit_graph = nx.from_numpy_matrix(matrix)
        # Shift node labels to start at 1
        orbit_graph = nx.convert_node_labels_to_integers(orbit_graph, first_label=1)
        return orbit_graph

### Foliage partition functions ###
# Function converting networkx graph to adjacency matrix usable for the foliage functions
def nx_to_adjacency_matrix(graph):
    '''
    Convert a networkx graph to an adjacency matrix.
    :param graph: networkx graph
    :return: adjacency matrix as list of lists
    '''
    # Convert networkx graph to adjacency matrix
    adj_matrix = nx.to_numpy_matrix(graph)
    # Convert entries into integers
    adj_matrix = adj_matrix.astype(int)
    # Convert adjacency matrix to list of lists
    adj_list = adj_matrix.tolist()
    return adj_list

# function for converting np.array to string with ; as separator
def np2csv(matrix):
    '''
    Convert a numpy array to a string with ";" as separator.
    :param matrix: numpy array adjacency matrix
    :return: string for exporting the adjacency matrix as .csv file
    '''
    export_string = ""
    for row in matrix:
        for element in row:
            export_string += str(element) + ";"
        export_string = export_string[:-1] # remove last ;
        export_string += "\n"
    export_string = export_string[:-1] # remove last \n
    return export_string

# Function for finding "1" in a list
def find_ones(list):
    '''
    Find the positions of "1" in a list.
    :param list: Input list
    :return: List of positions of "1" in the input list
    '''
    positions = []
    for i in range(len(list)):
        if list[i] == 1.0:
            positions.append(i)
    return positions

# Function for finding nodes with one neighbour in a graph
def find_leaves(matrix):
    '''
    Find the leaves of a graph. Counting starts at 1.
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :return: list of nodes that are the leaves of the graph
    '''
    M = matrix
    n = len(M)
    leaves = []
    for i in range(n):
        gamma_i = M[i]
        one_positions = find_ones(gamma_i)
        if len(one_positions) == 1:
            leaves.append(i+1)# natural counting
    return(leaves)

# Function for finding the neighbours of a node
def find_neighbours(matrix, node):
    '''
    Find the neighbours of a node in a graph. Counting starts at 1.
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :param node: node for which the neighbours are to be found
    :return: list of neighbours of the node "node" in the graph defined by "matrix"
    '''
    M = matrix
    n = len(M)
    gamma_node = M[node-1]# natural counting
    one_positions = find_ones(gamma_node)
    neighbours = [i+1 for i in one_positions]
    return(neighbours)

# Function for calculating axil and leaves partitions
def find_axil_leaves_sets(matrix):
    '''
    Find those partitions of the graph vertex set that contain one axil and all its leaves.
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :return: list of lists of the form [axil, leaf_1, leaf_2, ...]
    '''
    M = matrix
    n = len(M)
    leaves = find_leaves(M)
    axil_list = []
    for leaf in leaves:
        neighbours = find_neighbours(M, leaf)
        axil_list.extend(neighbours)
    # Remove duplicates
    axil_list = list(set(axil_list))
    # Construct the sets containing leafs and their axil
    axil_leaves_sets = []
    for axil in axil_list:
        new_set = find_neighbours(M, axil)
        # intersect new_set with leaves
        new_set = list(set(new_set) & set(leaves))
        new_set.append(axil)
        new_set.sort()
        # Check if new_set already contained in axil_leaves_sets
        if new_set not in axil_leaves_sets:
            axil_leaves_sets.append(new_set)
    return(axil_leaves_sets)

# Function for calculating potential twins (vertices that are not part of an axil-leaves type partition)
def potential_twins(matrix):
    '''
    Find the vertices that are not part of an axil-leaves type partition.
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :return: list of vertices that are not part of an axil-leaves type partition
    '''
    M = matrix
    n = len(M)
    # First calculate the axil-leaves sets
    axil_leaves_sets = find_axil_leaves_sets(M)
    # Join the sets in axil_leaves_sets to vertices_to_remove
    vertices_to_remove = []
    if len(axil_leaves_sets) == 1:
        vertices_to_remove = axil_leaves_sets[0]
    if len(axil_leaves_sets) > 1:
        for set in range(len(axil_leaves_sets)):
            vertices_to_remove.extend(axil_leaves_sets[set])
    vertices_to_remove.sort()
    # Calculate the potential twins
    potential_twins = [i+1 for i in range(n)]
    for vertex in vertices_to_remove:
        potential_twins.remove(vertex)
    return potential_twins

# Function for calculating twin and trivial partitions (that are not of the axil_leaves type)
def twin_and_trivial_partitions(matrix):
    '''
    Find the twin and trivial partitions of the graph defined by the adjacency matrix "matrix".
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :return: list of lists where each list is a list of twins or a trivial partition containing only one element
    '''
    M = matrix
    n = len(M)
    # Calculate the potential twins
    potential_twins_list = potential_twins(M)
    # Check if the potential twins are twins
    twin_partitions = [] # return this if there are no twins (for example if len(potential_twin_list) < 2)
    # Create a dictionary with the neighbors of the potential twins
    neighbors_dict = {}
    for twin in potential_twins_list:
        neighbors_dict[twin] = find_neighbours(M, twin)
    # Group the potential twins according to their neighborhoods
    for v in potential_twins_list:
        current_partition = [v]
        other_potential_twins = deepcopy(potential_twins_list)
        other_potential_twins.remove(v)
        for w in other_potential_twins:
            neighbors_v = deepcopy(neighbors_dict[v])
            if w in neighbors_v:
                neighbors_v.remove(w)
            neighbors_w = deepcopy(neighbors_dict[w])
            if v in neighbors_w:
                neighbors_w.remove(v)
            if neighbors_v == neighbors_w:
                current_partition.append(w)
            current_partition.sort()
        if current_partition not in twin_partitions:
            twin_partitions.append(current_partition)
    return(twin_partitions)

# Function for calculating the foliage partition
def foliage_partition(matrix):
    '''
    Find the foliage partition of the graph defined by the adjacency matrix "matrix".
    :param matrix: adjacency matrix as list of lists representing the graph's adjacency matrix
    :return: list of lists where each list is a list of vertices that in the same partition
    '''
    M = matrix
    # Calculate the axil-leaves sets
    axil_leaves_sets = find_axil_leaves_sets(M)
    # Calculate twin and trivial partitions
    twin_and_trivial_partitions_list = twin_and_trivial_partitions(M)
    # Combine the axil-leaves sets and the twin and trivial partitions
    foliage_sets = []
    foliage_sets.extend(axil_leaves_sets)
    foliage_sets.extend(twin_and_trivial_partitions_list)
    foliage_sets.sort()
    return(foliage_sets)

# Function to store the edges of a graph as a string
def edges_to_str(nx_graph):
    '''
    Read in a networkx graph and return the edges as a string.
    :param nx_graph: input graph in networkx format
    :return: edges as a string in canonical order
    '''
    edges = list(nx_graph.edges())
    edges_new = []
    for edge in edges:
        edge = list(edge)
        edge.sort()
        edges_new.append(edge)
    edges_new.sort()
    edges_str = ""
    for edge in edges_new:
        edges_str += str(edge[0]) + str(edge[1])
    return edges_str
