import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import csv #to read in table contents
import glob #to import colours etc.
import os, sys
import networkx as nx
from copy import deepcopy
import itertools as it

# Set filepaths
# Graph representative data
data_path = os.path.join("DATA")
# Output path
out_path = os.path.join("OUTPUT")

# Function for writing to file
def print_to_file(string, out_path, out_filename):
    string = str(string)
    with open(os.path.join(out_path, out_filename), "w") as fout:
        fout.write(string)

# convert matrix into string
def matrix_2_string(adjacency):
    adj_str = np.array2string(adjacency)
    adj_str = adj_str.replace(".",";") # add separators
    adj_str = adj_str.replace("[","+") #remove a few things
    adj_str = adj_str.replace("++","+")
    adj_str = adj_str.replace("+","")
    adj_str = adj_str.replace("]","\n")
    adj_str = adj_str.replace(" ","")
    adj_str = adj_str.replace(";\n\n","\n")
    input_string = adj_str
    return input_string

# function for converting np.array to string with ; as separator
def np2csv(matrix):
    export_string = ""
    for row in matrix:
        for element in row:
            export_string += str(element) + ";"
        export_string = export_string[:-1] # remove last ;
        export_string += "\n"
    export_string = export_string[:-1] # remove last \n
    return export_string

# function for converting np.array adjacency matrix to networkx graph
def adj2nxgraph(matrix):
    G = nx.Graph()
    # add nodes
    G.add_nodes_from(range(len(matrix)))
    # extract edges from adjacency matrix
    edges = []
    for i in range(len(matrix)):
        # only look at upper triangle
        for j in range(i+1, len(matrix)):
            if int(matrix[i][j]) == 1:
                edges.append((i,j))
    # add edges to graph
    G.add_edges_from(edges)
    return G

# local complementation for orbit generation
def local_complementation(graph, vertex):
	"""
	Performs a local complementation on a vertex in graph.

	Args:
		vertex (int): Index of vertex.
	"""
	graph = deepcopy(graph)
	neighborhood = graph.neighbors(vertex)
	neighborhood_edges = it.combinations(neighborhood, 2)
	for v1, v2 in neighborhood_edges:
		if graph.has_edge(v1, v2):
			graph.remove_edge(v1, v2)
		else:
			graph.add_edge(v1, v2, weight=1)
	return graph

# function for multiple local complementations
def multiple_local_complementations(graph, vertex_list):
    new_graph = deepcopy(graph)
    for vertex in vertex_list:
        new_graph = local_complementation(new_graph, vertex)
    return new_graph


# Search for available adjacency matrices in the csv format
adjacency_names = [file[5:] for file in glob.glob(os.path.join(data_path, '*.csv'))]
adjacency_names.sort()
print(adjacency_names)

for filename_id in adjacency_names: # takes about 3h for the first 146 graphs with iteration in range(1, 100000)
    print(f"Now at graph for {filename_id}:")
    # set export file name
    idx = 1
    out_filename = str(idx) + ".csv"
    # Content input
    input_matrix = []
    with open(os.path.join(data_path, filename_id)) as csv_file:
        for row in csv_file:
            row_string = str(row)
            row_list = row_string.split(",")
            row_list[-1] = row_list[-1].replace("\n", "")
            input_matrix.append(row_list)
    M = np.array(input_matrix)
    n = len(M) #dimension of adjacency matrix

    # Create folder for output
    if not os.path.exists(os.path.join(out_path, filename_id[:-4])):
        os.makedirs(os.path.join(out_path, filename_id[:-4]))
    # Set current_out_path
    current_out_path = os.path.join(out_path, filename_id[:-4])

    # String for exporting initial graph
    current_graph = np2csv(M)
    # store the csv file in the OUTPUT folder
    print_to_file(current_graph, current_out_path, out_filename)

    # Convert to networkx graph
    G = adj2nxgraph(M)

    ### Generate orbits
    # Initialize vertex list for local complementation
    lc_list = []
    # Initialize orbit dictionary
    orbit_dict = {}
    orbit_dict["0_init"] = matrix_2_string(nx.to_numpy_array(G))
    for iteration in range(1, 100000):# set the threshold for the number of local complementations
        # Randomly select a list of vertices
        # Randomly select a number of vertices such that no vertex repeats
        num_vertices = np.random.randint(1, 3*n)
        lc_list = np.random.choice(range(n), size=num_vertices, replace=True)
        # Locally complement with respect to all vertices in lc_list
        H = multiple_local_complementations(G, lc_list)
        H_string = matrix_2_string(nx.to_numpy_array(H))
        # Add to orbit dictionary if not already present
        if H_string not in orbit_dict.values():
            orbit_dict[str(lc_list)] = H_string
            # Convert back to numpy array
            current_M = nx.to_numpy_array(H)
            # Change entries of current_M to integers
            current_M = current_M.astype(int)
            # increase index and store
            idx += 1
            out_filename = str(idx) + "_LC_on_" + str(lc_list) + ".csv"
            current_graph = np2csv(current_M)
            print_to_file(current_graph, current_out_path, out_filename)

