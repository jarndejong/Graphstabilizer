import os, sys
import numpy as np
import csv #to export stabilizer
import networkx as nx


# Give the project a name
project_name = "example"

# Set filepaths
data_path = os.path.join("")
filename_id = str(project_name) + ".txt"
csv_path = os.path.join("")
project_name = "Stabilizer_" + project_name
out_filename = str(project_name) + ".csv"

# Content input from txt
input_string = ""
with open(os.path.join(data_path, filename_id)) as txt_file:
    for row in txt_file:
        row = row.replace("I", "_") #In case other format is used
        row = row.replace(" ", "") #Remove any spaces that might be in there
        row_list = [row[i] for i in range(len(row))]
        row = ";".join(row_list)
        row = row.replace(";\n", "\n")
        row = row.replace("_", " ")
        input_string += row

# # Content input networkx graph
# edges = [(1,2),(1,3),(1,4),(1,5),(1,6)]
# G = nx.Graph()
# G.add_edges_from(edges)
# adjacency = nx.to_numpy_array(G)
# for i in range(len(adjacency)):
#     adjacency[i,i] = 7. # placeholder for the X support in stabilizer
# adj_str = np.array2string(adjacency)
# adj_str = adj_str.replace(".",";") # add separators
# adj_str = adj_str.replace("[","+") #remove a few things
# adj_str = adj_str.replace("++","+")
# adj_str = adj_str.replace("+","+;") 
# adj_str = adj_str.replace("]","\n")
# adj_str = adj_str.replace(" ","")
# adj_str = adj_str.replace(";\n\n","\n")
# adj_str = adj_str.replace("1","Z") # put in the Z support
# adj_str = adj_str.replace("7","X") # put in the X support
# adj_str = adj_str.replace("0"," ") # put in the _ support
# print(adj_str)
# input_string = adj_str



# Create output file
def print_to_file(string):
    string = str(string)
    with open(os.path.join(csv_path, out_filename), "w") as fout:
        fout.write(string)



# Generate csv table
print_to_file(input_string)