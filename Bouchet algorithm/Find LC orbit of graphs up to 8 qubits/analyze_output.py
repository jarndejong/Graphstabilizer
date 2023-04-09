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
# Output path
out_path = os.path.join("OUTPUT")
# Analysis path
analysis_path = os.path.join("ANALYSIS")

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

# Create a dictionary of all .csv files in the OUTPUT folder
LC_orbit_dict = {}
# Loop over all folders in OUTPUT
folder_list = list_folders(out_path)
for folder in folder_list:
    # Loop over all .csv files in each folder and add them to the dictionary
    csv_list = list_csv_files(os.path.join(out_path, folder))
    LC_orbit_dict[int(folder)] = csv_list

# Test LC_orbit_dict
# Check if all orbits are there
orbit_name_list = list(LC_orbit_dict.keys())
orbit_name_list.sort()
if orbit_name_list == [i+1 for i in range(146)]:
    print("All orbits present.")
# Create orbit size dictionary
orbit_size_dict = {}
for key in LC_orbit_dict.keys():
    orbit_size_dict[key] = len(LC_orbit_dict[key])
# print(orbit_size_dict)

# Store orbit sizes in a list
orbit_size_list = [orbit_size_dict[i+1] for i in range(146)]
print(orbit_size_list)
# Save list as csv.file in ANALYSIS folder
print_string = "Class; Size\n"
for i in range(len(orbit_size_list)):
    print_string += str(i+1) + ";" + str(orbit_size_list[i]) + "\n"
print_to_file(print_string, analysis_path, "orbit_size.csv")

