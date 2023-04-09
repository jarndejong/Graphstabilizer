import os, sys
from turtle import color; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import csv #to read in table contents
import glob #to import colours etc.

# Give the project a name
project_name = "Stabilizer_example"

# Set filepaths
data_path = os.path.join("DATA")
filename_id = str(project_name) + ".csv"
import_path = os.path.join("IMPORTS")
tikz_path = os.path.join("tikz")
out_filename = str(project_name) + ".tex"

# Content input
input_matrix = []
with open(os.path.join(data_path, filename_id)) as csv_file:
    for row in csv_file:
        row_string = str(row)
        row_list = row_string.split(";")
        row_list[-1] = row_list[-1].replace("\n", "")
        input_matrix.append(row_list)
M = np.array(input_matrix)
nr = len(M) #number of rows
nc = len(M[0]) #number of columns

# Local style import
from style import *
# Default style
hr_list = [hr for i in range(nr)] #default row heights
wc_list = [wc for i in range(nc)]   #default column widths
# modify according to inputs
for index, row in enumerate(special_rows):
    hr_list[row] = special_row_heights[index]
for index, column in enumerate(special_columns):
    wc_list[column] = special_column_widths[index]
# prepare for tikz (colmatrix holds the color information)
rowcollist = row_function(nr)
columncollist = column_function(nc)
if rowcollist[0][0].isdigit():#style where columns have a uniform color
    colmatrix = [[columncollist[i] + "!" + rowcollist[j] for i in range(nc)] for j in range(nr)]
    colmatrix = np.array(colmatrix)
else:#style where rows have a uniform color
    colmatrix = [[rowcollist[i] + "!" + columncollist[j] for i in range(nr)] for j in range(nc)]
    colmatrix = np.transpose(colmatrix)
    colmatrix = np.array(colmatrix)
# Replace with special colors based on contend
for i in range(nr):
    for j in range(nc):
        # print(f"row {i+1}, column {j+1}: {M[i][j]}")
        if M[i][j] in colordict.keys():
            # print(f"Standard color would be: {colmatrix[i][j]}.")
            # print(f"I will color {M[i][j]} in {colordict[M[i][j]]}.")
            try:
                saturation = saturationdict[M[i][j]]
            except:
                print("Error: Saturation needs to be defined for all special colors.")
            # saturation = colmatrix[i][j].split("!")[1]
            # print(saturation)
            colmatrix[i][j] = colordict[M[i][j]] + "!" + str(saturation)
            # print(f"The new color is: {colmatrix[i][j]}")
        # print(colmatrix[i][j].split("!"))
# print(colmatrix)

# Create output file
def print_to_file(string):
    string = str(string)
    with open(os.path.join(tikz_path, out_filename), "w") as fout:
        fout.write(string)

# LaTeX document setup
empty = str()
filenames = [file[8:] for file in glob.glob(os.path.join(import_path, '*.tex'))]
filenames.sort()
for file in filenames:
    with open(os.path.join(import_path, file)) as tex_file:
        for line in tex_file:
            line = str(line)
            empty += line
    empty += "\n\n"

# Generate tikz table
empty += "\\&egin{document}\n"
empty += "\\&egin{tikzpicture}\n\matrix (M) [\n\tmatrix of nodes,\n\tminimum width=1cm,\n\tminimum height=1cm,\n\tcolumn sep=0mm,\n\trow sep=0mm,\n"
empty = empty.replace("&","b")
empty += "\tnodes={\n\t\tdraw,\n\t\tcolor = &,\n\t\tline width=0.07mm,\n\t\tanchor=south,\n\t\talign=center,\n\t},\n"
empty = empty.replace("&", girdcolor) # put in color of grid from style (also default color of writing)
for row in range(nr):
    empty += "\trow " + str(row+1) +"/.style={\n" + "\t\tnodes={\n" + "\t\t\tminimum height=" + str(hr_list[row]) + "cm,\n" + "\t\t}\n" + "\t},\n"
for column in range(nc):
    empty += "\tcolumn " + str(column+1) +"/.style={\n" + "\t\tnodes={\n" + "\t\t\ttext width=" + str(wc_list[column]) + "cm,\n"+ "\t\t\tminimum width=" + str(wc) + "cm,\n" + "\t\t}\n" + "\t},\n"
empty = empty[:-2]
empty += "\n]{\n"
for row in range(nr):
    for column in range(nc):
        if column != 0:
            empty += "\t&"
        empty += "|[fill="+ colmatrix[row, column] + "]|" + "\\textcolor{farbe&}{$" + str(M[row, column]) + "$}" + "\n"
        if M[row][column] in celltextdict.keys():
            empty = empty.replace("farbe&", celltextdict[M[row][column]])
        else:
            empty = empty.replace("farbe&", celltextcolor)
    empty += "\\§\\§"
    empty = empty.replace("§","")
empty += "\n};\n"
empty += "\\§nd{tikzpicture}\n"
empty += "\\§nd{document}"
empty = empty.replace("§","e")

print_to_file(empty)