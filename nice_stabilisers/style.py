# color of grid lines
girdcolor = "white"
# color of text
celltextcolor = "green" # Set the default color
celltextdict = {} # modify for paricular entries
celltextdict["X"] = "white"
celltextdict["Z"] = "white"
celltextdict["Y"] = "white"

# Dictionary with special colors based on contend
colordict = {}
colordict["X"] = "green"
colordict["Z"] = "green"
colordict["Y"] = "green"

# Dictionary with saturation based on contend
saturationdict = {}
saturationdict["X"] = "100"
saturationdict["Z"] = "50"
saturationdict["Y"] = "75"

### Column styles
## Style for stabilizers
hr = .505 #height of row general
wc = .27 #1.4142 #width of column general
# rows and colums that should be different
special_rows = []
special_columns = [0]
# modifications to heights and widths
special_row_heights = []
special_column_widths = [1*wc]
def row_function(nr):
	return [str(100) for i in range(nr)] # Standard saturation background
def column_function(nc):
	return ["white"]*(nc) # Standard color background
# TODO: Fix error that is caused by the standard color and saturation taken together
# being shorter as a string than the names of the custom colors.

