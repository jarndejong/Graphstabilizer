1 Put all graphs whose LC orbits you want to generate in the DATA folder

optional (already done here): 2 Run generate.py -> This creates a folder in OUTPUT for every graph in the DATA folder. These newly created folders contain all graphs that the representative is LC equivalent to.
*** TAKES ABOUT 3 HOURS *** (with iteration in range(1, 100000) and Monte Carlo sampling between 1 and 3*n LCs with potential repititions)

3 Run checker.py to find out which orbits the graphs in CHECK belong to