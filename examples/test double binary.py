import pysubdisc
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('tennisdata.txt')

# this setting assumes two binary target columns. The second one of them is a regular target column, such as you would expect in a typical
# classification problem. The first target column identifies two datasets, where subgroups found in the one are compared to those found 
# in the other. For example, Relative WRAcc favours subgroups that have a high WRAcc in the one dataset, and low in the other.

table = pysubdisc.loadDataFrame(data)
table.makeColumnsBinary(['Dataset', 'Target'])
print(table.describeColumns())


sd = pysubdisc.doubleBinaryTarget(table, 'Dataset', 'Target')

sd.numericStrategy = 'NUMERIC_BEST'
sd.qualityMeasure = 'RELATIVE_WRACC'
sd.qualityMeasureMinimum = 2
sd.searchDepth = 2

#do the actual run and print the best subgroup
sd.run()
print(sd.asDataFrame())
