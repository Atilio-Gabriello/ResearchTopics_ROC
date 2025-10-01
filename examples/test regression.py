import pysubdisc
import matplotlib.pyplot as plt
import pandas

data = pandas.read_csv('adult.txt')
sd = pysubdisc.singleNumericTarget(data, 'age') #using age as a (somewhat useless) target
sd.searchDepth = 1
sd.numericStrategy = 'NUMERIC_BEST'
sd.qualityMeasure = 'EXPLAINED_VARIANCE'
sd.qualityMeasureMinimum = 0.01

#do the actual run and print the findings
sd.run()
print(sd.asDataFrame())

#set up the plot and plot the age distribution on the entire dataset
model = sd.getModel(0, relative=True)
model.plot()
plt.xlabel('age')
plt.savefig('age-distribution.pdf')
