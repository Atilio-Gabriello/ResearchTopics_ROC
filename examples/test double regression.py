import pysubdisc
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('adult.txt')

sd = pysubdisc.doubleRegressionTarget(data, 'age', 'hours-per-week')
sd.searchDepth = 2

#do the actual run and print the best subgroup
sd.run()
print(sd.asDataFrame().loc[0])

#plot the model for the subgroup compared to that of the entire dataset
df = sd.getModel(0)
plt.figure()
plt.xlabel('age')
plt.ylabel('hours-per-week')
plt.scatter(df['x'], df['y base'], c='lightgrey', s=0.5)
plt.scatter(df['x'], df['y 0'], c='blue', s=0.5)
plt.plot(df['x'], df['pred base'], c='lightgrey', linewidth=1)
plt.plot(df['x'], df['pred 0'], c='blue', linewidth=1)
plt.savefig('double_regression.pdf')