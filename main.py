import pysubdisc as subdisc
import pandas as pd

data = pd.read_csv('./tests/adult.txt')
sd = subdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.searchStratregy = 'ROC_BEAM'
sd.qualityMeasureMinimum = 0.25
sd.searchDepth = 4
sd.run()
results = sd.asDataFrame()
results = pd.DataFrame(results)
# print(sd.asDataFrame())
print(results[results['Depth'] == 4])
print(results)

