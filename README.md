# pySubDisc

pySubDisc is a Python wrapper for [SubDisc: Subgroup Discovery](https://github.com/SubDisc/SubDisc).

## Installation

pySubDisc is available from PyPI. To install it, run:

```bash
python3 -m pip install pysubdisc
```

## Installation from source

To install pySubDisc from its sources, follow these steps:

* From https://github.com/SubDisc/SubDisc, use `mvn package` to build `target/subdisc-gui.jar`
* Place `subdisc-gui.jar` in `src/pysubdisc/jars`
* Run `pip install .` from the root directory of the repository (containing pyproject.toml)

## Example

Using the data from https://github.com/SubDisc/SubDisc/blob/main/adult.txt :

```python
import pysubdisc
import pandas

data = pandas.read_csv('adult.txt')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasureMinimum = 0.25
sd.run()
print(sd.asDataFrame())
```

|    |   Depth |   Coverage |   Quality |   Target Share |   Positives |   p-Value | Conditions                            |
|---:|--------:|-----------:|----------:|---------------:|------------:|----------:|:--------------------------------------|
|  0 |       1 |        443 |  0.517601 |       0.440181 |         195 |       nan | marital-status = 'Married-civ-spouse' |
|  1 |       1 |        376 |  0.453305 |       0.446809 |         168 |       nan | relationship = 'Husband'              |
|  2 |       1 |        327 |  0.359959 |       0.428135 |         140 |       nan | education-num >= 11.0                 |
|  3 |       1 |        616 |  0.354077 |       0.334416 |         206 |       nan | age >= 33.0                           |
|  4 |       1 |        728 |  0.326105 |       0.311813 |         227 |       nan | age >= 29.0                           |
|  5 |       1 |        552 |  0.263425 |       0.317029 |         175 |       nan | education-num >= 10.0                 |

Some detailed examples can be found in the /examples folder.

## Documentation

The SubDisc documentation might be of help for working with pySubDisc: https://github.com/SubDisc/SubDisc/wiki.

### Data loading

pySubDisc uses `pandas.DataFrame` tables as input. There are two options to pass these from pySubDisc to SubDisc itself:

```python
data = pandas.read_csv('adult.txt')

# Either, create a SubgroupDiscovery target structure directly
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')

# Or, first load the dataframe into SubDisc for further preparation
table = pysubdisc.loadDataFrame(data)
sd = pysubdisc.singleNominalTarget(table, 'target', 'gr50K')
```

### Data preparation

A `pySubDisc.Table` object can be manipulated before creating a SubDisc target using the following functions:

```python
# Load a pandas.DataFrame object
table = pysubdisc.loadDataFrame(data)

# Describe the columns (name, type, cardinality, enabled)
print(table.describeColumns())

# Change column type to binary
table.makeColumnsBinary(['column', 'other_column']

# Change column type to numeric
table.makeColumnsNumeric(['column', 'other_column']

# Change column type to nominal
table.makeColumnsNominal(['column', 'other_column']

# Disable columns
table.disableColumns(['column', 'other_column']

# Enable columns
table.enableColumns(['column', 'other_column']

# Select a subset of the rows by passing a pandas boolean Series
table.setSelection(data['education'] == 'Bachelors')

# Reset selection of rows to the full data set
table.clearSelection()
```

### Configuring subgroup discovery

A `pySubDisc.SubgroupDiscovery` object can be created by the following target functions:

```python
# single nominal target
sd = pysubdisc.singleNominalTarget(data, targetColumn, targetValue)

# single numeric target
sd = pysubdisc.singleNumericTarget(data, targetColumn)

# double regression target
sd = pysubdisc.doubleRegressionTarget(data, primaryTargetColumn, secondaryTargetColumn)

# double correlation target
sd = pysubdisc.doubleCorrelationTarget(data, primaryTargetColumn, secondaryTargetColumn)

# double binary target
sd = pysubdisc.doubleBinaryTarget(data, primaryTargetColumn, secondaryTargetColumn)

# multi numeric target
sd = pysubdisc.multiNumericTarget(data, targetColumns)
```

After creating a `pySubDisc.SubgroupDiscovery` object, you can configure its search parameters. For example:

```python
print(sd.describeSearchParameters())

sd.numericStrategy = 'NUMERIC_BEST'
sd.qualityMeasure = 'RELATIVE_WRACC'
sd.qualityMeasureMinimum = 2
sd.searchDepth = 2
```

An appropriate value of the `qualityMeasure` option can in particular be computed for various target types using the `computeThreshold()` function.

```python
# If setAsMinimum is set to True, the qualityMeasureMinimum parameter is updated directly
threshold = sd.computeThreshold(significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100, setAsMinimum=True)
```

### Running subgroup discovery

After configuring the search parameters, you can run the subgroup discovery process by calling the `run()` method.

```python
sd.run()
```

### Examining the results

```python
# The resulting subgroups are given as a pandas.DataFrame, with one row per subgroup
print(sd.asDataFrame())
```


The function `getSubgroupMembers()` returns a set of members of a subgroup as a pandas boolean Series.

```python
# Get rows corresponding to subgroup #0
subset = data[sd.getSubgroupMembers(0)]
```


For a number of the target types, a `showModel()` method is available to aid visualization of the discovered subgroups. The scripts in the `/examples` directory demonstrate its use.


The function `getPatternTeam()` returns a Pattern Team for the discovered subgroups.

```python
# if returnGrouping is True, getPatternTeam will also return
# the grouping of subgroups according to the pattern team
patternTeam, grouping = sd.getPatternTeam(3, returnGrouping=True)

print(patternTeam)

# print the subgroups for the first of the three determined groups
df = sd.asDataFrame()
print(df[grouping[0]])
```
