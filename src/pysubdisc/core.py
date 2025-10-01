
# Core logic for pySubDisc: Table and SubgroupDiscovery classes, result handling, and model utilities
from .java import ensureJVMStarted, redirectSystemOutErr


# Table wraps a Java Table object and provides column manipulation and selection utilities
class Table(object):
  def __init__(self, table, index):
    """
    Initialize a Table object.
    table: Java Table object
    index: pandas Index for row alignment
    """
    ensureJVMStarted()
    self._table = table
    self._index = index
    self._selection = None
    self._selectionIndex = None


  def __str__(self):
    """String representation showing table size and selection."""
    t = f"{self._table.getNrRows()}x{self._table.getNrColumns()} SubDisc table"
    if self._selectionIndex is not None:
      t = t + f" with {len(self._selectionIndex)} rows selected"
    return t

  def _setColumnType(self, columns, t):
    """Internal: Set the type of one or more columns."""
    if isinstance(columns, str) or not hasattr(columns, '__iter__'):
      columns = [ columns ]
    for c in columns:
      if not self._table.getColumn(c).setType(t):
        raise RuntimeError(f"Failed to change type of column '{c}'")

  def _setColumnEnabled(self, columns, t):
    """Internal: Enable or disable one or more columns."""
    if isinstance(columns, str) or not hasattr(columns, '__iter__'):
      columns = [ columns ]
    for c in columns:
      self._table.getColumn(c).setIsEnabled(t)

  def makeColumnsBinary(self, columns):
    """Change the type of one or more columns to binary."""
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.BINARY)

  def makeColumnsNominal(self, columns):
    """Change the type of one or more columns to nominal."""
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.NOMINAL)

  def makeColumnsNumeric(self, columns):
    """Change the type of one or more columns to numeric."""
    from nl.liacs.subdisc import AttributeType
    self._setColumnType(columns, AttributeType.NUMERIC)

  def enableColumns(self, columns):
    """Enable one or more columns for subgroup discovery."""
    self._setColumnEnabled(columns, True)

  def disableColumns(self, columns):
    """Disable one or more columns for subgroup discovery."""
    self._setColumnEnabled(columns, False)

  def describeColumns(self):
    """Describe the columns/attributes in the table. Returns a DataFrame."""
    import pandas as pd
    L = [ [ str(column.getName()), column.getCardinality(), str(column.getType()), bool(column.getIsEnabled()) ] for column in self._table.getColumns() ]

    df = pd.DataFrame(L, columns=['Attribute', 'Cardinality', 'Type', 'Enabled'])
    return df

  def setSelection(self, subset):
    """Set selection of rows to use for subgroup discovery (mask as pandas Series)."""
    if not subset.index.equals(self._index):
      raise IndexError("Index of subset doesn't match data index")
    from java.util import BitSet
    import pandas as pd
    nrows = self._table.getNrRows()

    # re-index this to a standard RangeIndex to match the java Table
    d = subset.set_axis(pd.RangeIndex(nrows))
    self._selection = BitSet(nrows)
    for i in range(nrows):
      if d[i]:
        self._selection.set(i)

    self._selectionIndex = self._index[subset]

  def clearSelection(self):
    """Reset selection of rows to the full data set."""
    self._selection = None
    self._selectionIndex = None



# SubgroupDiscovery configures and runs the subgroup discovery process, and provides result utilities
class SubgroupDiscovery(object):
  def __init__(self, targetConcept, table):
    """
    Initialize a SubgroupDiscovery object.
    targetConcept: Java TargetConcept object
    table: Table object (Python wrapper)
    """
    ensureJVMStarted()
    self._targetConcept = targetConcept
    self._table = table._table
    self._index = table._index
    self._selection = table._selection
    self._selectionIndex = table._selectionIndex
    self._runCalled = False


  @property
  def targetType(self):
    """Return the type of the target (as string)."""
    return str(self._targetConcept.getTargetType())

  def _initSearchParameters(self, *, qualityMeasure='CORTANA_QUALITY', searchDepth=1, minimumCoverage=None, maximumCoverageFraction=0.9, minimumSupport=0, maximumSubgroups=1000, filterSubgroups=True, minimumImprovement=0.0, maximumTime=0, searchStrategy='BEAM', nominalSets=False, numericOperatorSetting='NORMAL', numericStrategy='NUMERIC_BINS', searchStrategyWidth=100, nrBins=8, nrThreads=None):
    """
    Initialize search parameters for subgroup discovery.
    All arguments are configurable search settings.
    """
    # use inspect to avoid duplicating the argument list
    from inspect import signature
    sig = signature(self._initSearchParameters)
    for arg in sig.parameters:
      if arg == 'self':
        continue
      setattr(self, arg, locals()[arg])

    from nl.liacs.subdisc import QM
    if not isinstance(qualityMeasure, QM):
      if hasattr(QM, qualityMeasure.upper()):
        qualityMeasure = getattr(QM, qualityMeasure.upper())
      else:
        raise ValueError("Invalid qualityMeasure")
    self.qualityMeasureMinimum = float(str(qualityMeasure.MEASURE_DEFAULT))

    if nrThreads is None:
      import os
      nrThreads = os.cpu_count()
      if nrThreads is not None:
        self.nrThreads = nrThreads
      else:
        self.nrThreads = 1

    if minimumCoverage is None:
      from math import ceil
      if self._selectionIndex is not None:
        self.minimumCoverage = ceil(0.1 * len(self._selectionIndex))
      else:
        self.minimumCoverage = ceil(0.1 * self._table.getNrRows())

  def _createSearchParametersObject(self):
    """Internal: Create a Java SearchParameters object from current settings."""
    from nl.liacs.subdisc import SearchParameters
    from nl.liacs.subdisc import QM, SearchStrategy, NumericOperatorSetting, NumericStrategy

    qualityMeasure = self.qualityMeasure
    if not isinstance(qualityMeasure, QM):
      if hasattr(QM, qualityMeasure.upper()):
        qualityMeasure = getattr(QM, qualityMeasure.upper())
      else:
        raise ValueError("Invalid qualityMeasure")
    searchStrategy = self.searchStrategy
    if not isinstance(searchStrategy, SearchStrategy):
      if hasattr(SearchStrategy, searchStrategy.upper()):
        searchStrategy = getattr(SearchStrategy, searchStrategy.upper())
      else:
        raise ValueError("Invalid searchStrategy")
    numericOperatorSetting = self.numericOperatorSetting
    if not isinstance(numericOperatorSetting, NumericOperatorSetting):
      if hasattr(NumericOperatorSetting, numericOperatorSetting.upper()):
        numericOperatorSetting = getattr(NumericOperatorSetting, numericOperatorSetting.upper())
      else:
        raise ValueError("Invalid numericOperatorSetting")
    numericStrategy = self.numericStrategy
    if not isinstance(numericStrategy, NumericStrategy):
      if hasattr(NumericStrategy, numericStrategy.upper()):
        numericStrategy = getattr(NumericStrategy, numericStrategy.upper())
      else:
        raise ValueError("Invalid numericStrategy")


    sp = SearchParameters()
    sp.setTargetConcept(self._targetConcept)
    sp.setQualityMeasure(qualityMeasure)
    sp.setQualityMeasureMinimum(self.qualityMeasureMinimum)
    sp.setSearchDepth(self.searchDepth)
    sp.setMinimumCoverage(self.minimumCoverage)
    sp.setMaximumCoverageFraction(self.maximumCoverageFraction)
    sp.setMaximumSubgroups(self.maximumSubgroups)
    sp.setSearchStrategy(searchStrategy)
    sp.setNominalSets(self.nominalSets)
    sp.setNumericOperators(numericOperatorSetting)
    sp.setNumericStrategy(numericStrategy)
    sp.setSearchStrategyWidth(self.searchStrategyWidth)
    sp.setFilterSubgroups(self.filterSubgroups)
    sp.setNrBins(self.nrBins)
    sp.setNrThreads(self.nrThreads)

    return sp

  def describeSearchParameters(self):
    """Describe the current search parameters. Returns a string."""
    sp = self._createSearchParametersObject()
    return str(self._targetConcept) + str(sp)

  def computeThreshold(self, *, significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100, setAsMinimum=False, verbose=False):
    """
    Compute a statistical threshold for subgroup quality using randomization.
    Optionally set as the minimum quality threshold for discovery.
    """
    sp = self._createSearchParametersObject()

    threshold = redirectSystemOutErr(computeThreshold, sp, self._targetConcept, self._table, self._selection, significanceLevel=significanceLevel, method=method, amount=amount, verbose=verbose)

    if setAsMinimum:
      self.qualityMeasureMinimum = threshold

    return threshold

  def getBaseModel(self, verbose=True):
    """Return the base model (e.g., regression/correlation) for the target."""
    # TODO: For regression/correlation types, consider also returning the
    # base regression model or correlation here (in some form).
    return self.getModel(None, includeBase=True, verbose=verbose)

  def _ensurePostRun(self):
    """Raise an error if run() has not been called yet."""
    if not self._runCalled:
      raise RuntimeError("This function is only available after a successful call of run()")

  def _checkColumnTypes(self):
    """Internal: Check that the target columns have the correct types for the selected target type."""
    from nl.liacs.subdisc import TargetType, AttributeType
    if self._targetConcept.getTargetType() == TargetType.SINGLE_NOMINAL:
      primaryTarget = self._targetConcept.getPrimaryTarget()
      if primaryTarget.getType() != AttributeType.NOMINAL:
        raise TypeError("targetColumn is not nominal")
    elif self._targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
      primaryTarget = self._targetConcept.getPrimaryTarget()
      if primaryTarget.getType() != AttributeType.NUMERIC:
        raise TypeError("targetColumn is not numeric")
    elif self._targetConcept.getTargetType() in \
         ( TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION ):
      primaryTarget = self._targetConcept.getPrimaryTarget()
      secondaryTarget = self._targetConcept.getSecondaryTarget()
      if primaryTarget.getType() != AttributeType.NUMERIC:
        raise TypeError("primaryTargetColumn is not numeric")
      if secondaryTarget.getType() != AttributeType.NUMERIC:
        raise TypeError("secondaryTargetColumn is not numeric")
    elif self._targetConcept.getTargetType() == TargetType.DOUBLE_BINARY:
      primaryTarget = self._targetConcept.getPrimaryTarget()
      secondaryTarget = self._targetConcept.getSecondaryTarget()
      if primaryTarget.getType() != AttributeType.BINARY:
        raise TypeError("primaryTargetColumn is not binary")
      if secondaryTarget.getType() != AttributeType.BINARY:
        raise TypeError("secondaryTargetColumn is not binary")
    elif self._targetConcept.getTargetType() == TargetType.MULTI_NUMERIC:
      for c in self._targetConcept.getMultiTargets():
        if c.getType() != AttributeType.NUMERIC:
          raise TypeError(f"Target column '{c.getName()}' is not numeric")
    else:
      # Don't block not yet implemented target types here
      pass

  def run(self, verbose=True):
    """
    Run the subgroup discovery process using the configured parameters.
    Results are stored internally and can be accessed via asDataFrame(), getModel(), etc.
    """
    self._checkColumnTypes()
    sp = self._createSearchParametersObject()
    # TODO: check functionality of nrThreads via sp.setNrThreads vs as argument to runSubgroupDiscovery
    from nl.liacs.subdisc import Process
    sd = redirectSystemOutErr(Process.runSubgroupDiscovery, self._table, 0, self._selection, sp, False, self.nrThreads, None, verbose=verbose)
    self._runCalled = True
    self._sd = sd

  def asDataFrame(self):
    """
    Return the discovered subgroups as a pandas DataFrame.
    """
    self._ensurePostRun()
    return generateResultDataFrame(self._sd, self._targetConcept.getTargetType())

  def getSubgroupMembers(self, index):
    """
    Return the members of a discovered subgroup as a boolean pandas Series.
    """
    self._ensurePostRun()
    import pandas
    subgroups = list(self._sd.getResult())
    members = subgroups[index].getMembers()
    return pandas.Series(map(members.get, range(self._index.size)), index=self._index)

  def getModel(self, index, includeBase=True, verbose=True, **kwargs):
    """
    Return the model (e.g., regression, correlation) for the given subgroup(s).
    """
    if index is None:
      # small hack to allow being called by getBaseModel() pre-run
      index = []
      sd = None
    else:
      self._ensurePostRun()
      sd = self._sd

    from nl.liacs.subdisc import TargetType
    if self._targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
      return redirectSystemOutErr(getModelSingleNumeric, self._targetConcept, sd, self._selection, index, includeBase=includeBase, verbose=verbose, **kwargs)
    if self._targetConcept.getTargetType() in \
         ( TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION ):
      return redirectSystemOutErr(getModelDoubleNumeric, self._targetConcept, sd, self._selection, index, dfIndex=self._index, selectionIndex=self._selectionIndex, includeBase=includeBase, verbose=verbose, **kwargs)
    else:
      raise NotImplementedError("getModel() is not implemented for this target type")

  def getPatternTeam(self, size, returnGrouping=False, verbose=True):
    """
    Return a team of patterns (subgroups) of the given size.
    Optionally return the grouping matrix as well.
    """
    self._ensurePostRun()
    size = int(size)
    return redirectSystemOutErr(getPatternTeam, self._sd.getResult(), self._table, self.asDataFrame(), size, returnGrouping=returnGrouping, verbose=verbose)

def generateResultDataFrame(sd, targetType):
  """
  Convert the Java result set to a pandas DataFrame for easy analysis.
  """
  import pandas as pd

  L = [ [ r.getDepth(), r.getCoverage(), r.getMeasureValue(), r.getSecondaryStatistic(), r.getTertiaryStatistic(), r.getPValue(), str(r) ] for r in sd.getResult() ]

  from nl.liacs.subdisc.gui import ResultTableModel
  rtm = ResultTableModel(sd.getResult(), targetType)

  secondaryName = str(rtm.getColumnName(4))
  tertiaryName = str(rtm.getColumnName(5))

  df = pd.DataFrame(L, columns=['Depth', 'Coverage', 'Quality', secondaryName, tertiaryName, 'p-Value', 'Conditions'], copy=True)
  return df



def computeThreshold(sp, targetConcept, table, selection, *, significanceLevel, method, amount, setAsMinimum=False):
  """
  Compute a statistical threshold for subgroup quality using randomization.
  Used internally by SubgroupDiscovery.computeThreshold().
  """
  from nl.liacs.subdisc import TargetType, QualityMeasure, Validation, NormalDistribution
  from nl.liacs.subdisc.gui import RandomQualitiesWindow
  import scipy.stats

  methods = [ 'RANDOM_DESCRIPTIONS', 'RANDOM_SUBSETS', 'SWAP_RANDOMIZATION' ]
  if method.upper() not in methods:
    raise ValueError("Invalid method. Options are: " + ", ".join(methods))
  method = getattr(RandomQualitiesWindow, method).toString()

  # Logic duplicated from java MiningWindow
  if targetConcept.getTargetType() == TargetType.SINGLE_NOMINAL:
    positiveCount = targetConcept.getPrimaryTarget().countValues(targetConcept.getTargetValue(), None)
    qualityMeasure = QualityMeasure(sp.getQualityMeasure(), table.getNrRows(), positiveCount)
  elif targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC:
    from nl.liacs.subdisc import QM, Stat, ProbabilityDensityFunction2
    from java.util import BitSet
    target = targetConcept.getPrimaryTarget()
    qm = sp.getQualityMeasure()
    b = BitSet(table.getNrRows())
    b.set(0, table.getNrRows())

    # TODO (from SubDisc): "check for subset selection"
    statistics = target.getStatistics(None, b, qm == QM.MMAD, QM.requiredStats(qm).contains(Stat.COMPL))

    pdf = ProbabilityDensityFunction2(target, selection)
    pdf.smooth()
    qualityMeasure = QualityMeasure(qm, table.getNrRows(),
                                    statistics.getSubgroupSum(),
                                    statistics.getSubgroupSumSquaredDeviations(),
                                    pdf)
  elif targetConcept.getTargetType() == TargetType.DOUBLE_REGRESSION:
    qualityMeasure = None
  elif targetConcept.getTargetType() == TargetType.DOUBLE_BINARY:
    qualityMeasure = None
  elif targetConcept.getTargetType() == TargetType.DOUBLE_CORRELATION:
    qualityMeasure = None
  elif targetConcept.getTargetType() == TargetType.MULTI_NUMERIC:
    raise NotImplementedError()
  else:
    raise NotImplementedError()

  validation = Validation(sp, table, selection, qualityMeasure)
  qualities = validation.getQualities([ method, str(amount) ])
  if qualities is None:
    # TODO: Check how to handle this
    raise RuntimeError()

  distro = NormalDistribution(qualities)

  threshold = distro.getMu() + scipy.stats.norm.ppf(1 - significanceLevel) * distro.getSigma()

  return threshold

def getModelSingleNumeric(targetConcept, sd, selection, index, relative=True, includeBase=True):
  """
  Return a DataFrame with the probability density function for single numeric targets.
  """
  from nl.liacs.subdisc import TargetType, ProbabilityDensityFunction2
  from pandas import DataFrame
  import numpy as np

  assert targetConcept.getTargetType() == TargetType.SINGLE_NUMERIC
  if not hasattr(index, '__iter__'):
    index = [ index ]

  pdfBase = ProbabilityDensityFunction2(targetConcept.getPrimaryTarget(), selection)
  pdfBase.smooth()
  if includeBase:
    L = [ pdfBase ]
    columns = [ 'base' ]
    scales = [ 1. ]
  else:
    L = []
    columns = []
    scales = []

  subgroups = None
  for i in index:
    if subgroups is None:
      # small hack to avoid calling getResult() if index is empty
      subgroups = list(sd.getResult())
    s = subgroups[i]
    pdfSub = ProbabilityDensityFunction2(pdfBase, s.getMembers())
    pdfSub.smooth()
    assert pdfSub.size() == pdfBase.size()
    L.append(pdfSub)
    columns.append(i)
    if relative:
      scales.append( pdfSub.getAbsoluteCount() / pdfBase.getAbsoluteCount() )
    else:
      scales.append(1.)

  rows = np.zeros((pdfBase.size(), ), dtype=float)
  for i in range(pdfBase.size()):
    rows[i] = pdfBase.getMiddle(i)

  data = np.zeros((pdfBase.size(), len(L)), dtype=float)
  for j, pdf in enumerate(L):
    for i in range(pdfBase.size()):
      data[i, j] = pdf.getDensity(i) * scales[j]

  df = DataFrame(data=data, index=rows, columns=columns, copy=True)

  return df

def getModelDoubleNumeric(targetConcept, sd, selection, index, dfIndex=None, selectionIndex=None, includeBase=True):
  """
  Return a DataFrame with regression/correlation models for double numeric targets.
  """
  # TODO: should this support selection?
  from nl.liacs.subdisc import TargetType, QM, RegressionMeasure
  from pandas import DataFrame
  import numpy as np

  assert targetConcept.getTargetType() in \
         ( TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION )
  if not hasattr(index, '__iter__'):
    index = [ index ]
  
  regression = (targetConcept.getTargetType() == TargetType.DOUBLE_REGRESSION)

  # Create a dataframe with one row per sample.
  # Columns: 'x', the primary target column
  #          'y base', the secondary target column
  #          'pred base', the predicted value with the base regression model
  # and for each requested subgroup nr #:
  #          'y #', NaN if sample is not in subgroup, otherwise y value
  #          'pred #', the predicted value with the subgroup's regression model


  if includeBase:
    if regression:
      columns = [ 'x', 'y base', 'pred base' ]
    else:
      columns = [ 'x', 'y base' ]
  else:
    columns = [ 'x' ]
  baseCols = len(columns)

  L = []
  subgroups = None
  for i in index:
    if subgroups is None:
      # small hack to avoid calling getResult() if index is empty
      subgroups = list(sd.getResult())
    s = subgroups[i]
    L.append(s)
    columns.append(f'y {i}')
    if regression:
      columns.append(f'pred {i}')

  xcoords = np.array(targetConcept.getPrimaryTarget().getFloats())
  nrRows = xcoords.shape[0]

  if regression:
    # REGRESSION_SSD_COMPLEMENT is the default here, but doesn't really matter
    # TODO: Should this take selection into account?
    RM = RegressionMeasure(QM.REGRESSION_SSD_COMPLEMENT, targetConcept.getPrimaryTarget(), targetConcept.getSecondaryTarget())
    slope = RM.getSlope()
    intercept = RM.getIntercept()

  if dfIndex is not None:
    rows = range(nrRows)
  else:
    rows = dfIndex

  data = np.zeros((nrRows, len(columns)), dtype=float)

  data[:, 0] = xcoords
  if includeBase:
    data[:, 1] = targetConcept.getSecondaryTarget().getFloats()
    if regression:
      data[:, 2] = intercept + slope * data[:, 0]

  f = 2 if regression else 1

  for j, s in enumerate(L):
    members = s.getMembers()
    if regression:
      slope = s.getSecondaryStatistic()
      intercept = s.getTertiaryStatistic()
    data[:, f*j+baseCols] = targetConcept.getSecondaryTarget().getFloats()
    if regression:
      data[:, f*j+baseCols+1] = intercept + slope * data[:, 0]
    for i in range(data.shape[0]):
      if not members.get(i):
        data[i, f*j+baseCols] = np.nan

  df = DataFrame(data=data, index=rows, columns=columns, copy=True)

  return df

def getPatternTeam(result, table, df, size, returnGrouping):
  """
  Return a DataFrame of the selected pattern team, and optionally the grouping matrix.
  """
  import pandas as pd
  import numpy as np

  pt = result.getPatternTeam(table, size)
  if returnGrouping:
    grouping = result.getGrouping(pt)

  pt = list(pt)

  count = df.shape[0]
  assert df.index.equals(pd.RangeIndex(count))
  assert len(pt) == size

  L = [ False ] * count

  for p in pt:
    # IDs are 1-based
    L[p.getID() - 1] = True

  pt_df = df[L]

  if not returnGrouping:
    return pt_df

  grouping = list(grouping)
  grouping = [ list(x) for x in grouping ]

  g = np.zeros((count, size), dtype=bool)
  for i,x in enumerate(grouping):
    for s in x:
      g[s.getID() - 1, i] = True

  g_df = pd.DataFrame(data=g, index=df.index, columns=range(size))

  return pt_df, g_df
