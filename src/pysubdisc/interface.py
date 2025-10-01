
# Ensure the Java Virtual Machine is started and import the main SubgroupDiscovery class
from .java import ensureJVMStarted
from .core import SubgroupDiscovery


def loadDataFrame(data):
  """
  Create a SubDisc Table from a pandas DataFrame.
  This function converts a pandas DataFrame into a Java Table object compatible with SubDisc.
  """
  ensureJVMStarted()  # Start JVM if not already running

  from nl.liacs.subdisc import Column
  from nl.liacs.subdisc import AttributeType
  from nl.liacs.subdisc import Table as sdTable
  from jpype import JArray, JString, JBoolean, JFloat
  from java.io import File
  import pandas as pd
  from .core import Table


  dummyfile = File('pandas.DataFrame')  # Dummy file name for Java Table
  nrows, ncols = data.shape
  table = sdTable(dummyfile, nrows, ncols)
  columns = table.getColumns()
  index = pd.RangeIndex(nrows)


  # Convert each DataFrame column to a Java Column and add to the Table
  for i, name in enumerate(data.columns):
    # Determine the attribute type for each column
    if pd.api.types.is_string_dtype(data.dtypes[name]):
      atype = AttributeType.NOMINAL
      ctype = str
      jtype = JString
    elif pd.api.types.is_bool_dtype(data.dtypes[name]):
      atype = AttributeType.BINARY
      ctype = bool
      jtype = JBoolean
    elif pd.api.types.is_numeric_dtype(data.dtypes[name]):
      atype = AttributeType.NUMERIC
      ctype = float
      jtype = JFloat
    else:
      raise ValueError(f"""Unsupported column type '{data.dtypes[name]}' for column '{name}'""")
    column = Column(name, name, atype, i, nrows)
    # Set the column data using the appropriate Java array type
    column.setData(JArray(jtype)@data[name].set_axis(index).astype(ctype))
    columns.add(column)


  table.update()  # Finalize the Java Table

  t = Table(table, data.index)  # Wrap in Python Table class

  return t


def _createTable(data):
  """
  Helper function to ensure input data is converted to a Table object.
  Accepts a Java Table, Python Table, or pandas DataFrame.
  """
  from nl.liacs.subdisc import Table as sdTable
  from .core import Table

  if isinstance(data, sdTable):
    import pandas as pd
    index = pd.RangeIndex(data.getNrRows())
    table = Table(data, index)
  elif isinstance(data, Table):
    table = data
  else:
    table = loadDataFrame(data)

  return table



# Factory function for single nominal target subgroup discovery
def singleNominalTarget(data, targetColumn, targetValue):
  """
  Create subdisc interface of type 'single nominal'.
  Arguments:
    data -- the data as a DataFrame
    targetColumn -- the name/index of the target (nominal) column
    targetValue -- the target value
  Returns a SubgroupDiscovery object configured for single nominal targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.SINGLE_NOMINAL

  # Can use column index or column name
  target = table._table.getColumn(targetColumn)
  if target is None:
    raise ValueError(f"Unknown column '{targetColumn}'")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)
  targetConcept.setTargetValue(targetValue)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = 'CORTANA_QUALITY')
  sd._checkColumnTypes()

  return sd


# Factory function for single numeric target subgroup discovery
def singleNumericTarget(data, targetColumn):
  """
  Create subdisc interface of type 'single numeric'.
  Arguments:
    data -- the data as a DataFrame
    targetColumn -- the name/index of the target (numeric) column
  Returns a SubgroupDiscovery object configured for single numeric targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.SINGLE_NUMERIC

  # Can use column index or column name
  target = table._table.getColumn(targetColumn)
  if target is None:
    raise ValueError(f"Unknown column '{targetColumn}'")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(target)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = 'Z_SCORE')
  sd._checkColumnTypes()

  return sd


# Factory function for double regression target subgroup discovery
def doubleRegressionTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """
  Create subdisc interface of type 'double regression'.
  Arguments:
    data -- the data as a DataFrame
    primaryTargetColumn -- the name/index of the primary target (numeric)
    secondaryTargetColumn -- the name/index of the secondary target (numeric)
  Returns a SubgroupDiscovery object configured for double regression targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_REGRESSION

  # Can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = 'REGRESSION_SSD_COMPLEMENT')
  sd._checkColumnTypes()

  return sd


# Factory function for double binary target subgroup discovery
def doubleBinaryTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """
  Create subdisc interface of type 'double binary'.
  Arguments:
    data -- the data as a DataFrame
    primaryTargetColumn -- the name/index of the primary target (binary)
    secondaryTargetColumn -- the name/index of the secondary target (binary)
  Returns a SubgroupDiscovery object configured for double binary targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_BINARY

  # Can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = 'RELATIVE_WRACC')
  sd._checkColumnTypes()

  return sd


# Factory function for double correlation target subgroup discovery
def doubleCorrelationTarget(data, primaryTargetColumn, secondaryTargetColumn):
  """
  Create subdisc interface of type 'double correlation'.
  Arguments:
    data -- the data as a DataFrame
    primaryTargetColumn -- the name/index of the primary target (numeric)
    secondaryTargetColumn -- the name/index of the secondary target (numeric)
  Returns a SubgroupDiscovery object configured for double correlation targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType

  table = _createTable(data)

  targetType = TargetType.DOUBLE_CORRELATION

  # Can use column index or column name
  primaryTarget = table._table.getColumn(primaryTargetColumn)
  secondaryTarget = table._table.getColumn(secondaryTargetColumn)
  if primaryTarget is None:
    raise ValueError(f"Unknown column '{primaryTargetColumn}'")
  if secondaryTarget is None:
    raise ValueError(f"Unknown column '{secondaryTargetColumn}'")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setPrimaryTarget(primaryTarget)
  targetConcept.setSecondaryTarget(secondaryTarget)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = 'CORRELATION_R')
  sd._checkColumnTypes()

  return sd


# Factory function for multi-numeric target subgroup discovery
def multiNumericTarget(data, targetColumns):
  """
  Create subdisc interface of type 'multi numeric'.
  Arguments:
    data -- the data as a DataFrame
    targetColumns -- list of name/index of the target columns (numeric)
  Returns a SubgroupDiscovery object configured for multi-numeric targets.
  """
  ensureJVMStarted()

  from nl.liacs.subdisc import TargetConcept, TargetType
  from java.util import ArrayList

  table = _createTable(data)

  targetType = TargetType.MULTI_NUMERIC

  L = ArrayList()
  for c in targetColumns:
    # Can use column index or column name
    target = table._table.getColumn(c)
    if target is None:
      raise ValueError(f"Unknown column '{c}'")
    L.add(target)

  if L.size() < 2:
    raise ValueError("At least 2 columns must be selected")

  # Set up the target concept for subgroup discovery
  targetConcept = TargetConcept()
  targetConcept.setTargetType(targetType)
  targetConcept.setMultiTargets(L)

  # Create the SubgroupDiscovery object
  sd = SubgroupDiscovery(targetConcept, table)

  # Choose quality measure based on number of targets
  if L.size() == 2:
    # This qualityMeasure is only available for 2D
    qm = 'SQUARED_HELLINGER_2D'
  else:
    qm = 'L2'

  # Initialize search parameters and check column types
  sd._initSearchParameters(qualityMeasure = qm)
  sd._checkColumnTypes()

  return sd
