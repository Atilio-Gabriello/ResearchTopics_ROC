# Tests that SubDisc behaviour matches pySubDisc expectations

import pytest
import pysubdisc

def test_JVM():
  pysubdisc.java.ensureJVMStarted()

def test_classes():
  # Test (a non-exhaustive set of) imports
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import Table, Process, TargetType, TargetConcept
  from nl.liacs.subdisc import ProbabilityDensityFunction2, Validation, NormalDistribution, QualityMeasure
  from nl.liacs.subdisc.gui import ResultTableModel, RandomQualitiesWindow

def test_enums():
  # Test defaults for enums
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import TargetType, QM, SearchStrategy, NumericOperatorSetting, NumericStrategy
  from nl.liacs.subdisc.gui import RandomQualitiesWindow
  _ = RandomQualitiesWindow.RANDOM_SUBSETS
  _ = RandomQualitiesWindow.RANDOM_DESCRIPTIONS
  _ = RandomQualitiesWindow.SWAP_RANDOMIZATION
  _ = TargetType.SINGLE_NOMINAL
  _ = TargetType.SINGLE_NUMERIC
  _ = TargetType.DOUBLE_REGRESSION
  _ = TargetType.DOUBLE_BINARY
  _ = TargetType.DOUBLE_CORRELATION
  _ = TargetType.MULTI_NUMERIC
  _ = QM.CORTANA_QUALITY
  _ = QM.Z_SCORE
  _ = QM.REGRESSION_SSD_COMPLEMENT
  _ = QM.RELATIVE_WRACC
  _ = QM.CORRELATION_R
  _ = QM.SQUARED_HELLINGER_2D
  _ = QM.L2
  _ = SearchStrategy.BEAM
  _ = NumericStrategy.NUMERIC_BINS
  _ = NumericOperatorSetting.NORMAL
  
def test_ResultTableModel():
  # Test that ResultTableModel columns 4 and 5 correspond to secondary/tertiary
  # result statistics
  pysubdisc.java.ensureJVMStarted()
  from nl.liacs.subdisc import TargetType
  from nl.liacs.subdisc.gui import ResultTableModel
  assert ResultTableModel.getColumnName(4, TargetType.SINGLE_NOMINAL) == 'Target Share'
  assert ResultTableModel.getColumnName(5, TargetType.SINGLE_NOMINAL) == 'Positives'

def test_pdf2():
  from nl.liacs.subdisc import ProbabilityDensityFunction
  assert ProbabilityDensityFunction.USE_ProbabilityDensityFunction2
