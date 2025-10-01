# Tests for the implemented target types

import pytest
import pysubdisc
import pandas
import io
import numpy as np

@pytest.fixture
def adult_data():
  return pandas.read_csv('tests/adult.txt')

def test_compute_threshold(adult_data):
  sd = pysubdisc.singleNominalTarget(adult_data, 'target', 'gr50K')
  # We need to use a relatively high amount to get higher precision
  t = sd.computeThreshold(amount=10000, verbose=False)
  assert t == pytest.approx(0.1004, abs=0.001)


def test_getmodel_singlenumeric(adult_data):
  sd = pysubdisc.singleNumericTarget(adult_data, 'age')
  df = sd.getBaseModel(verbose=False)
  assert df['base'].mean() == pytest.approx(0.010283, abs=1e-6)
  assert df['base'].std() == pytest.approx(0.0104838, abs=1e-6)

  sd.run(verbose=False)

  df = sd.getModel(0, verbose=False)
  assert df['base'].mean() == pytest.approx(0.010283, abs=1e-6)
  assert df['base'].std() == pytest.approx(0.0104838, abs=1e-6)
  assert df[0].mean() == pytest.approx(0.003866, abs=1e-6)
  assert df[0].std() == pytest.approx(0.004420, abs=1e-6)

  df = sd.getModel([0, 1], verbose=False)
  assert df['base'].mean() == pytest.approx(0.010283, abs=1e-6)
  assert df['base'].std() == pytest.approx(0.0104838, abs=1e-6)
  assert df[0].mean() == pytest.approx(0.003866, abs=1e-6)
  assert df[0].std() == pytest.approx(0.004420, abs=1e-6)
  assert df[1].mean() == pytest.approx(0.004555, abs=1e-6)
  assert df[1].std() == pytest.approx(0.005234, abs=1e-6)

  df = sd.getModel(0, includeBase=False, verbose=False)
  with pytest.raises(KeyError):
    _ = df['base']
  assert df[0].mean() == pytest.approx(0.003866, abs=1e-6)
  assert df[0].std() == pytest.approx(0.004420, abs=1e-6)


def test_getmodel_doubleregression(adult_data):
  sd = pysubdisc.doubleRegressionTarget(adult_data, 'age', 'hours-per-week')
  df = sd.getBaseModel(verbose=False)
  assert all(df.columns == ['x', 'y base', 'pred base'])
  assert all(df.index == adult_data.index)

  sd.run(verbose=False)

  df = sd.asDataFrame()
  assert df['Slope'][0] == pytest.approx(0.345898, abs=1e-6)
  assert df['Intercept'][0] == pytest.approx(26.339434, abs=1e-6)

  df = sd.getModel(0, verbose=False)
  assert all(df.columns == ['x', 'y base', 'pred base', 'y 0', 'pred 0'])
  assert df['y base'].count() == 1000
  assert df['y 0'].count() == 344
  assert df['pred 0'].count() == 1000
  assert df['pred 0'][0] ==  pytest.approx(39.0 * 0.345898 + 26.339434, abs=1e-4)

  df = sd.getModel(0, includeBase=False, verbose=False)
  assert all(df.columns == ['x', 'y 0', 'pred 0'])
  assert df['y 0'].count() == 344
  assert df['pred 0'].count() == 1000
  assert df['pred 0'][0] ==  pytest.approx(39.0 * 0.345898 + 26.339434, abs=1e-4)

def test_view(adult_data):
  view = adult_data[adult_data['education'] == 'Bachelors']
  assert view.shape == (166, 15)

  sd = pysubdisc.singleNominalTarget(view, 'target', 'gr50K')
  assert sd.minimumCoverage == 17

  sd.run(verbose=False)

  df = sd.asDataFrame()
  assert df['Quality'][0] == pytest.approx(0.539402, abs=1e-6)
  assert df['Coverage'][0] == 84

  members = sd.getSubgroupMembers(0)
  assert members.sum() == 84
  assert all(members.index == view.index)

def test_selection(adult_data):
  view = adult_data[adult_data['education'] == 'Bachelors']
  sd = pysubdisc.singleNominalTarget(view, 'target', 'gr50K')
  sd.run(verbose=False)
  df = sd.asDataFrame()

  table = pysubdisc.loadDataFrame(adult_data)
  table.setSelection(adult_data['education'] == 'Bachelors')
  sd = pysubdisc.singleNominalTarget(table, 'target', 'gr50K')
  assert sd.minimumCoverage == 17

  sd.run(verbose=False)

  df2 = sd.asDataFrame()
  assert df.equals(df2)
