# Tests for the implemented target types

import pytest
import pysubdisc
import pandas
import io

@pytest.fixture
def adult_data():
  return pandas.read_csv('tests/adult.txt')

SINGLE_NOMINAL_PARAMS = \
"""target_concept
\tnr_target_attributes = 1
\ttarget_type = single nominal
\tprimary_target = target
\ttarget_value = gr50K
search_parameters
\tquality_measure = Cortana Quality
\tquality_measure_minimum = 0.1
\tsearch_depth = 1
\tminimum_coverage = 100
\tmaximum_coverage_fraction = 0.9
\tminimum_support = 0
\tmaximum_subgroups = 1000
\tminimum_improvement = 0.0
\tfilter_subgroups = true
\tmaximum_time = 0.0
\tsearch_strategy = beam
\tuse_nominal_sets = false
\tsearch_strategy_width = 100
\tnumeric_operators = [<=, >=]
\tnumeric_strategy = bins
\tnr_bins = 8
\tnr_threads = 1
\talpha = 0.5
\tbeta = 1.0
\tpost_processing_do_autorun = true
\tpost_processing_count = 20
\tbeam_seed = []
\toverall_ranking_loss = 0.0
"""

SINGLE_NOMINAL_OUTPUT = \
"""Depth,Coverage,Quality,Target Share,Positives,p-Value,Conditions
1,443,0.517601,0.440181,195.0,NaN,marital-status = 'Married-civ-spouse'
1,376,0.453305,0.446809,168.0,NaN,relationship = 'Husband'
1,327,0.359959,0.428135,140.0,NaN,education-num >= 11.0
1,616,0.354077,0.334416,206.0,NaN,age >= 33.0
1,728,0.326105,0.311813,227.0,NaN,age >= 29.0
1,552,0.263425,0.317029,175.0,NaN,education-num >= 10.0
1,268,0.234734,0.388060,104.0,NaN,hours-per-week >= 43.0
1,282,0.233342,0.379433,107.0,NaN,hours-per-week >= 42.0
1,283,0.232040,0.378092,107.0,NaN,hours-per-week >= 41.0
1,124,0.220187,0.548387,68.0,NaN,occupation = 'Exec-managerial'
1,671,0.198276,0.284650,191.0,NaN,sex = 'Male'
1,851,0.194010,0.272620,232.0,NaN,age >= 24.0
1,166,0.193561,0.439759,73.0,NaN,education = 'Bachelors'
1,863,0.150323,0.263036,227.0,NaN,hours-per-week >= 26.0
1,124,0.124776,0.411290,51.0,NaN,occupation = 'Prof-specialty'
1,873,0.120465,0.256586,224.0,NaN,education-num >= 9.0"""

def test_single_nominal(adult_data):
  sd = pysubdisc.singleNominalTarget(adult_data, 'target', 'gr50K')
  sd.nrThreads = 1
  paramdesc = sd.describeSearchParameters()
  assert paramdesc == SINGLE_NOMINAL_PARAMS
  sd.run(verbose=False)
  df = sd.asDataFrame()
  ref = pandas.read_csv(io.StringIO(SINGLE_NOMINAL_OUTPUT))
  pandas.testing.assert_frame_equal(df, ref, atol=1e-06)




SINGLE_NUMERIC_PARAMS = \
"""target_concept
\tnr_target_attributes = 1
\ttarget_type = single numeric
\tprimary_target = age
\ttarget_value = null
search_parameters
\tquality_measure = Z-Score
\tquality_measure_minimum = 1.0
\tsearch_depth = 1
\tminimum_coverage = 100
\tmaximum_coverage_fraction = 0.9
\tminimum_support = 0
\tmaximum_subgroups = 1000
\tminimum_improvement = 0.0
\tfilter_subgroups = true
\tmaximum_time = 0.0
\tsearch_strategy = beam
\tuse_nominal_sets = false
\tsearch_strategy_width = 100
\tnumeric_operators = [<=, >=]
\tnumeric_strategy = bins
\tnr_bins = 8
\tnr_threads = 1
\talpha = 0.5
\tbeta = 1.0
\tpost_processing_do_autorun = true
\tpost_processing_count = 20
\tbeam_seed = []
\toverall_ranking_loss = 0.0
"""

SINGLE_NUMERIC_OUTPUT = \
"""Depth,Coverage,Quality,Average,St. Dev.,p-Value,Conditions
1,376,7.903417,43.489361,11.650006,NaN,relationship = 'Husband'
1,443,7.205008,42.618511,11.669712,NaN,marital-status = 'Married-civ-spouse'
1,232,6.323667,43.590519,9.771879,NaN,target = 'gr50K'
1,136,3.663591,42.242645,9.817217,NaN,marital-status = 'Divorced'
1,124,3.046701,41.701614,11.187231,NaN,occupation = 'Exec-managerial'
1,500,2.585858,39.594002,13.755857,NaN,fnlwgt <= 180572.0
1,109,2.199817,40.862385,12.213733,NaN,relationship = 'Unmarried'
1,625,2.122866,39.183998,13.497620,NaN,fnlwgt <= 201080.0
1,184,2.036660,40.054348,10.074436,NaN,hours-per-week >= 50.0
1,261,1.993368,39.697319,10.564289,NaN,hours-per-week >= 45.0
1,124,1.875606,40.298386,11.227855,NaN,occupation = 'Prof-specialty'
1,448,1.664761,39.100445,15.002796,NaN,education-num <= 9.0
1,671,1.631180,38.891209,13.414467,NaN,sex = 'Male'
1,244,1.547618,39.372952,10.576988,NaN,education-num >= 13.0
1,321,1.483455,39.155762,14.159153,NaN,education = 'HS-grad'
1,755,1.367879,38.715233,11.864048,NaN,hours-per-week >= 40.0
1,750,1.228084,38.649334,13.452622,NaN,fnlwgt <= 247019.0
1,279,1.161091,38.978493,10.430629,NaN,education-num >= 12.0"""


def test_single_numeric(adult_data):
  sd = pysubdisc.singleNumericTarget(adult_data, 'age')
  sd.nrThreads = 1
  paramdesc = sd.describeSearchParameters()
  assert paramdesc == SINGLE_NUMERIC_PARAMS
  sd.run(verbose=False)
  df = sd.asDataFrame()
  ref = pandas.read_csv(io.StringIO(SINGLE_NUMERIC_OUTPUT))
  pandas.testing.assert_frame_equal(df, ref, atol=1e-06)

DOUBLE_REGRESSION_PARAMS = \
"""target_concept
\tnr_target_attributes = 1
\ttarget_type = double regression
\tprimary_target = age
\ttarget_value = null
\tsecondary_target = hours-per-week
search_parameters
\tquality_measure = Sign. of Slope Diff. (complement)
\tquality_measure_minimum = 3.0
\tsearch_depth = 1
\tminimum_coverage = 100
\tmaximum_coverage_fraction = 0.9
\tminimum_support = 0
\tmaximum_subgroups = 1000
\tminimum_improvement = 0.0
\tfilter_subgroups = true
\tmaximum_time = 0.0
\tsearch_strategy = beam
\tuse_nominal_sets = false
\tsearch_strategy_width = 100
\tnumeric_operators = [<=, >=]
\tnumeric_strategy = bins
\tnr_bins = 8
\tnr_threads = 1
\talpha = 0.5
\tbeta = 1.0
\tpost_processing_do_autorun = true
\tpost_processing_count = 20
\tbeam_seed = []
\toverall_ranking_loss = 0.0
"""

DOUBLE_REGRESSION_OUTPUT = \
"""Depth,Coverage,Quality,Slope,Intercept,p-Value,Conditions
1,344,7.719850,0.345898,26.339434,NaN,marital-status = 'Never-married'
1,151,6.339853,0.731753,14.308075,NaN,relationship = 'Own-child'
1,376,4.589319,-0.200944,52.310714,NaN,relationship = 'Husband'
1,443,4.279790,-0.156311,49.533067,NaN,marital-status = 'Married-civ-spouse'
1,279,3.988836,-0.105054,44.648198,NaN,relationship = 'Not-in-family'
1,126,3.684870,-0.229244,50.648992,NaN,occupation = 'Craft-repair'
1,136,3.643090,-0.272924,53.771664,NaN,marital-status = 'Divorced'
1,500,3.198273,0.172905,33.307597,NaN,fnlwgt >= 180609.0
1,500,3.196495,-0.010646,40.553510,NaN,fnlwgt <= 180572.0"""

def test_double_regression(adult_data):
  sd = pysubdisc.doubleRegressionTarget(adult_data, 'age', 'hours-per-week')
  sd.qualityMeasureMinimum = 3.0
  sd.nrThreads = 1
  paramdesc = sd.describeSearchParameters()
  assert paramdesc == DOUBLE_REGRESSION_PARAMS
  sd.run(verbose=False)
  df = sd.asDataFrame()
  ref = pandas.read_csv(io.StringIO(DOUBLE_REGRESSION_OUTPUT))
  pandas.testing.assert_frame_equal(df, ref, atol=1e-06)
