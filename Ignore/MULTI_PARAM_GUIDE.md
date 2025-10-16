# Multiple Parameter Testing Guide

## Overview

The script `run_all_methods_all_datasets.py` now supports testing multiple parameter values for n-points and distance percentages in a single run. This allows comprehensive comparison across different parameter settings.

## What Changed

### Multiple N-Points
Methods 3 and 4 (closest points to hull and furthest from diagonal) now accept multiple `--n-points` values:
- **Before**: Only one n-points value per run
- **Now**: Test multiple values simultaneously

### Multiple Distance Percentages
Methods 5 and 6 (below hull and above diagonal) now accept multiple `--distance-percentage` values:
- **Before**: Only one percentage per run
- **Now**: Test multiple percentages simultaneously

## Usage Examples

### Test Single Parameters (Default Behavior)
```bash
python run_all_methods_all_datasets.py --depth 2
```
This runs with default values: `n-points=[10]`, `distance-percentage=[1.0]`

### Test Multiple N-Points
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 5 10 15 20
```
This will test methods 3 and 4 with n=5, 10, 15, and 20 points.

### Test Multiple Distance Percentages
```bash
python run_all_methods_all_datasets.py --depth 2 --distance-percentage 0.5 1.0 2.0 5.0
```
This will test methods 5 and 6 with 0.5%, 1.0%, 2.0%, and 5.0% thresholds.

### Test Both Multiple Parameters
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 5 10 20 --distance-percentage 0.5 1.0 2.0 5.0
```
This creates a comprehensive comparison with:
- Methods 3 and 4: 3 variations each (n=5, 10, 20)
- Methods 5 and 6: 4 variations each (0.5%, 1.0%, 2.0%, 5.0%)

### Test on Specific Datasets
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 10 20 30 --datasets adult ionosphere
```
Run only on adult and ionosphere datasets with 3 n-points variations.

### Full Example with All Options
```bash
python run_all_methods_all_datasets.py \
    --depth 3 \
    --min-coverage 100 \
    --n-points 5 10 15 20 25 \
    --distance-percentage 0.1 0.5 1.0 2.0 5.0 10.0 \
    --datasets adult mushroom ionosphere \
    --output ./runs/comprehensive_param_sweep
```

## Output Structure

### Results Organization
For each dataset, the script generates:

1. **Method Keys with Parameters**:
   - `basic` - Basic ROC search (no parameters)
   - `remove_hull` - Remove hull points (no parameters)
   - `closest_points_n5`, `closest_points_n10`, `closest_points_n15`, etc.
   - `furthest_points_n5`, `furthest_points_n10`, `furthest_points_n15`, etc.
   - `below_hull_pct0.5`, `below_hull_pct1.0`, `below_hull_pct2.0`, etc.
   - `above_diagonal_pct0.5`, `above_diagonal_pct1.0`, `above_diagonal_pct2.0`, etc.

2. **Consolidated CSV**: `consolidated_results.csv`
   - Includes a `parameter` column showing the parameter value used
   - Sorted by dataset and method for easy comparison

3. **Visualizations**: One comprehensive plot per dataset showing all method variations

4. **JSON Files**: Detailed results for each dataset with all parameter combinations

## CSV Output Columns

The consolidated results CSV includes:
- `dataset`: Dataset name
- `method`: Method description
- `method_key`: Unique key identifying method and parameters
- `parameter`: Parameter value (n-points or percentage)
- `num_subgroups`: Total subgroups generated
- `num_selected`: Number of points selected
- `num_hull_points`: Points on the final hull
- `auc`: Area under ROC curve
- `original_auc`: Original AUC before selection
- `auc_reduction`: Difference from original
- `best_tpr`, `best_fpr`: Best point coordinates
- `max_quality`: Maximum TPR - FPR
- `avg_quality`: Average quality
- `time_seconds`: Execution time

## Analysis Tips

### Finding Optimal Parameters

1. **Compare AUC across parameters**:
   ```bash
   # Run comprehensive sweep
   python run_all_methods_all_datasets.py --depth 2 --n-points 5 10 15 20 25 30
   
   # Open consolidated_results.csv and filter by method to see AUC trends
   ```

2. **Balance quality vs. compression**:
   - Lower n-points = more compression, potentially lower AUC
   - Higher n-points = less compression, closer to original AUC
   - Find the "elbow" point where AUC drops significantly

3. **Compare percentage thresholds**:
   - Lower % = stricter selection, fewer points
   - Higher % = looser selection, more points
   - Identify the threshold that maintains quality

### Visualization Insights

The generated plots show all method variations side-by-side, making it easy to:
- See how ROC curves change with different parameters
- Identify which methods preserve hull shape better
- Compare point distributions across parameter settings

## Performance Considerations

### Number of Runs
Total method variations = 2 (fixed) + n_points_count × 2 + percentages_count × 2

Example:
- `--n-points 5 10 15 20` → 4 variations for methods 3 and 4 = 8 runs
- `--distance-percentage 0.5 1.0 2.0 5.0` → 4 variations for methods 5 and 6 = 8 runs
- Total: 2 + 8 + 8 = **18 method variations per dataset**

### Execution Time
- Each method variation runs independently
- Time scales linearly with number of parameter values
- Use `--datasets` to limit scope during parameter exploration

### Recommended Parameter Ranges

**For n-points** (methods 3 and 4):
- Small datasets: `--n-points 5 10 15 20`
- Large datasets: `--n-points 10 20 30 50`
- Exploration: `--n-points 5 10 15 20 25 30 40 50`

**For distance percentages** (methods 5 and 6):
- Fine-grained: `--distance-percentage 0.1 0.5 1.0 2.0`
- Standard: `--distance-percentage 1.0 2.0 5.0`
- Wide range: `--distance-percentage 0.1 0.5 1.0 2.0 5.0 10.0`

## Example Workflow

### 1. Quick Test
```bash
# Test with 2-3 parameter values on one dataset
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 10 20 \
    --distance-percentage 1.0 2.0 \
    --datasets adult
```

### 2. Comprehensive Sweep
```bash
# Full parameter sweep on all datasets
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 5 10 15 20 25 30 \
    --distance-percentage 0.5 1.0 2.0 5.0 10.0 \
    --output ./runs/full_param_sweep
```

### 3. Analyze Results
```python
import pandas as pd

# Load results
df = pd.read_csv('./runs/full_param_sweep/consolidated_results.csv')

# Compare AUC by parameter for a specific method
method_df = df[df['method'].str.contains('Closest')].copy()
method_df['n_points'] = method_df['parameter'].astype(int)
print(method_df.pivot_table(values='auc', index='dataset', columns='n_points'))

# Find best parameter per dataset
best_params = df.loc[df.groupby(['dataset', 'method'])['auc'].idxmax()]
print(best_params[['dataset', 'method', 'parameter', 'auc']])
```

## Summary

The enhanced script provides flexible parameter testing capabilities:
- ✅ Test multiple n-points values in one run
- ✅ Test multiple percentage thresholds in one run
- ✅ Comprehensive comparison across all parameter combinations
- ✅ Organized output with parameter identification
- ✅ Easy-to-analyze CSV format

This enables systematic parameter optimization and method comparison across datasets.
