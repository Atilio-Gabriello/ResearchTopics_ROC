# Quick Start: Multiple Parameter Testing

## 🚀 Ready-to-Use Commands

### Example 1: Test Multiple N-Points on Adult Dataset
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 5 10 15 20 --datasets adult
```
**What it does**: 
- Runs depth 2 search on adult dataset
- Tests methods 3 & 4 with n=5, 10, 15, 20 points
- Creates 10 method variations: basic, remove_hull, 4×closest, 4×furthest, below_hull, above_diagonal

**Output**: `./runs/all_methods_comparison/adult/`

---

### Example 2: Test Multiple Percentages on Ionosphere Dataset
```bash
python run_all_methods_all_datasets.py --depth 2 --distance-percentage 0.5 1.0 2.0 5.0 --datasets ionosphere
```
**What it does**:
- Runs depth 2 search on ionosphere dataset
- Tests methods 5 & 6 with 0.5%, 1.0%, 2.0%, 5.0% thresholds
- Creates 10 method variations: basic, remove_hull, closest, furthest, 4×below_hull, 4×above_diagonal

**Output**: `./runs/all_methods_comparison/ionosphere/`

---

### Example 3: Full Parameter Sweep on Multiple Datasets
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 10 20 30 --distance-percentage 1.0 2.0 5.0 --datasets adult mushroom ionosphere
```
**What it does**:
- Runs on 3 datasets: adult, mushroom, ionosphere
- Tests n-points: 10, 20, 30
- Tests percentages: 1.0%, 2.0%, 5.0%
- Creates 14 method variations per dataset: basic, remove_hull, 3×closest, 3×furthest, 3×below_hull, 3×above_diagonal

**Output**: `./runs/all_methods_comparison/{adult,mushroom,ionosphere}/`

---

### Example 4: Comprehensive Analysis (All Datasets)
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 5 10 15 20 25 --distance-percentage 0.5 1.0 2.0 5.0 10.0
```
**What it does**:
- Runs on ALL datasets in tests folder
- Tests n-points: 5, 10, 15, 20, 25 (5 variations)
- Tests percentages: 0.5%, 1.0%, 2.0%, 5.0%, 10.0% (5 variations)
- Creates 22 method variations per dataset

**Output**: `./runs/all_methods_comparison/` with subdirectories for each dataset

---

### Example 5: Quick Test (Single Parameter Each)
```bash
python run_all_methods_all_datasets.py --depth 1 --datasets adult
```
**What it does**:
- Minimal test on adult dataset at depth 1
- Uses default parameters: n-points=10, percentage=1.0
- Creates 6 basic method variations
- Fast execution for testing

**Output**: `./runs/all_methods_comparison/adult/`

---

### Example 6: Deep Search with Parameter Sweep
```bash
python run_all_methods_all_datasets.py --depth 3 --min-coverage 100 --n-points 10 20 30 40 --distance-percentage 1.0 3.0 5.0 --datasets adult ionosphere wisconsin
```
**What it does**:
- Deeper search (depth 3) on 3 datasets
- Higher coverage requirement (100)
- Tests 4 n-points values and 3 percentage values
- More thorough but slower

**Output**: `./runs/all_methods_comparison/{adult,ionosphere,wisconsin}/`

---

## 📊 Understanding the Output

After running, you'll get:

1. **Consolidated CSV**: `consolidated_results.csv`
   - All results in one table
   - Easy to import into Excel/Python for analysis
   - Includes parameter column showing which value was used

2. **Visualizations**: One PNG per dataset
   - Side-by-side comparison of all method variations
   - Shows ROC curves and hull shapes
   - AUC and point counts labeled

3. **JSON Files**: Detailed results per dataset
   - Full numerical results
   - Can be loaded programmatically

4. **Console Output**: Real-time progress
   - Shows what's running
   - Displays key metrics
   - Final summary table

---

## 🔍 Analyzing Results

### Load and Explore Results
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load consolidated results
df = pd.read_csv('./runs/all_methods_comparison/consolidated_results.csv')

# View summary
print(df.groupby('method')['auc'].describe())

# Compare parameters for closest points method
closest_df = df[df['method_key'].str.contains('closest')]
print(closest_df.pivot_table(values='auc', index='dataset', columns='parameter'))

# Plot AUC vs parameter
for method in ['closest_points', 'below_hull']:
    method_df = df[df['method_key'].str.contains(method)]
    plt.figure(figsize=(10, 6))
    for dataset in method_df['dataset'].unique():
        data = method_df[method_df['dataset'] == dataset]
        plt.plot(data['parameter'].astype(float), data['auc'], marker='o', label=dataset)
    plt.xlabel('Parameter Value')
    plt.ylabel('AUC')
    plt.title(f'AUC vs Parameter - {method}')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Find Best Parameters
```python
# Find best n-points for each dataset (closest method)
closest = df[df['method_key'].str.contains('closest_points')]
best_closest = closest.loc[closest.groupby('dataset')['auc'].idxmax()]
print("\nBest n-points per dataset:")
print(best_closest[['dataset', 'parameter', 'auc', 'num_hull_points']])

# Find best percentage for each dataset (below hull method)
below_hull = df[df['method_key'].str.contains('below_hull')]
best_below = below_hull.loc[below_hull.groupby('dataset')['auc'].idxmax()]
print("\nBest percentage per dataset:")
print(best_below[['dataset', 'parameter', 'auc', 'num_hull_points']])
```

---

## ⚡ Performance Tips

1. **Start Small**: Test with 1-2 parameter values first
   ```bash
   python run_all_methods_all_datasets.py --depth 1 --n-points 10 --datasets adult
   ```

2. **Use Specific Datasets**: Don't run all datasets during exploration
   ```bash
   --datasets adult ionosphere  # Just 2 datasets
   ```

3. **Limit Parameters**: Start with 2-3 values per parameter
   ```bash
   --n-points 10 20  # Just 2 values
   --distance-percentage 1.0 2.0  # Just 2 values
   ```

4. **Increase Gradually**: Once you understand the patterns, expand
   ```bash
   --n-points 5 10 15 20 25 30  # Full range
   ```

---

## 🎯 Common Scenarios

### Scenario 1: Find Optimal Compression
**Goal**: Reduce points while maintaining AUC

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 5 10 15 20 25 30 \
    --datasets adult \
    --output ./runs/compression_test
```
Then analyze where AUC starts dropping significantly.

### Scenario 2: Compare Selection Strategies
**Goal**: Which selection method works best?

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 15 \
    --distance-percentage 2.0 \
    --datasets adult mushroom ionosphere
```
Compare basic, remove_hull, closest, furthest, below_hull, above_diagonal.

### Scenario 3: Dataset-Specific Tuning
**Goal**: Find best parameters for specific dataset

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 5 10 15 20 25 30 40 50 \
    --distance-percentage 0.1 0.5 1.0 2.0 5.0 10.0 \
    --datasets ionosphere \
    --output ./runs/ionosphere_tuning
```
Comprehensive sweep on one dataset.

---

## 📝 Summary

| Parameter | Default | Description | Example Values |
|-----------|---------|-------------|----------------|
| `--depth` | 2 | Search depth | 1, 2, 3 |
| `--n-points` | [10] | Points to select | 5 10 15 20 |
| `--distance-percentage` | [1.0] | % thresholds | 0.5 1.0 2.0 5.0 |
| `--datasets` | all | Which datasets | adult mushroom |
| `--min-coverage` | 50 | Min subgroup size | 50, 100 |
| `--output` | ./runs/all_methods_comparison | Output dir | ./runs/my_test |

**Total Method Variations** = 2 + (n_points_count × 2) + (percentage_count × 2)

Example: 3 n-points + 4 percentages = 2 + 6 + 8 = **16 methods per dataset**

---

Ready to run? Start with Example 5 (Quick Test) and work your way up! 🚀
