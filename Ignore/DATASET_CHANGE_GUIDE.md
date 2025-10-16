# Quick Reference: Changing Datasets in Test Scripts

## 📍 Where to Change the Dataset

All test scripts use one of two approaches to get data:

---

## Option 1: Synthetic Data (Original Test Scripts)

**Files:**
- `test_point_selection.py`
- `test_vertical_distance_selection.py`
- `test_all_selection_methods.py`

**Current data source (Line ~40-55):**
```python
def generate_test_points(n_points=50, seed=42):
    """Generate test ROC points for demonstration."""
    np.random.seed(seed)
    
    # Generate points above diagonal with varying quality
    fpr = np.random.uniform(0, 0.8, n_points)
    tpr = fpr + np.random.uniform(0.1, 0.4, n_points)
    
    # Clip to valid ROC space
    tpr = np.clip(tpr, 0, 1)
    
    points = np.column_stack([fpr, tpr])
    
    # Filter to only points above diagonal
    points = points[points[:, 1] > points[:, 0]]
    
    return points
```

**To use YOUR data instead:**
```python
def load_real_data():
    """Load real ROC points from your datasets."""
    import pandas as pd
    
    # Choose your dataset
    df = pd.read_csv('runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv')
    
    # Extract FPR and TPR columns
    points = df[['fpr', 'tpr']].values
    
    return points
```

**Then change the main() function (around line 250-300):**
```python
# OLD:
points = generate_test_points(n_points=50, seed=42)

# NEW:
points = load_real_data()
```

---

## Option 2: Real Data (Automated Test Script)

**File:** `test_real_datasets.py`

**Current datasets (Line ~380-387):**
```python
datasets = {
    'adult': 'runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv',
    'ionosphere': 'runs/all_datasets_complete/ionosphere/alpha_pure_roc/roc_points.csv',
    'mushroom': 'runs/all_datasets_complete/mushroom/alpha_pure_roc/roc_points.csv'
}
```

**To add more datasets:**
```python
datasets = {
    'adult': 'runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv',
    'ionosphere': 'runs/all_datasets_complete/ionosphere/alpha_pure_roc/roc_points.csv',
    'mushroom': 'runs/all_datasets_complete/mushroom/alpha_pure_roc/roc_points.csv',
    # Add your datasets here:
    'wisconsin': 'path/to/wisconsin/roc_points.csv',
    'credit': 'path/to/credit/roc_points.csv'
}
```

**To test different N values (Line ~390):**
```python
# OLD:
n_points_list = [5, 10]

# NEW: Test more variations
n_points_list = [5, 10, 15, 20, 25]
```

---

## Quick Copy-Paste Examples

### Example 1: Test One Specific Dataset

**Create `test_my_dataset.py`:**
```python
import pandas as pd
import numpy as np
from true_roc_search import select_points_below_hull

# Load YOUR dataset
df = pd.read_csv('runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv')
points = df[['fpr', 'tpr']].values

print(f"Loaded {len(points)} ROC points from Adult dataset")

# Test with different N values
for n in [5, 10, 15]:
    print(f"\n=== Testing with N={n} ===")
    
    result = select_points_below_hull(points, n, return_details=True, exclude_hull_points=True)
    
    print(f"Original AUC: {result['original_auc']:.4f}")
    print(f"New AUC: {result['new_auc']:.4f}")
    print(f"Change: {result['auc_reduction_percentage']:.2f}%")
    print(f"New hull points: {len(result['new_hull'])}")
```

### Example 2: Compare Two Datasets

**Create `compare_datasets.py`:**
```python
import pandas as pd
from true_roc_search import select_points_below_hull

# Load two datasets
adult_df = pd.read_csv('runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv')
adult_points = adult_df[['fpr', 'tpr']].values

mushroom_df = pd.read_csv('runs/all_datasets_complete/mushroom/alpha_pure_roc/roc_points.csv')
mushroom_points = mushroom_df[['fpr', 'tpr']].values

# Compare with N=10
n = 10

adult_result = select_points_below_hull(adult_points, n, return_details=True)
mushroom_result = select_points_below_hull(mushroom_points, n, return_details=True)

print(f"\n{'Dataset':<15} {'Points':<10} {'Orig AUC':<12} {'New AUC':<12} {'Change %':<10}")
print("-" * 60)
print(f"{'Adult':<15} {len(adult_points):<10} {adult_result['original_auc']:<12.4f} "
      f"{adult_result['new_auc']:<12.4f} {adult_result['auc_reduction_percentage']:<10.2f}")
print(f"{'Mushroom':<15} {len(mushroom_points):<10} {mushroom_result['original_auc']:<12.4f} "
      f"{mushroom_result['new_auc']:<12.4f} {mushroom_result['auc_reduction_percentage']:<10.2f}")
```

### Example 3: Load from Custom Path

**If your ROC points are elsewhere:**
```python
import pandas as pd
from true_roc_search import select_closest_points_to_hull

# Custom path to your CSV
custom_path = 'C:/my_research/experiment_results/my_roc_data.csv'

# Load data (assuming it has fpr and tpr columns)
df = pd.read_csv(custom_path)
points = df[['fpr', 'tpr']].values

# Use any function
result = select_closest_points_to_hull(points, 15, return_details=True)

print(f"Loaded {len(points)} points from custom file")
print(f"AUC reduction: {result['auc_reduction_percentage']:.2f}%")
```

---

## File Paths Reference

**Your current ROC data locations:**

```
runs/all_datasets_complete/
├── adult/
│   └── alpha_pure_roc/
│       └── roc_points.csv          ← 18 points
├── ionosphere/
│   └── alpha_pure_roc/
│       └── roc_points.csv          ← 7 points
└── mushroom/
    └── alpha_pure_roc/
        └── roc_points.csv          ← 16 points
```

**Other possible locations (if they exist):**
```
runs/batch_analysis/{dataset}/alpha_pure_roc/roc_points.csv
runs/test_all_datasets/{dataset}/alpha_pure_roc/roc_points.csv
tests/{dataset}.txt                    ← Original raw data (not ROC points)
```

---

## Common CSV Formats

**Format 1: ROC Points (YOUR current format)**
```csv
fpr,tpr,quality,coverage
0.2318,0.8060,0.5743,365
```
**Load with:**
```python
df = pd.read_csv('path/to/file.csv')
points = df[['fpr', 'tpr']].values
```

**Format 2: With Depth Column**
```csv
depth,fpr,tpr,quality
1,0.23,0.81,0.57
2,0.28,0.83,0.55
```
**Load with:**
```python
df = pd.read_csv('path/to/file.csv')
depth_2 = df[df['depth'] == 2]
points = depth_2[['fpr', 'tpr']].values
```

**Format 3: Different Column Names**
```csv
false_positive_rate,true_positive_rate,score
0.23,0.81,0.57
```
**Load with:**
```python
df = pd.read_csv('path/to/file.csv')
points = df[['false_positive_rate', 'true_positive_rate']].values
```

---

## Troubleshooting

### Error: "KeyError: 'fpr'"
**Problem:** CSV doesn't have 'fpr' column

**Solution:** Check column names
```python
df = pd.read_csv('your_file.csv')
print(df.columns)  # See what columns exist
```

### Error: "FileNotFoundError"
**Problem:** Wrong path to CSV

**Solution:** Use absolute path or check current directory
```python
import os
print(os.getcwd())  # See current directory
```

### Error: "Not enough points"
**Problem:** N > number of available points

**Solution:** Check how many points you have
```python
df = pd.read_csv('your_file.csv')
print(f"Available points: {len(df)}")
# Use smaller N value
```

---

## Summary Table

| What You Want | File to Modify | Line to Change | What to Change |
|---------------|----------------|----------------|----------------|
| Use different dataset | `test_real_datasets.py` | 380-387 | Add/change CSV path in `datasets` dict |
| Change N values | `test_real_datasets.py` | 390 | Modify `n_points_list` |
| Test single dataset quickly | Create new file | - | Copy Example 1 above |
| Compare datasets | Create new file | - | Copy Example 2 above |
| Use synthetic data | Any test script | 40-55 | Use `generate_test_points()` |
| Use real data | Any test script | In main() | Replace with `pd.read_csv()` |

---

## Best Practice

**For experimentation:** Use `test_real_datasets.py` (automated)
**For specific analysis:** Create custom script using examples above
**For quick tests:** Use Python interactive mode:

```python
>>> import pandas as pd
>>> from true_roc_search import select_points_below_hull
>>> df = pd.read_csv('runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv')
>>> points = df[['fpr', 'tpr']].values
>>> result = select_points_below_hull(points, 5, return_details=True)
>>> print(result['auc_reduction_percentage'])
6.09
```

---

**That's it!** You now know exactly where and how to change the dataset for any test script. 🎉
