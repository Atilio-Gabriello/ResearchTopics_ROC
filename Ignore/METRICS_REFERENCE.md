# Hull Comparison Metrics - Complete Reference

## Overview

The enhanced hull comparison functionality now provides comprehensive ROC search metrics for both the **original hull** and the **new recalculated hull** after removing top-performing subgroups. This allows you to compare search performance with and without the best subgroups.

## Available Metrics

### 1. Subgroup Counts
- `original_num_subgroups`: Number of points on original convex hull
- `new_num_subgroups`: Number of points on new hull (after removal)
- `subgroups_removed`: How many top subgroups were removed
- `subgroups_remaining`: Total points left after removal
- `all_points_num`: Total candidate points

### 2. AUC (Area Under Curve)
- `original_auc`: AUC of original hull
- `new_auc`: AUC of new hull
- `all_points_auc`: AUC using all candidate points
- `auc_reduction`: Absolute AUC loss (original - new)
- `auc_reduction_percentage`: Percentage AUC loss

### 3. Quality Metrics (TPR - FPR)
- `original_max_quality`: Best quality on original hull
- `original_avg_quality`: Average quality of original hull points
- `new_max_quality`: Best quality on new hull
- `new_avg_quality`: Average quality of new hull points
- `quality_reduction`: Loss in best quality

### 4. Best Subgroup Performance
- `original_best_tpr`: Best TPR on original hull
- `original_best_fpr`: Best FPR on original hull
- `new_best_tpr`: Best TPR on new hull
- `new_best_fpr`: Best FPR on new hull

### 5. Hull Area Metrics
- `original_hull_area`: Geometric area under original hull
- `new_hull_area`: Geometric area under new hull
- `hull_area_reduction`: Absolute area reduction
- `reduction_percentage`: Percentage area reduction

## How to Access Metrics

### Method 1: Python API

```python
from true_roc_search import remove_hull_points_and_recalculate
import numpy as np

# Your ROC points (FPR, TPR)
points = np.array([[0.1, 0.3], [0.2, 0.6], [0.3, 0.8]])

# Get comprehensive metrics
hull_data = remove_hull_points_and_recalculate(points, return_details=True)

# Access metrics
print(f"Original AUC: {hull_data['original_auc']:.4f}")
print(f"New AUC: {hull_data['new_auc']:.4f}")
print(f"AUC Reduction: {hull_data['auc_reduction_percentage']:.1f}%")
print(f"Subgroups removed: {hull_data['subgroups_removed']}")
print(f"Best quality reduction: {hull_data['quality_reduction']:.4f}")
```

### Method 2: From ROC Search Results

```python
from true_roc_search import true_roc_search, load_data

# Run search
data = load_data('./tests/adult.txt')
results = true_roc_search(data, 'target', alphas=[0.5], max_depth=3)

# Access hull comparison metrics for each depth
for hull_data in results[0.5]['hull_comparisons']:
    depth = hull_data['depth']
    print(f"\nDepth {depth}:")
    print(f"  AUC: {hull_data['original_auc']:.4f} → {hull_data['new_auc']:.4f}")
    print(f"  Subgroups: {hull_data['original_num_subgroups']} → {hull_data['new_num_subgroups']}")
    print(f"  Best Quality: {hull_data['original_max_quality']:.4f} → {hull_data['new_max_quality']:.4f}")
```

### Method 3: From CSV Files

After running `true_roc_search()`, metrics are saved in CSV format:

```python
import pandas as pd

# Load comprehensive metrics
df = pd.read_csv('./runs/true_roc/alpha_0.5/hull_comparisons/hull_comparison_summary.csv')

# View metrics by depth
print(df[['depth', 'original_auc', 'new_auc', 'auc_reduction_percentage']])
```

### Method 4: Generate Comparison Table

Use the dedicated script:

```bash
python create_metrics_comparison_table.py
```

This creates:
- `comprehensive_metrics_comparison.csv` - All metrics in tabular format
- `comprehensive_metrics_visualization.png` - 6-panel comparison plot

## CSV Output Columns

The `hull_comparison_summary.csv` contains these columns:

| Column | Description |
|--------|-------------|
| `depth` | Search depth |
| `total_points` | Total candidate points |
| `original_hull_points` | Points on original hull |
| `new_hull_points` | Points on new hull |
| `subgroups_removed` | Number removed |
| `original_auc` | Original AUC |
| `new_auc` | New AUC |
| `auc_reduction` | AUC loss |
| `auc_reduction_percentage` | AUC loss % |
| `original_max_quality` | Best quality (original) |
| `new_max_quality` | Best quality (new) |
| `quality_reduction` | Quality loss |
| `original_best_tpr` | Best TPR (original) |
| `original_best_fpr` | Best FPR (original) |
| `new_best_tpr` | Best TPR (new) |
| `new_best_fpr` | Best FPR (new) |

## Interpretation Guide

### AUC Reduction

```python
auc_reduction_pct = hull_data['auc_reduction_percentage']

if auc_reduction_pct > 10:
    print("High impact: Top subgroups critical for performance")
elif auc_reduction_pct > 5:
    print("Moderate impact: Balanced contribution")
else:
    print("Low impact: Good diversity across subgroups")
```

### Quality Analysis

```python
quality_reduction = hull_data['quality_reduction']
original_quality = hull_data['original_max_quality']

quality_impact = (quality_reduction / original_quality) * 100

if quality_impact > 20:
    print("Significant drop in best subgroup quality")
elif quality_impact > 10:
    print("Moderate drop in quality")
else:
    print("New hull maintains good quality")
```

### Subgroup Diversity

```python
removed = hull_data['subgroups_removed']
remaining = hull_data['subgroups_remaining']
new_hull = hull_data['new_num_subgroups']

diversity_ratio = new_hull / remaining

if diversity_ratio > 0.1:
    print("High diversity: Many good subgroups beneath top tier")
elif diversity_ratio > 0.05:
    print("Moderate diversity")
else:
    print("Low diversity: Search converging to specific regions")
```

## Example Analysis Workflow

```python
from true_roc_search import true_roc_search, load_data
import pandas as pd

# 1. Run search
data = load_data('./tests/adult.txt')
results = true_roc_search(data, 'target', alphas=[0.5], max_depth=3,
                         output='./runs/my_analysis')

# 2. Extract metrics for analysis
metrics_list = []
for hull_data in results[0.5]['hull_comparisons']:
    metrics_list.append({
        'depth': hull_data['depth'],
        'auc_original': hull_data['original_auc'],
        'auc_new': hull_data['new_auc'],
        'auc_loss_pct': hull_data['auc_reduction_percentage'],
        'subgroups_removed': hull_data['subgroups_removed'],
        'quality_loss': hull_data['quality_reduction']
    })

df = pd.DataFrame(metrics_list)

# 3. Analyze trends
print("\n=== Trend Analysis ===")
print(f"AUC loss increases with depth: {df['auc_loss_pct'].is_monotonic_increasing}")
print(f"Average AUC retention: {100 - df['auc_loss_pct'].mean():.1f}%")
print(f"Total subgroups removed: {df['subgroups_removed'].sum()}")

# 4. Identify critical depths
critical_depth = df.loc[df['auc_loss_pct'].idxmax(), 'depth']
print(f"Most critical depth: {critical_depth} (highest AUC loss)")

# 5. Quality assessment
avg_quality_loss = df['quality_loss'].mean()
if avg_quality_loss > 0.1:
    print("⚠️  Significant quality loss - top subgroups are critical")
else:
    print("✅ Good quality retention - diverse subgroup set")
```

## Visualization Examples

### Compare AUC Across Depths

```python
import matplotlib.pyplot as plt

depths = [hd['depth'] for hd in results[0.5]['hull_comparisons']]
orig_auc = [hd['original_auc'] for hd in results[0.5]['hull_comparisons']]
new_auc = [hd['new_auc'] for hd in results[0.5]['hull_comparisons']]

plt.figure(figsize=(10, 6))
plt.plot(depths, orig_auc, 'ro-', label='Original Hull AUC', linewidth=2)
plt.plot(depths, new_auc, 'bs-', label='New Hull AUC', linewidth=2)
plt.xlabel('Depth')
plt.ylabel('AUC')
plt.title('AUC Comparison: Original vs New Hull')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./auc_comparison.png', dpi=300)
plt.show()
```

### Quality vs Depth

```python
depths = [hd['depth'] for hd in results[0.5]['hull_comparisons']]
quality_orig = [hd['original_max_quality'] for hd in results[0.5]['hull_comparisons']]
quality_new = [hd['new_max_quality'] for hd in results[0.5]['hull_comparisons']]

plt.figure(figsize=(10, 6))
plt.bar(np.array(depths) - 0.2, quality_orig, 0.4, label='Original', color='red', alpha=0.7)
plt.bar(np.array(depths) + 0.2, quality_new, 0.4, label='New', color='purple', alpha=0.7)
plt.xlabel('Depth')
plt.ylabel('Best Quality (TPR - FPR)')
plt.title('Quality Comparison Across Depths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('./quality_comparison.png', dpi=300)
plt.show()
```

## Use Cases

### 1. Algorithm Evaluation
Compare how different alpha values affect hull stability:

```python
results = true_roc_search(data, 'target', alphas=[0.3, 0.5, 0.7])

for alpha in [0.3, 0.5, 0.7]:
    avg_auc_loss = np.mean([
        hd['auc_reduction_percentage'] 
        for hd in results[alpha]['hull_comparisons']
    ])
    print(f"Alpha {alpha}: Avg AUC loss = {avg_auc_loss:.1f}%")
```

### 2. Depth Selection
Find optimal depth before performance degrades:

```python
for hull_data in results[0.5]['hull_comparisons']:
    depth = hull_data['depth']
    auc_retention = 100 - hull_data['auc_reduction_percentage']
    
    if auc_retention < 90:  # More than 10% AUC loss
        print(f"⚠️  Depth {depth}: AUC retention only {auc_retention:.1f}%")
    else:
        print(f"✅ Depth {depth}: Good AUC retention ({auc_retention:.1f}%)")
```

### 3. Subgroup Discovery Assessment
Evaluate search effectiveness:

```python
hull_data = results[0.5]['hull_comparisons'][-1]  # Last depth

if hull_data['new_num_subgroups'] > 3:
    print("✅ Good subgroup diversity discovered")
    print(f"   Even after removing {hull_data['subgroups_removed']} best,")
    print(f"   {hull_data['new_num_subgroups']} strong subgroups remain")
else:
    print("⚠️  Limited diversity - consider:")
    print("   • Increasing search depth")
    print("   • Adjusting min_coverage")
    print("   • Different alpha value")
```

## Summary

The enhanced hull comparison now provides:

✅ **AUC metrics** - Compare original vs new hull performance  
✅ **Subgroup counts** - Track number of candidates and removals  
✅ **Quality metrics** - Measure TPR-FPR quality changes  
✅ **Best subgroup info** - Identify top performer characteristics  
✅ **Comprehensive CSV** - Export all metrics for analysis  
✅ **Rich visualizations** - 6-panel comparison plots  

All metrics are automatically calculated and saved when you run `true_roc_search()`!
