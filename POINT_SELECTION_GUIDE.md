# Point Selection Functions Guide

This guide explains the two new ROC curve point selection functions that create alternative ROC curves based on different selection criteria.

## Overview

Both functions select a subset of points from the ROC space and compute a new convex hull, similar to `remove_hull_points_and_recalculate` but with different selection strategies:

1. **`select_closest_points_to_hull`** - Selects points closest to the original convex hull
2. **`select_furthest_points_from_diagonal`** - Selects points furthest from the diagonal (highest quality)

## Functions

### 1. select_closest_points_to_hull()

Selects the n points that are closest to the original ROC convex hull and creates a new curve.

**Use Case**: Identifies points that are near-optimal, forming a "second tier" of subgroups that almost made it to the hull.

```python
from true_roc_search import select_closest_points_to_hull

# Select 10 closest points to hull
result = select_closest_points_to_hull(
    points=roc_points,           # Array of (fpr, tpr) points
    n_points=10,                 # Number of points to select
    return_details=True          # Get full metrics
)

# Access results
print(f"New hull AUC: {result['new_auc']}")
print(f"Selected {result['n_selected']} points")
print(f"New hull has {result['new_num_subgroups']} points")
```

**Algorithm**:
1. Computes the original convex hull
2. Uses KDTree to find the Euclidean distance from each point to the nearest hull point
3. Selects the n points with smallest distances
4. Computes a new convex hull from selected points
5. Returns comprehensive metrics

**Key Metrics**:
- `selected_points`: The n closest points (may include original hull points)
- `new_hull`: Convex hull of selected points
- `new_auc`: AUC of the new hull
- `n_selected`: Actual number of points selected

---

### 2. select_furthest_points_from_diagonal()

Selects the n points that are furthest from the diagonal (y=x), i.e., highest ROC quality (TPR - FPR).

**Use Case**: Focuses on the highest-quality subgroups, regardless of their position on the hull.

```python
from true_roc_search import select_furthest_points_from_diagonal

# Select 15 furthest points from diagonal
result = select_furthest_points_from_diagonal(
    points=roc_points,
    n_points=15,
    return_details=True
)

# Access distance metrics
print(f"Avg distance from diagonal: {result['avg_distance_from_diagonal']:.4f}")
print(f"Max distance from diagonal: {result['max_distance_from_diagonal']:.4f}")
print(f"New AUC: {result['new_auc']:.4f}")
```

**Algorithm**:
1. Computes the original convex hull
2. Calculates distance from diagonal for each point as: TPR - FPR
3. Selects the n points with largest distances (highest quality)
4. Computes a new convex hull from selected points
5. Returns comprehensive metrics including distance statistics

**Key Metrics**:
- `avg_distance_from_diagonal`: Mean quality (TPR-FPR) of selected points
- `max_distance_from_diagonal`: Best quality in selection
- `min_distance_from_diagonal`: Worst quality in selection
- `new_auc`: AUC of the new hull

---

## Return Value Structure

Both functions return a dictionary with the following fields when `return_details=True`:

### Point Sets
- `original_hull`: Points on the original convex hull
- `new_hull`: Points on the new convex hull
- `selected_points`: All n selected points
- `all_points`: All original points (filtered above diagonal)

### Selection Information
- `selection_criterion`: Description of selection method
- `n_selected`: Actual number of points selected

### Hull Metrics
- `original_hull_area`: Area of original convex hull
- `new_hull_area`: Area of new convex hull
- `hull_area_reduction`: Difference in hull areas
- `reduction_percentage`: Percentage reduction in hull area

### ROC Metrics - Original Hull
- `original_auc`: AUC of original hull
- `original_num_subgroups`: Number of points on original hull
- `original_best_tpr`: TPR of best original subgroup
- `original_best_fpr`: FPR of best original subgroup
- `original_avg_quality`: Average quality (TPR-FPR) of original hull
- `original_max_quality`: Best quality on original hull

### ROC Metrics - New Hull
- `new_auc`: AUC of new hull
- `new_num_subgroups`: Number of points on new hull
- `new_best_tpr`: TPR of best new subgroup
- `new_best_fpr`: FPR of best new subgroup
- `new_avg_quality`: Average quality of new hull
- `new_max_quality`: Best quality on new hull

### Comparison Metrics
- `auc_reduction`: Difference between original and new AUC
- `auc_reduction_percentage`: Percentage change in AUC
- `quality_reduction`: Change in best quality score

### Additional (furthest from diagonal only)
- `avg_distance_from_diagonal`: Mean TPR-FPR for selected points
- `min_distance_from_diagonal`: Minimum TPR-FPR in selection
- `max_distance_from_diagonal`: Maximum TPR-FPR in selection

---

## Usage Examples

### Example 1: Compare Selection Methods

```python
import numpy as np
from true_roc_search import (
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal
)

# Generate or load ROC points
points = np.array([[0.1, 0.5], [0.2, 0.6], [0.3, 0.8], ...])

# Method 1: Closest to hull
closest = select_closest_points_to_hull(points, n_points=10, return_details=True)

# Method 2: Furthest from diagonal
furthest = select_furthest_points_from_diagonal(points, n_points=10, return_details=True)

# Compare results
print(f"Closest to hull - AUC: {closest['new_auc']:.4f}")
print(f"Furthest from diagonal - AUC: {furthest['new_auc']:.4f}")
print(f"Furthest avg quality: {furthest['avg_distance_from_diagonal']:.4f}")
```

### Example 2: Varying Selection Size

```python
# Test different numbers of selected points
for n in [5, 10, 15, 20]:
    result = select_furthest_points_from_diagonal(
        points, 
        n_points=n, 
        return_details=True
    )
    
    print(f"N={n}: AUC={result['new_auc']:.4f}, "
          f"Hull points={result['new_num_subgroups']}, "
          f"Avg quality={result['avg_distance_from_diagonal']:.4f}")
```

### Example 3: Integration with ROC Search

```python
# After running ROC search
from true_roc_search import true_roc_search

results = true_roc_search(
    data=df,
    target_col='target',
    alphas=[0.5],
    max_depth=3
)

# Get final subgroups
final_subgroups = results[0.5]['final_subgroups']
points = np.array([(sg['fpr'], sg['tpr']) for sg in final_subgroups])

# Analyze top performers
top_quality = select_furthest_points_from_diagonal(
    points, 
    n_points=10, 
    return_details=True
)

print(f"Top 10 subgroups AUC: {top_quality['new_auc']:.4f}")
print(f"Average quality: {top_quality['avg_distance_from_diagonal']:.4f}")
```

### Example 4: Visualizing Results

```python
import matplotlib.pyplot as plt

# Get both selections
closest = select_closest_points_to_hull(points, 15, return_details=True)
furthest = select_furthest_points_from_diagonal(points, 15, return_details=True)

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Closest to hull
ax1.scatter(closest['all_points'][:, 0], closest['all_points'][:, 1], 
           alpha=0.3, label='All points')
ax1.scatter(closest['selected_points'][:, 0], closest['selected_points'][:, 1],
           color='green', label='Selected')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_title('Closest to Hull')
ax1.legend()

# Plot 2: Furthest from diagonal
ax2.scatter(furthest['all_points'][:, 0], furthest['all_points'][:, 1],
           alpha=0.3, label='All points')
ax2.scatter(furthest['selected_points'][:, 0], furthest['selected_points'][:, 1],
           color='red', label='Selected')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_title('Furthest from Diagonal')
ax2.legend()

plt.tight_layout()
plt.show()
```

---

## Comparison: Three Selection Methods

| Method | Selection Criterion | Use Case | Expected AUC |
|--------|-------------------|----------|--------------|
| **Remove Hull Points** | Remove original hull, use remaining | Evaluate robustness | Lower |
| **Closest to Hull** | Points near hull | Second-tier subgroups | Similar/lower |
| **Furthest from Diagonal** | Highest quality (TPR-FPR) | Top performers | Similar/higher |

### When to Use Each Method

**Remove Hull Points**:
- Testing hull stability
- Finding alternative interpretations
- Robustness analysis

**Closest to Hull**:
- Finding near-optimal alternatives
- Diversity in high-quality subgroups
- Identifying "second best" options

**Furthest from Diagonal**:
- Focus on highest quality only
- Performance-driven selection
- Identifying dominant subgroups

---

## Testing

Run the comprehensive test suite:

```bash
python test_point_selection.py
```

This will:
1. Generate synthetic ROC points
2. Test both selection methods with n=10, 15, 20
3. Create 6-panel comparison visualizations
4. Print detailed metrics and summary table
5. Save results to `runs/point_selection_test/`

---

## Performance Notes

### Computational Complexity

- **select_closest_points_to_hull**: O(n log n) due to KDTree construction
- **select_furthest_points_from_diagonal**: O(n log n) due to sorting

Both are efficient for typical ROC search results (n < 1000 points).

### Memory Usage

Both functions create copies of point arrays and are safe for large datasets.

---

## Advanced Usage

### Custom Visualization

You can use the returned data to create custom plots:

```python
result = select_furthest_points_from_diagonal(points, 10, return_details=True)

# Extract specific data
selected = result['selected_points']
new_hull = result['new_hull']
qualities = selected[:, 1] - selected[:, 0]  # TPR - FPR for each

# Custom plot
plt.scatter(selected[:, 0], selected[:, 1], c=qualities, cmap='viridis')
plt.colorbar(label='Quality (TPR-FPR)')
plt.title('Selected Points by Quality')
plt.show()
```

### Exporting Results

```python
import pandas as pd

result = select_furthest_points_from_diagonal(points, 15, return_details=True)

# Create DataFrame
df = pd.DataFrame({
    'fpr': result['selected_points'][:, 0],
    'tpr': result['selected_points'][:, 1],
    'quality': result['selected_points'][:, 1] - result['selected_points'][:, 0]
})

df.to_csv('selected_subgroups.csv', index=False)
```

---

## Troubleshooting

### Issue: "Not enough points to form hull"

**Solution**: Ensure n_points ≥ 3 and that you have enough points above the diagonal.

```python
# Filter first
points_above = points[points[:, 1] > points[:, 0]]
if len(points_above) >= 3:
    result = select_closest_points_to_hull(points_above, 10, return_details=True)
```

### Issue: AUC change is 0% with closest to hull

**Explanation**: If the selected points include all original hull points, the new hull will be identical.

**Solution**: This is expected behavior - it means the closest points already define the hull.

### Issue: New hull has fewer points than n_points

**Explanation**: Not all selected points will be on the convex hull - only those on the boundary.

**Solution**: This is correct. Access `new_num_subgroups` for hull points, `n_selected` for selected points.

---

## References

- Original function: `remove_hull_points_and_recalculate()`
- Test script: `test_point_selection.py`
- Visualization: `plot_selection_comparison()`

For more information on ROC analysis, see `METRICS_REFERENCE.md`.
