# Quick Reference: ROC Hull Manipulation Functions

## Function Summary

| Function | Selection Method | Points Returned | Use Case |
|----------|-----------------|-----------------|----------|
| `remove_hull_points_and_recalculate` | Remove original hull points | Remaining points | Robustness testing |
| `select_closest_points_to_hull` | N closest to hull | N selected points | Near-optimal alternatives |
| `select_furthest_points_from_diagonal` | N highest quality | N selected points | Top performers |

---

## Quick Comparison

### remove_hull_points_and_recalculate()

```python
result = remove_hull_points_and_recalculate(points, return_details=True)
```

**What it does**: Removes points on the original convex hull, then creates a new hull from remaining points

**Outputs**:
- `original_hull`: Original hull points (removed)
- `new_hull`: New hull from remaining points
- `removed_points`: Points that were on original hull
- `remaining_points`: All non-hull points

**Typical Results**:
- New hull < Original hull (area decreases)
- AUC reduction: 1-3%
- Fewer subgroups on new hull

---

### select_closest_points_to_hull()

```python
result = select_closest_points_to_hull(points, n_points=10, return_details=True)
```

**What it does**: Selects N points closest to the original hull (Euclidean distance)

**Outputs**:
- `original_hull`: Original hull points
- `new_hull`: Hull of N selected points
- `selected_points`: The N closest points
- `n_selected`: Actual count selected

**Typical Results**:
- New hull ≈ Original hull (minimal change)
- AUC reduction: 0-2%
- Similar quality to original

**Use when**: Finding "second tier" high-quality subgroups

---

### select_furthest_points_from_diagonal()

```python
result = select_furthest_points_from_diagonal(points, n_points=10, return_details=True)
```

**What it does**: Selects N points with highest quality (TPR - FPR)

**Outputs**:
- `original_hull`: Original hull points
- `new_hull`: Hull of N selected points
- `selected_points`: The N highest-quality points
- `avg_distance_from_diagonal`: Mean quality
- `max_distance_from_diagonal`: Best quality

**Typical Results**:
- New hull ≈ Original hull (quality-focused)
- AUC reduction: 0-2%
- High average quality

**Use when**: Focusing on best-performing subgroups only

---

## Example Workflow

```python
import numpy as np
from true_roc_search import (
    remove_hull_points_and_recalculate,
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal
)

# Your ROC points (fpr, tpr)
points = np.array([[0.1, 0.5], [0.2, 0.7], ...])

# Method 1: Remove hull and recalculate
removed = remove_hull_points_and_recalculate(points, return_details=True)
print(f"After removing hull: {removed['new_auc']:.4f}")

# Method 2: Select 15 closest to hull
closest = select_closest_points_to_hull(points, 15, return_details=True)
print(f"Closest 15: {closest['new_auc']:.4f}")

# Method 3: Select 15 highest quality
furthest = select_furthest_points_from_diagonal(points, 15, return_details=True)
print(f"Top 15 quality: {furthest['new_auc']:.4f}, avg={furthest['avg_distance_from_diagonal']:.4f}")
```

---

## Common Metrics (All Functions)

### Point Counts
- `all_points`: All original points (above diagonal)
- `original_hull`: Points on original convex hull
- `new_hull`: Points on new convex hull

### AUC Metrics
- `original_auc`: Original hull AUC
- `new_auc`: New hull AUC
- `auc_reduction`: Absolute difference
- `auc_reduction_percentage`: Percentage change

### Quality Metrics
- `original_max_quality`: Best TPR-FPR on original hull
- `new_max_quality`: Best TPR-FPR on new hull
- `original_avg_quality`: Average quality
- `new_avg_quality`: Average quality
- `quality_reduction`: Change in best quality

### Hull Metrics
- `original_hull_area`: Convex hull area (original)
- `new_hull_area`: Convex hull area (new)
- `hull_area_reduction`: Absolute change
- `reduction_percentage`: Percentage change

---

## Visualization

All three can use the same plotting function:

```python
from true_roc_search import plot_hull_comparison

# Get results
result = select_furthest_points_from_diagonal(points, 10, return_details=True)

# Plot
plot_hull_comparison(
    hull_data=result,
    depth=1,
    output_path='./results/comparison.png',
    title_suffix=' - Top 10 Subgroups'
)
```

This creates a 3-panel plot:
1. Original hull
2. New hull
3. Overlay comparison

---

## Testing

```bash
# Test all three methods
python test_hull_comparison.py          # Tests remove_hull_points
python test_point_selection.py          # Tests closest & furthest

# Generate comparison tables
python create_metrics_comparison_table.py
```

---

## Decision Guide

**Choose `remove_hull_points_and_recalculate` when:**
- Testing robustness of hull
- Finding completely alternative subgroups
- Evaluating what happens without top performers

**Choose `select_closest_points_to_hull` when:**
- Finding near-optimal alternatives
- Exploring "close but not quite" subgroups
- Building diverse portfolios

**Choose `select_furthest_points_from_diagonal` when:**
- Focusing on quality above all
- Building a "top N" leaderboard
- Maximizing performance metrics

---

## Performance Characteristics

| Method | Time Complexity | Space | Deterministic |
|--------|----------------|-------|---------------|
| Remove hull | O(n log n) | O(n) | Yes |
| Closest to hull | O(n log n) | O(n) | Yes |
| Furthest from diagonal | O(n log n) | O(n) | Yes |

All three are efficient for typical ROC search outputs (< 1000 points).

---

## Tips & Tricks

### 1. Combine Multiple Approaches

```python
# Get insights from all three methods
methods = {
    'remove_hull': remove_hull_points_and_recalculate(points, True),
    'closest_10': select_closest_points_to_hull(points, 10, True),
    'top_10': select_furthest_points_from_diagonal(points, 10, True)
}

for name, result in methods.items():
    print(f"{name}: AUC={result['new_auc']:.4f}")
```

### 2. Vary Selection Size

```python
# Test different N values
for n in [5, 10, 15, 20]:
    result = select_furthest_points_from_diagonal(points, n, True)
    print(f"N={n}: AUC={result['new_auc']:.4f}, Hull={result['new_num_subgroups']}")
```

### 3. Export for Analysis

```python
import pandas as pd

result = select_furthest_points_from_diagonal(points, 15, True)

# Create comparison table
comparison = pd.DataFrame({
    'Metric': ['AUC', 'Hull Points', 'Avg Quality', 'Max Quality'],
    'Original': [
        result['original_auc'],
        result['original_num_subgroups'],
        result['original_avg_quality'],
        result['original_max_quality']
    ],
    'New': [
        result['new_auc'],
        result['new_num_subgroups'],
        result['new_avg_quality'],
        result['new_max_quality']
    ]
})

print(comparison)
comparison.to_csv('method_comparison.csv', index=False)
```

---

## Documentation

- **Full Guide**: `POINT_SELECTION_GUIDE.md`
- **Metrics Reference**: `METRICS_REFERENCE.md`
- **Hull Comparison**: `HULL_COMPARISON_GUIDE.md`
- **Implementation**: `true_roc_search.py`

---

## Example Output

```
=== Hull Comparison Statistics ===

--- Point Counts ---
Total points: 50
Original hull points: 6
Selected points: 10
New hull points: 4

--- AUC Metrics ---
Original hull AUC: 0.8026
New hull AUC: 0.7870
AUC reduction: 0.0156 (1.9%)

--- Quality Metrics ---
Original best quality (TPR-FPR): 0.3961
New best quality (TPR-FPR): 0.3961
Quality reduction: 0.0000

--- Distance from Diagonal (furthest method only) ---
Avg distance: 0.3658
Max distance: 0.3961
Min distance: 0.3182
```
