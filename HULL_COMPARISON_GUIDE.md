# Hull Comparison Functionality Guide

## Overview

This guide describes the new hull comparison functionality that removes original convex hull points and recalculates the hull with the remaining points. This is useful for analyzing how the ROC convex hull changes across different search depths.

## Key Functions

### 1. `remove_hull_points_and_recalculate(points, return_details=False)`

Removes points on the original convex hull and recalculates the hull with remaining points.

**Parameters:**
- `points`: Array of (fpr, tpr) points, shape (n, 2)
- `return_details`: If True, returns detailed comparison information

**Returns:**
- If `return_details=False`: Array of new hull points
- If `return_details=True`: Dictionary with:
  - `original_hull`: Points on the original convex hull
  - `new_hull`: Points on the recalculated hull
  - `removed_points`: Points that were on the original hull
  - `remaining_points`: Points not on the original hull
  - `all_points`: All points above the diagonal
  - `original_hull_area`: Area under original hull
  - `new_hull_area`: Area under new hull
  - `hull_area_reduction`: Difference in areas
  - `reduction_percentage`: Percentage reduction

**Example:**
```python
import numpy as np
from true_roc_search import remove_hull_points_and_recalculate

# Generate sample ROC points
points = np.array([[0.1, 0.3], [0.2, 0.6], [0.3, 0.7], [0.5, 0.8], [0.7, 0.9]])

# Get detailed comparison
hull_data = remove_hull_points_and_recalculate(points, return_details=True)

print(f"Original hull points: {len(hull_data['original_hull'])}")
print(f"New hull points: {len(hull_data['new_hull'])}")
print(f"Area reduction: {hull_data['hull_area_reduction']:.4f}")
```

### 2. `plot_hull_comparison(hull_data, depth, output_path=None, title_suffix="")`

Creates a three-panel visualization comparing original and recalculated hulls.

**Parameters:**
- `hull_data`: Dictionary from `remove_hull_points_and_recalculate` with `return_details=True`
- `depth`: Search depth for labeling
- `output_path`: Path to save the plot (optional)
- `title_suffix`: Additional text for the title

**Visualization includes:**
- Panel 1: Original hull with all points
- Panel 2: New hull with remaining points
- Panel 3: Overlay comparison with statistics

**Example:**
```python
from true_roc_search import plot_hull_comparison

# After getting hull_data
plot_hull_comparison(hull_data, depth=2, 
                    output_path='./results/hull_comparison_depth_2.png',
                    title_suffix=' (Adult Dataset)')
```

### 3. `demonstrate_hull_comparison(points, depth=1, output_dir=None)`

Standalone demonstration function that combines hull comparison and visualization.

**Parameters:**
- `points`: Array of (fpr, tpr) points
- `depth`: Depth label for visualization
- `output_dir`: Optional output directory for saving plots

**Returns:**
- Dictionary with hull comparison results

**Example:**
```python
from true_roc_search import demonstrate_hull_comparison

points = np.random.rand(50, 2)  # Random points
hull_data = demonstrate_hull_comparison(points, depth=1, output_dir='./results')
```

## Integration with True ROC Search

The hull comparison functionality is automatically integrated into the `true_roc_search` function:

```python
from true_roc_search import true_roc_search, load_data

# Load data
data = load_data('./tests/adult.txt')

# Run search with hull comparison tracking
results = true_roc_search(data, target_col='target', alphas=[0.5], max_depth=3)

# Access hull comparison data
for alpha, result in results.items():
    if 'hull_comparisons' in result:
        for hull_data in result['hull_comparisons']:
            depth = hull_data['depth']
            reduction = hull_data.get('hull_area_reduction', 0)
            print(f"Depth {depth}: Area reduction = {reduction:.4f}")
```

## Output Files

When running `true_roc_search`, hull comparison data is saved to:

```
output_dir/
├── alpha_0.5/
│   ├── hull_comparisons/
│   │   ├── hull_comparison_summary.csv
│   │   ├── hull_comparison_depth_1.png
│   │   ├── hull_comparison_depth_2.png
│   │   └── hull_comparison_depth_3.png
│   ├── subgroups.csv
│   └── roc_curve.png
```

### Hull Comparison Summary CSV

Contains the following columns:
- `depth`: Search depth
- `original_hull_points`: Number of points on original hull
- `new_hull_points`: Number of points on new hull
- `remaining_points`: Number of points after removing hull points
- `original_hull_area`: Area under original hull
- `new_hull_area`: Area under new hull
- `hull_area_reduction`: Absolute area reduction
- `reduction_percentage`: Percentage reduction

## Testing

Run the test script to verify functionality:

```bash
python test_hull_comparison.py
```

This will:
1. Test basic hull comparison
2. Compare hulls across multiple depths
3. Test edge cases (few points, points below diagonal, many points)
4. Generate visualizations in `./runs/hull_test/`

## Use Cases

### 1. Understanding Search Behavior
Analyze how the convex hull evolves as search depth increases:

```python
results = true_roc_search(data, 'target', alphas=[None], max_depth=5)

for hull_data in results['pure_roc']['hull_comparisons']:
    depth = hull_data['depth']
    reduction = hull_data.get('reduction_percentage', 0)
    print(f"Depth {depth}: {reduction:.1f}% hull area reduction")
```

### 2. Comparing Algorithms
Compare hull evolution between different alpha values:

```python
results = true_roc_search(data, 'target', alphas=[0.3, 0.5, 0.7], max_depth=3)

for alpha, result in results.items():
    print(f"\nAlpha = {alpha}")
    for hull_data in result['hull_comparisons']:
        print(f"  Depth {hull_data['depth']}: "
              f"{len(hull_data['original_hull'])} -> "
              f"{len(hull_data.get('new_hull', []))} hull points")
```

### 3. Quality Assessment
Assess the quality of subgroup discovery by examining hull stability:

```python
hull_data = remove_hull_points_and_recalculate(points, return_details=True)

stability_ratio = hull_data['new_hull_area'] / hull_data['original_hull_area']
print(f"Hull stability: {stability_ratio:.2%}")

if stability_ratio > 0.8:
    print("High stability - good subgroup diversity")
else:
    print("Low stability - dominated by few subgroups")
```

## Interpretation

### Hull Area Reduction
- **High reduction (>50%)**: Original hull dominated by a few key points
- **Medium reduction (20-50%)**: Balanced hull with moderate diversity
- **Low reduction (<20%)**: Many points contribute to hull, high diversity

### New Hull Points
- **Many new hull points**: Good subgroup diversity beneath top performers
- **Few new hull points**: Search may be converging to specific regions
- **No new hull points**: All non-hull points are far from optimal

## Best Practices

1. **Always use `return_details=True`** for analysis to get full statistics
2. **Save visualizations** for each depth to track progression
3. **Compare across alphas** to understand parameter sensitivity
4. **Monitor reduction percentage** as an indicator of search behavior
5. **Check remaining points** to ensure sufficient diversity

## Troubleshooting

### Issue: No new hull points
**Cause:** All remaining points are below the diagonal or too few points
**Solution:** Increase `min_coverage` or adjust search parameters

### Issue: Very small area reduction
**Cause:** Original hull uses many points (high diversity)
**Solution:** This is normal for well-distributed point sets

### Issue: Error during hull calculation
**Cause:** Insufficient points to form convex hull (<3 points)
**Solution:** Check that candidate generation produces enough valid points

## References

For more information on ROC convex hull analysis:
- Provost, F., & Fawcett, T. (2001). Robust classification for imprecise environments. Machine Learning, 42(3), 203-231.
- Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.
