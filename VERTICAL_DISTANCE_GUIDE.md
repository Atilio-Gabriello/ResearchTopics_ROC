# Vertical Distance Selection Functions - Quick Reference

## Overview
Two new functions for selecting points based on **vertical (y-axis) distance metrics** in ROC space.

## Functions

### 1. `select_points_below_hull()`
**Purpose**: Select N points with smallest vertical distance below the convex hull

**Key Features**:
- Uses **linear interpolation** to find hull TPR at each point's FPR
- Calculates `vertical_distance = hull_tpr - point_tpr`
- Selects points with **largest positive distances** (furthest below hull)
- Fallback: If no points below hull, selects closest points

**Usage**:
```python
result = select_points_below_hull(
    points,                     # ROC points [(fpr, tpr), ...]
    n_points=10,                # Number of points to select
    return_details=True,        # Return full metrics
    exclude_hull_points=False   # True = force curve change
)
```

**Unique Metrics Returned**:
- `avg_vertical_distance`: Average distance below hull
- `max_vertical_distance`: Largest distance below hull
- `min_vertical_distance`: Smallest distance below hull
- `num_points_below_hull`: How many points are actually below hull
- Plus standard 40+ metrics (AUC, quality, hull info, etc.)

**When to Use**:
- Select **worst performers** relative to hull
- Create **pessimistic** ROC curves
- Study points that **underperform** the envelope
- Analyze **vertical degradation** from optimal curve

---

### 2. `select_points_above_diagonal()`
**Purpose**: Select N points with highest TPR values (largest vertical distance above diagonal)

**Key Features**:
- Uses **TPR directly** as distance metric (diagonal is y=x)
- Vertical distance from diagonal = TPR value
- Selects **highest TPR points** (best performers in y-direction)
- Simpler than below_hull (no interpolation needed)

**Usage**:
```python
result = select_points_above_diagonal(
    points,                     # ROC points [(fpr, tpr), ...]
    n_points=10,                # Number of points to select
    return_details=True,        # Return full metrics
    exclude_hull_points=False   # True = force curve change
)
```

**Unique Metrics Returned**:
- `avg_tpr_selected`: Average TPR of selected points
- `max_tpr_selected`: Highest TPR in selection
- `min_tpr_selected`: Lowest TPR in selection
- Plus standard 40+ metrics (AUC, quality, hull info, etc.)

**When to Use**:
- Select **best performers** by TPR magnitude
- Create **optimistic** ROC curves
- Emphasize points with **high sensitivity**
- Study **vertical distance** from diagonal

---

## Comparison with Horizontal Distance Methods

| Method | Distance Metric | Selection Criterion | Primary Use |
|--------|----------------|---------------------|-------------|
| **Closest to Hull** | Euclidean (horizontal) | Nearest to hull surface | Preserve hull shape |
| **Furthest from Diagonal** | TPR-FPR (diagonal) | Largest quality gap | Emphasize quality difference |
| **Below Hull** ⭐ | Vertical (below) | Furthest below hull | Select worst performers |
| **Above Diagonal** ⭐ | Vertical (above) | Highest TPR | Select best performers |

## Parameter: `exclude_hull_points`

**Default**: `False` (include hull points in candidates)
**Set to `True`**: Force curve changes by excluding original hull points

### Behavior Comparison:

```python
# Example with 10 points

# Include hull (may result in identical curve)
result_inc = select_points_below_hull(points, 10, exclude_hull_points=False)
# AUC change: May be 0% if hull points selected

# Exclude hull (forces different curve)
result_exc = select_points_below_hull(points, 10, exclude_hull_points=True)
# AUC change: Guaranteed > 0% (curve must change)
```

**Test Results** (N=10):
- Below Hull (include): 26.22% AUC change
- Below Hull (exclude): 26.22% AUC change (same - no hull points in worst performers)
- Above Diagonal (include): 12.90% AUC change
- Above Diagonal (exclude): 11.84% AUC change (different - hull excluded)

---

## Return Value Structure

Both functions return identical structure with 40+ fields:

```python
{
    # Selection info
    'selected_points': array([[fpr, tpr], ...]),    # N selected points
    'n_points_selected': int,                        # Should equal n_points
    
    # Original hull
    'original_hull': array([[fpr, tpr], ...]),      # Original convex hull
    'original_num_hull_points': int,
    'original_auc': float,
    'original_best_tpr': float,
    'original_best_fpr': float,
    'original_avg_quality': float,
    'original_max_quality': float,
    
    # New hull (from selected points)
    'new_hull': array([[fpr, tpr], ...]),          # New convex hull
    'new_num_hull_points': int,
    'new_auc': float,
    'new_best_tpr': float,
    'new_best_fpr': float,
    'new_avg_quality': float,
    'new_max_quality': float,
    
    # Comparison metrics
    'auc_reduction': float,                         # Absolute AUC change
    'auc_reduction_percentage': float,              # % change
    'quality_reduction': float,                     # Quality change
    'hull_points_reduction': int,                   # Hull size change
    
    # Method-specific metrics
    # For below_hull:
    'avg_vertical_distance': float,
    'max_vertical_distance': float,
    'min_vertical_distance': float,
    'num_points_below_hull': int,
    
    # For above_diagonal:
    'avg_tpr_selected': float,
    'max_tpr_selected': float,
    'min_tpr_selected': float
}
```

---

## Implementation Details

### Below Hull Algorithm:
1. Calculate original convex hull
2. For each point:
   - Use `np.interp()` to find hull TPR at point's FPR
   - Calculate `distance = hull_tpr - point_tpr`
3. Sort by distance (descending = furthest below)
4. Select top N points
5. If exclude_hull=True, filter out original hull points first
6. Calculate new hull and metrics

### Above Diagonal Algorithm:
1. Calculate original convex hull
2. For each point:
   - Use TPR directly as distance metric
3. Sort by TPR (descending = highest TPR)
4. Select top N points
5. If exclude_hull=True, filter out original hull points first
6. Calculate new hull and metrics

---

## Code Location

**File**: `true_roc_search.py`

**Lines**:
- `select_points_below_hull()`: ~766-920
- `select_points_above_diagonal()`: ~922-1050

**MODIFICATION Comments**: Both functions include rollback documentation

---

## Testing

**Test Files**:
- `test_vertical_distance_selection.py` - Dedicated tests for vertical methods
- `test_all_selection_methods.py` - Comparison of all 4 methods

**Sample Results** (N=10, exclude_hull=True):
```
Method                    AUC Change    Hull Points    Quality Δ
Closest to Hull            4.54%            6          0.0299
Furthest from Diagonal     6.10%            3          0.0052
Below Hull                26.22%            3          0.2365  ← Largest change
Above Diagonal            11.84%            4          0.0276
```

---

## Examples

### Example 1: Select Worst Performers
```python
# Get 15 points furthest below hull (worst performers)
result = select_points_below_hull(roc_points, 15, return_details=True)

print(f"Selected {result['n_points_selected']} worst points")
print(f"Average distance below hull: {result['avg_vertical_distance']:.4f}")
print(f"AUC dropped by {result['auc_reduction_percentage']:.1f}%")
```

### Example 2: Select Best Performers
```python
# Get 20 highest-TPR points (best performers)
result = select_points_above_diagonal(roc_points, 20, return_details=True)

print(f"Selected {result['n_points_selected']} best points")
print(f"Average TPR: {result['avg_tpr_selected']:.4f}")
print(f"Max TPR: {result['max_tpr_selected']:.4f}")
```

### Example 3: Force Curve Changes
```python
# Ensure curve changes by excluding hull points
result = select_points_below_hull(
    roc_points, 
    10, 
    return_details=True,
    exclude_hull_points=True  # Guaranteed curve change
)

print(f"Original hull: {result['original_num_hull_points']} points")
print(f"New hull: {result['new_num_hull_points']} points")
print(f"AUC change: {result['auc_reduction_percentage']:.2f}%")
```

---

## Visualization

Both functions work with existing visualization tools:

```python
# Plot comparison
import matplotlib.pyplot as plt

result = select_points_below_hull(points, 10, return_details=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].scatter(points[:, 0], points[:, 1], alpha=0.3)
hull = result['original_hull']
hull_sorted = hull[np.argsort(hull[:, 0])]
axes[0].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2)

# Selected
selected = result['selected_points']
axes[1].scatter(selected[:, 0], selected[:, 1])

# Overlay
axes[2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', label='Original')
new_hull = result['new_hull']
new_sorted = new_hull[np.argsort(new_hull[:, 0])]
axes[2].plot(new_sorted[:, 0], new_sorted[:, 1], 'b--', label='New')

plt.show()
```

---

## Summary

✅ **select_points_below_hull**: Vertical distance below hull → worst performers
✅ **select_points_above_diagonal**: TPR magnitude → best performers
✅ Both return 40+ comprehensive metrics
✅ Both support exclude_hull_points parameter
✅ Complement horizontal (Euclidean) and diagonal (TPR-FPR) methods
✅ Fully tested and documented

**Total Selection Methods**: 4 (horizontal, diagonal, vertical-below, vertical-above)
