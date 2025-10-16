# exclude_hull_points Parameter - Rollback Guide

## Overview

The `exclude_hull_points` parameter was added to both point selection functions to control whether original hull points can be included in the selection. This addresses the issue where the curve wouldn't change when selecting points.

## What Changed

### Modified Functions
1. `select_closest_points_to_hull(points, n_points, return_details=False, exclude_hull_points=False)`
2. `select_furthest_points_from_diagonal(points, n_points, return_details=False, exclude_hull_points=False)`

### New Parameter
- **`exclude_hull_points`**: Boolean (default: `False`)
  - `False`: Original behavior - may include original hull points in selection
  - `True`: Only select from non-hull points - **guarantees a different curve**

## Why This Was Needed

### The Problem
When selecting "closest points to hull" with the original implementation:
```python
# Original hull has 6 points
result = select_closest_points_to_hull(points, n_points=10)

# Result: AUC change = 0.00%
# Why? The 10 "closest" points include all 6 hull points (distance=0)
# So the new hull is identical to the original hull!
```

**The curve didn't change** because:
1. Original hull points have distance = 0 from the hull (they ARE the hull)
2. When selecting N closest points, we select all hull points + (N - hull_size) interior points
3. The convex hull of these points is still defined by the same boundary points
4. Result: No change in AUC or curve shape

### The Solution
Add `exclude_hull_points=True` to force selection from non-hull points only:
```python
# Force a different curve
result = select_closest_points_to_hull(points, n_points=10, exclude_hull_points=True)

# Result: AUC change = 4.54%
# The curve is now different because we're excluding the original hull
```

## Test Results Comparison

From `test_point_selection.py` with N=10:

| Method | exclude_hull | New Hull Pts | AUC | AUC Δ% | Quality Δ |
|--------|-------------|--------------|-----|--------|-----------|
| Closest to Hull | False (default) | 6 | 0.8026 | **0.00%** | 0.0000 |
| Closest to Hull | **True** | 6 | 0.7661 | **4.54%** | 0.0299 |
| Furthest from Diagonal | False (default) | 4 | 0.7870 | 1.94% | 0.0000 |
| Furthest from Diagonal | **True** | 3 | 0.7536 | **6.10%** | 0.0052 |

**Key Observation**: 
- With `exclude_hull_points=False`: "Closest to Hull" has 0% AUC change (curve unchanged)
- With `exclude_hull_points=True`: All methods show clear curve changes (3-6% AUC reduction)

## When to Use Each Setting

### Use `exclude_hull_points=False` (default) when:
✓ You want to test if including nearby points affects the hull  
✓ You're interested in hull stability  
✓ You want to allow the "best" points even if they're on the hull  
✓ You're not concerned if the curve stays the same  

**Example Use Case**: Testing hull robustness - "If I include the 10 best points, do I still get the same hull?"

### Use `exclude_hull_points=True` when:
✓ You **must** have a different curve from the original  
✓ You want to analyze "second-tier" or "backup" subgroups  
✓ You're comparing alternative ROC curves  
✓ You want to see performance without the top performers  

**Example Use Case**: Diversity analysis - "What does the ROC curve look like with only non-hull points?"

## Code Implementation Details

### Location in true_roc_search.py

**Function 1: select_closest_points_to_hull** (around line 420)
```python
# MODIFICATION: Option to exclude hull points from selection
# This ensures the new curve will be different from the original
if exclude_hull_points:
    # Select only from non-hull points
    non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
    non_hull_mask[original_hull_indices] = False
    candidate_points = above_diagonal[non_hull_mask]
    candidate_indices = np.where(non_hull_mask)[0]
    
    if len(candidate_points) < 3:
        # Not enough non-hull points
        return ...
else:
    # Include all points (original behavior - may include hull points)
    candidate_points = above_diagonal
    candidate_indices = np.arange(len(above_diagonal))
```

**Function 2: select_furthest_points_from_diagonal** (around line 590)
```python
# MODIFICATION: Option to exclude hull points from selection
# This ensures the new curve will be different from the original
if exclude_hull_points:
    # Select only from non-hull points
    non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
    non_hull_mask[original_hull_indices] = False
    candidate_points = above_diagonal[non_hull_mask]
    
    if len(candidate_points) < 3:
        # Not enough non-hull points
        return ...
else:
    # Include all points (original behavior - may include hull points)
    candidate_points = above_diagonal
```

## How to Rollback

If you need to revert to the original behavior (no `exclude_hull_points` parameter):

### Step 1: Remove the parameter from function signatures
Change:
```python
def select_closest_points_to_hull(points, n_points, return_details=False, exclude_hull_points=False):
```

Back to:
```python
def select_closest_points_to_hull(points, n_points, return_details=False):
```

### Step 2: Remove the conditional logic
Delete the `if exclude_hull_points:` block and keep only the `else:` branch:

```python
# OLD CODE (with parameter):
if exclude_hull_points:
    non_hull_mask = ...
    candidate_points = above_diagonal[non_hull_mask]
else:
    candidate_points = above_diagonal

# ROLLBACK TO:
candidate_points = above_diagonal  # Always include all points
```

### Step 3: Update all function calls
Remove `exclude_hull_points=...` from all calls:
```python
# OLD:
result = select_closest_points_to_hull(points, 10, True, exclude_hull_points=True)

# ROLLBACK TO:
result = select_closest_points_to_hull(points, 10, True)
```

## Testing the Change

Run the updated test script:
```bash
python test_point_selection.py
```

You should see:
- **4 tests per N value** (include/exclude for both methods)
- **Clear AUC differences** between include and exclude versions
- **Summary table** showing all variations

## Impact on Existing Code

### Backward Compatible
✓ Default value is `False` (original behavior)  
✓ Existing code without the parameter still works  
✓ No breaking changes to API  

### New Capability
✓ Can now force different curves with `exclude_hull_points=True`  
✓ Better control over point selection  
✓ More meaningful comparisons  

## Recommended Usage

```python
# For most analysis - use exclude_hull_points=True for clearer results
result = select_closest_points_to_hull(
    points, 
    n_points=15, 
    return_details=True,
    exclude_hull_points=True  # Force a different curve
)

# For hull stability testing - use default (False)
result = select_closest_points_to_hull(
    points, 
    n_points=15, 
    return_details=True
    # exclude_hull_points=False is default - may keep same curve
)
```

## Summary

**Before**: Curve wouldn't change when selecting closest points  
**After**: Can force curve change with `exclude_hull_points=True`  
**Default**: Original behavior preserved (`False`)  
**Rollback**: Simple - remove parameter and conditional logic  

---

**Date Modified**: October 15, 2025  
**Reason**: Address user observation that curves weren't changing  
**Files Modified**: 
- `true_roc_search.py` (functions: select_closest_points_to_hull, select_furthest_points_from_diagonal)
- `test_point_selection.py` (added tests for both modes)
