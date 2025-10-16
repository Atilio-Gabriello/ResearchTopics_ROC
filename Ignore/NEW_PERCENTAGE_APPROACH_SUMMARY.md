# New Percentage-Based Threshold Approach - Summary

## Overview

Modified functions 4 and 5 to use a **unified reference point** (the furthest point from the diagonal) for both distance calculations.

## Key Changes

### Previous Approach
- Function 4: Used minimum distance as reference
- Function 5: Used maximum distance as reference  
- Different reference points for each function

### NEW Approach
- **Both functions**: Use the MAXIMUM diagonal distance as the reference
- **Unified reference**: Same distance value (furthest point from diagonal)
- **Percentage-based thresholds**: Calculate thresholds as percentages of this reference

## Implementation Details

### Function 4: `select_points_below_hull`

**Reference Distance**: Maximum diagonal distance = max(TPR - FPR) across all points

**Threshold Calculation**:
```python
threshold = max_diagonal_distance × (distance_percentage / 100)
```

**Selection Criterion**:
- Calculate vertical distance below hull for each point
- Select points where `vertical_distance_below_hull ≤ threshold`

**Example** (with max_diagonal_distance = 0.8):
- Parameter 1%: threshold = 0.8 × 0.01 = 0.008
- Selects points within 0.008 of the hull
- Parameter 10%: threshold = 0.8 × 0.10 = 0.080
- Selects points within 0.080 of the hull

### Function 5: `select_points_above_diagonal`

**Reference Distance**: Same maximum diagonal distance = max(TPR - FPR)

**Threshold Calculation**:
```python
threshold = max_diagonal_distance × ((100 - distance_percentage) / 100)
```

**Selection Criterion**:
- Calculate diagonal distance (TPR - FPR) for each point
- Select points where `diagonal_distance ≥ threshold`

**Example** (with max_diagonal_distance = 0.8):
- Parameter 1%: threshold = 0.8 × 0.99 = 0.792
- Selects points at least 0.792 from diagonal (≥ 99% of max)
- Parameter 10%: threshold = 0.8 × 0.90 = 0.720
- Selects points at least 0.720 from diagonal (≥ 90% of max)

## Parameter Changes

### Function Signature Updates

**Before**:
```python
select_points_below_hull(points, n_points, return_details=False, exclude_hull_points=False)
select_points_above_diagonal(points, n_points, return_details=False, exclude_hull_points=False)
```

**After**:
```python
select_points_below_hull(points, distance_percentage=1.0, return_details=False, exclude_hull_points=True)
select_points_above_diagonal(points, distance_percentage=1.0, return_details=False, exclude_hull_points=True)
```

**Changes**:
- `n_points` → `distance_percentage` (percentage of max diagonal distance)
- Default `distance_percentage`: 1.0 (means 1%)
- Default `exclude_hull_points`: Changed from `False` to `True`

## Interpretation Guide

### What the Percentage Parameter Means

**For Function 4 (Below Hull)**:
- 1% = Include points within 1% of the max diagonal distance from the hull
- 10% = Include points within 10% of the max diagonal distance from the hull
- Higher percentage → More points selected → Larger threshold

**For Function 5 (Above Diagonal)**:
- 1% = Include points at least 99% as far as the furthest point
- 10% = Include points at least 90% as far as the furthest point  
- Higher percentage → Fewer points selected → Lower threshold

### Complementary Relationship

The two functions have a complementary relationship:
- Function 4: `threshold = ref × pct` (grows with percentage)
- Function 5: `threshold = ref × (1 - pct)` (shrinks with percentage)

This makes sense because:
- **Below hull**: We want to expand the tolerance as percentage increases
- **Above diagonal**: We want to be more selective (higher quality) as percentage increases

## Test Results Summary

Tested on 3 datasets (adult, ionosphere, mushroom) with percentages: 0.5%, 1%, 2%, 5%, 10%

### Key Findings

1. **Low percentages (0.5-2%)**: 
   - Very few or no points selected
   - Thresholds too tight for most datasets

2. **Medium percentages (5%)**: 
   - Function 4: Starts selecting points (6-9 points)
   - Function 5: Still very selective (0 points on some datasets)

3. **Higher percentages (10%)**:
   - Function 4: More points selected (9-12 points)
   - Function 5: Begins selecting points on some datasets

### Dataset-Specific Results

**Adult** (not tested in this run - need to check separately)

**Ionosphere**:
- Max diagonal distance: 0.5375
- 10% Function 4: 9 points selected, 3.26% AUC reduction
- 10% Function 5: 0 points (threshold 0.484 too high)

**Mushroom**:
- Max diagonal distance: 0.9040
- 5% Function 4: 6 points selected, 21.08% AUC reduction
- 10% Function 4: 12 points selected, 21.08% AUC reduction
- Function 5: 0 points even at 10% (threshold 0.814 very high)

## Return Dictionary Updates

Both functions now include additional fields in `return_details`:

```python
{
    # New fields
    'reference_distance': max_diagonal_distance,  # The unified reference
    'threshold_distance': threshold,              # Calculated threshold
    'distance_percentage': distance_percentage,   # Input parameter
    'n_selected': len(selected_points),          # Number of points selected
    
    # Existing fields (unchanged)
    'original_hull': ...,
    'new_hull': ...,
    'selected_points': ...,
    'all_points': ...,
    'original_auc': ...,
    'new_auc': ...,
    # ... etc
}
```

## Recommendations

Based on initial testing:

1. **Function 4 (Below Hull)**:
   - Use percentages between 5-20% for meaningful selection
   - Lower percentages (1-2%) may select too few points
   - Monitor AUC reduction to ensure quality

2. **Function 5 (Above Diagonal)**:
   - May need higher percentages (20-50%) to select points
   - Very selective due to using (100 - percentage) formula
   - Consider adjusting threshold calculation if too few points selected

3. **General**:
   - The unified reference distance provides consistency
   - Easy to compare behavior across functions
   - Percentage parameter is intuitive for users

## Files Modified

1. **`true_roc_search.py`**:
   - `select_points_below_hull()` - Lines 766-1007
   - `select_points_above_diagonal()` - Lines 1010-1239

2. **Test Scripts**:
   - `test_new_percentage_approach.py` - New comprehensive test

## Next Steps

1. **Calibration**: Test with wider range of percentages (0.5-50%)
2. **Comparison**: Compare with old n_points approach
3. **Fine-tuning**: Adjust threshold formulas if needed
4. **Documentation**: Update user guides and examples
5. **Integration**: Update main workflow to use new parameters
