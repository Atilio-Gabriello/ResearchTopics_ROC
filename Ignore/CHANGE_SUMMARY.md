# Change Summary: exclude_hull_points Parameter

## What You Observed
✓ **Correct observation!** Sometimes the curve wasn't changing despite selecting different points.

## Why It Happened
When selecting "closest points to hull":
- Original hull points have **distance = 0** from the hull (they ARE the hull)
- Selecting N=10 closest points → includes all 6 hull points + 4 interior points
- New hull is still defined by same 6 boundary points
- **Result: AUC change = 0% (curve unchanged)**

This is **mathematically correct** but not always useful for analysis.

## The Fix
Added `exclude_hull_points` parameter to both functions:

```python
# DEFAULT BEHAVIOR (may include hull points - curve might not change)
result = select_closest_points_to_hull(points, 10, return_details=True)
# Result with N=10: AUC = 0.8026, AUC change = 0.00%

# FORCE DIFFERENT CURVE (exclude hull points - guarantees change)
result = select_closest_points_to_hull(points, 10, return_details=True, exclude_hull_points=True)
# Result with N=10: AUC = 0.7661, AUC change = 4.54%
```

## Results Comparison (N=10)

| Method | exclude_hull | AUC Change | Notes |
|--------|-------------|------------|-------|
| Closest (default) | False | **0.00%** ❌ | Same curve |
| Closest (forced) | **True** | **4.54%** ✓ | Different curve |
| Furthest (default) | False | 1.94% | Usually changes |
| Furthest (forced) | **True** | **6.10%** ✓ | Guaranteed change |

## When to Use

**Use `exclude_hull_points=False` (default)**:
- Testing hull stability
- "Do nearby points form the same hull?"
- Allow best points even if on hull

**Use `exclude_hull_points=True`**:
- **Force a different curve** (recommended for comparison analysis)
- Analyze "second-tier" subgroups
- Find backup/alternative patterns

## Files Modified

1. ✅ `true_roc_search.py` - Added parameter to both functions
2. ✅ `test_point_selection.py` - Tests both modes
3. ✅ `EXCLUDE_HULL_ROLLBACK_GUIDE.md` - Complete rollback instructions

## Rollback Instructions

See `EXCLUDE_HULL_ROLLBACK_GUIDE.md` for step-by-step rollback guide with:
- What changed
- Where to find the code
- How to revert
- Impact assessment

## Comments Added

All modifications marked with:
```python
# MODIFICATION: Option to exclude hull points from selection
# This ensures the new curve will be different from the original
```

You can easily search for "MODIFICATION" to find all changes.

## Backward Compatible

✓ Default is `False` (original behavior)  
✓ Existing code still works  
✓ No breaking changes  

## Test It

```bash
python test_point_selection.py
```

Look for the summary table showing **4 variations per N value**:
- Closest (include hull) vs Closest (exclude hull)
- Furthest (include hull) vs Furthest (exclude hull)

---

**Bottom Line**: You were right! The curve wasn't changing. Now you have control with the `exclude_hull_points` parameter, and full rollback documentation if needed.
