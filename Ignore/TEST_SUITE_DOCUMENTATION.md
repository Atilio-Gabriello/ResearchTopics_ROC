# Comprehensive Test Suite Documentation

## Overview

This document describes the comprehensive testing framework for evaluating hull manipulation functions and the True ROC Search algorithm at depth 3.

## Test Components

### 1. Hull Manipulation Functions (5 Functions)

#### Function 1: Remove Hull Points and Recalculate
- **Description:** Removes all points on the original convex hull and recalculates a new hull from remaining points
- **Purpose:** Creates maximum differentiation from the original ROC curve
- **Parameters:** None (operates on all non-hull points)
- **Expected Outcome:** New hull with completely different points

#### Function 2: Select Closest Points to Hull
- **Description:** Selects the n points closest to the original convex hull
- **Purpose:** Find points just below the hull that might improve with small changes
- **Parameters:** n ∈ {5, 10, 20}
- **Method:** Uses KD-Tree for efficient nearest neighbor search
- **Expected Outcome:** Points clustered near the hull boundary

#### Function 3: Select Furthest Points from Diagonal
- **Description:** Selects the n points with maximum distance from the diagonal (TPR - FPR)
- **Purpose:** Find high-quality subgroups (best performers)
- **Parameters:** n ∈ {5, 10, 20}
- **Method:** Sorts by diagonal distance and selects top n
- **Expected Outcome:** Points with highest ROC quality values

#### Function 4: Select Points Below Hull (Percentage-based)
- **Description:** Selects points within a percentage threshold of maximum diagonal distance, measured by vertical distance below hull
- **Purpose:** Adaptive selection based on dataset-specific distance distribution
- **Parameters:** distance_percentage ∈ {5%, 10%, 20%}
- **Method:** 
  1. Find maximum diagonal distance from candidates (after hull exclusion)
  2. Calculate threshold = max_distance × (percentage / 100)
  3. Select points where vertical_distance_below_hull ≤ threshold
- **Expected Outcome:** Variable number of points based on distance distribution

#### Function 5: Select Points Above Diagonal (Percentage-based)
- **Description:** Selects points at or above a percentage threshold of maximum diagonal distance
- **Purpose:** Select high-quality points using adaptive threshold
- **Parameters:** distance_percentage ∈ {5%, 10%, 20%}
- **Method:**
  1. Find maximum diagonal distance from candidates (after hull exclusion)
  2. Calculate threshold = max_distance × ((100 - percentage) / 100)
  3. Select points where diagonal_distance ≥ threshold
- **Expected Outcome:** High-quality points meeting minimum performance threshold

### 2. True ROC Search
- **Description:** Original adaptive width ROC search algorithm
- **Purpose:** Baseline comparison for hull manipulation functions
- **Parameters:**
  - Alpha: 0.5 (balanced ROC quality)
  - Max depth: 3
  - Min coverage: 10 instances
- **Method:** Adaptive pruning based on convex hull membership
- **Expected Outcome:** Adaptive width based on ROC quality distribution

## Test Configuration

### Datasets
- **Priority Datasets:** ionosphere, wisconsin, mushroom, adult
- **Additional Datasets:** Credit-a, tic-tac-toe, covertype, YPMSD (if available)
- **Selection Criteria:** Variety of sizes and characteristics

### Search Parameters
- **Depth:** 3 (balance between comprehensiveness and computation time)
- **Minimum Coverage:** 10 instances
- **Alpha:** 0.5 for True ROC Search
- **Hull Exclusion:** Enabled for all functions to ensure curve differentiation

### Evaluation Metrics
1. **AUC (Area Under Curve):** Primary performance metric
2. **Number of Points:** Hull size after manipulation
3. **Computation Time:** Search efficiency
4. **Adaptive Width:** For True ROC Search

## Output Structure

```
runs/comprehensive_depth3/
├── COMPREHENSIVE_REPORT.md          # Main report with all results
├── ionosphere/
│   ├── summary.csv                  # Dataset summary statistics
│   └── ionosphere_hull_functions.png # Visualization of all functions
├── wisconsin/
│   ├── summary.csv
│   └── wisconsin_hull_functions.png
├── mushroom/
│   ├── summary.csv
│   └── mushroom_hull_functions.png
└── adult/
    ├── summary.csv
    └── adult_hull_functions.png
```

## Visualization

Each dataset gets a comprehensive visualization showing:
- **All ROC points** (light blue dots)
- **Original hull** (red line)
- **New hull** for each function (green line with markers)
- **Function name and metrics** in subplot title

Layout: 4 columns × n rows (to fit all function variants)

## Expected Results

### Function Behavior Patterns
1. **Function 1:** Largest AUC reduction (removes best points)
2. **Function 2:** Moderate AUC, close to original (near-hull points)
3. **Function 3:** High AUC (selects best performers)
4. **Function 4:** Variable, depends on distance distribution
5. **Function 5:** High AUC when sufficient points meet threshold

### Performance Comparisons
- **True ROC Search:** Should have highest/comparable AUC (keeps hull points)
- **Function 3 (Furthest):** Expected to have 2nd highest AUC
- **Function 2 (Closest):** Expected to have moderate AUC
- **Function 1 (Remove Hull):** Expected to have lowest AUC

## Technical Details

### Hull Exclusion Logic
All functions use `exclude_hull_points=True` to ensure:
1. Original hull points are identified
2. Excluded from candidate selection
3. New hull is guaranteed to be different

### Distance Calculations
- **Diagonal Distance:** TPR - FPR (perpendicular to y=x line)
- **Hull Distance:** Vertical distance from point to hull at same FPR
- **Euclidean Distance:** For nearest neighbor (Function 2)

### Edge Cases Handled
1. **Insufficient points:** Returns empty array if < 3 points
2. **All points on hull:** Detected and handled gracefully
3. **Threshold too strict:** Function 5 may return 0 points (expected behavior)

## Report Contents

The generated markdown report includes:
1. **Executive Summary:** Overview of analysis
2. **Methodology:** Detailed description of approach
3. **Dataset Results:** Individual results for each dataset
   - Data shape and target
   - True ROC Search results
   - Hull function results table
   - Visualization
4. **Consolidated Analysis:** Cross-dataset comparison
   - Summary table
   - Key findings

## Usage

### Running the Test
```bash
python quick_comprehensive_test.py
```

### Expected Runtime
- Small datasets (< 1000 rows): ~30-60 seconds each
- Medium datasets (1000-10000 rows): ~1-3 minutes each
- Large datasets (> 10000 rows): ~5-10 minutes each

### Output Files
- `COMPREHENSIVE_REPORT.md`: Main report
- `*/summary.csv`: Per-dataset metrics
- `*/*_hull_functions.png`: Visualizations

## Validation Criteria

### Success Indicators
- ✓ All datasets processed without errors
- ✓ All 5 functions produce valid results (or expected empty results)
- ✓ Visualizations show clear differentiation between hulls
- ✓ AUC values are reasonable (0 < AUC < 1)
- ✓ Point counts match expected ranges

### Quality Checks
1. **Function 1:** New hull has different points than original
2. **Functions 2-3:** Correct number of points selected (n)
3. **Functions 4-5:** Points meet threshold criteria
4. **True ROC:** Adaptive width > 0
5. **All:** No duplicate points in new hull

## Troubleshooting

### Common Issues
1. **Unicode errors:** Set `PYTHONIOENCODING=utf-8`
2. **Memory errors:** Reduce depth or use smaller datasets
3. **JVM crashes:** Disable wide beam search (already done)
4. **Empty results:** Check if dataset has sufficient above-diagonal points

### Debug Mode
Add print statements in hull functions to see:
- Candidate pool size after hull exclusion
- Distance calculations
- Threshold values
- Selection criteria applied

## Future Enhancements

### Potential Additions
1. **Additional hull functions:** Other selection criteria
2. **Parameter sweeps:** More granular percentage steps
3. **Performance profiling:** Detailed timing analysis
4. **Statistical tests:** Significance of AUC differences
5. **Interactive visualizations:** HTML-based exploration

### Optimization Opportunities
1. **Parallel processing:** Process datasets concurrently
2. **Caching:** Store intermediate results
3. **Incremental depth:** Start at depth 1, increase gradually
4. **Early stopping:** Skip functions if too few points

---

*Document created: 2025-10-16*
*Last updated: 2025-10-16*
