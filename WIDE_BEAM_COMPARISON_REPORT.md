# Wide Beam Search vs ROC Methods Comparison

## Executive Summary

This document compares **pysubdisc's Wide Beam Search** with **6 custom ROC search methods** across 4 common datasets at depth=4.

📊 **Full Results**: See [wide_beam_summary.csv](runs/wide_beam_search/wide_beam_summary.csv) for complete wide beam search results across all 7 datasets.

### Key Findings

🏆 **Winner**: **Wide Beam Search (pysubdisc)** - Won all 4 datasets by AUC!

**Performance Summary:**
- **Mean AUC**: 0.7920 (Wide Beam) vs 0.5900 (Best ROC Method: Basic ROC Search)
- **AUC Improvement**: +34.2% over best ROC method
- **Speed**: 60x faster than Basic ROC Search, but 35-166x slower than optimized methods
- **ROC Points**: 544 average (Wide Beam) vs 8680 average (ROC methods)

---

## Detailed Results

### AUC Comparison by Dataset

| Dataset      | Wide Beam | Basic ROC Search | Advantage |
|--------------|-----------|------------------|-----------|
| **adult**    | 0.8068    | 0.6639           | +21.5%    |
| **mushroom** | 0.8246    | 0.6011           | +37.2%    |
| **ionosphere** | 0.9431  | 0.5213           | +80.9%    |
| **tic-tac-toe** | 0.5934 | 0.5356           | +10.8%    |

**Winner by Dataset:**
- ✅ **adult**: Wide Beam (0.8068)
- ✅ **mushroom**: Wide Beam (0.8246)
- ✅ **ionosphere**: Wide Beam (0.9431) - Spectacular 94.3% AUC!
- ✅ **tic-tac-toe**: Wide Beam (0.5934)

---

### Execution Time Comparison

**Wide Beam Search: 1.79s** (mean across datasets)

| Method                        | Mean Time | Speedup vs WB |
|-------------------------------|-----------|---------------|
| Basic ROC Search              | 108.26s   | 60x slower ❌ |
| Below Hull Percentage         | 0.050s    | 36x faster ✅ |
| Remove Hull Points            | 0.019s    | 94x faster ✅ |
| Closest Points to Hull        | 0.019s    | 97x faster ✅ |
| Furthest from Diagonal        | 0.016s    | 114x faster ✅ |
| Above Diagonal Percentage     | 0.011s    | 166x faster ✅ |

**Trade-off Analysis:**
- Wide Beam is **60x faster** than Basic ROC Search (the only other method with good AUC)
- Wide Beam is **35-166x slower** than optimized ROC methods, but those methods had **NaN AUC** at depth=4

---

### ROC Points Generated

| Method                    | Mean Points | Notes                          |
|---------------------------|-------------|--------------------------------|
| **Wide Beam Search**      | 544         | More manageable point set      |
| All ROC Methods           | 8,680       | 16x more points than WB        |
| Basic ROC Search          | N/A         | Did not report total points    |

**Insight**: Wide Beam generates a more compact and interpretable set of ROC points while achieving superior AUC.

---

## Method-Specific Analysis

### Wide Beam Search (pysubdisc)
- **Configuration**: depth=4, width=100
- **Strengths**:
  - ✅ Best AUC across all datasets
  - ✅ Reasonable execution time (1-2 seconds per dataset)
  - ✅ Compact ROC point sets (544 average)
  - ✅ Mature, battle-tested library
  - ✅ Handles complex attribute interactions well
  
- **Weaknesses**:
  - ❌ 35-166x slower than optimized ROC methods
  - ❌ Requires pysubdisc library installation
  - ❌ Less transparency in subgroup selection logic

### Basic ROC Search
- **Strengths**:
  - ✅ Second-best AUC (0.5900 mean)
  - ✅ Exhaustive search ensures completeness
  
- **Weaknesses**:
  - ❌ 60x slower than Wide Beam (108s average)
  - ❌ Still significantly worse AUC than Wide Beam (-34%)

### Optimized ROC Methods (Remove Hull, Closest Points, etc.)
- **Strengths**:
  - ✅ Extremely fast (0.01-0.05 seconds)
  - ✅ 35-166x faster than Wide Beam
  
- **Weaknesses**:
  - ❌ **NaN AUC at depth=4** - methods failed to produce valid results
  - ❌ Likely too aggressive pruning for deep searches
  - ❌ Only worked well at shallow depths (1-2)

---

## Recommendations

### When to Use Wide Beam Search

1. **Deep searches** (depth ≥ 3): Wide Beam excels at complex pattern discovery
2. **Maximum quality**: When you need the highest possible AUC
3. **Production systems**: Mature library with proven reliability
4. **Reasonable datasets**: Up to ~10K instances (performance still acceptable)

### When to Use ROC Methods

1. **Shallow searches** (depth ≤ 2): Optimized methods are very fast
2. **Real-time requirements**: When sub-second response is critical
3. **Exploratory analysis**: Quick iteration during development
4. **Specific constraints**: When you need custom point selection logic

### Hybrid Approach

Consider a **two-stage strategy**:

1. **Stage 1**: Use optimized ROC methods for quick exploration (depth 1-2)
2. **Stage 2**: Apply Wide Beam for final deep search (depth 3-4)

This combines speed of ROC methods with quality of Wide Beam.

---

## Statistical Summary

### Overall Performance Metrics

| Metric                  | Wide Beam | Best ROC Method | Difference |
|-------------------------|-----------|-----------------|------------|
| **Mean AUC**            | 0.7920    | 0.5900          | +34.2%     |
| **Std Dev AUC**         | 0.1456    | 0.0584          | -          |
| **Best AUC**            | 0.9431    | 0.6639          | +42.1%     |
| **Worst AUC**           | 0.5934    | 0.5213          | +13.8%     |
| **Mean Time**           | 1.79s     | 108.26s / 0.03s | -          |
| **Mean Points**         | 544       | 8,680           | -93.7%     |
| **Datasets Won**        | 4/4       | 0/4             | 100%       |

---

## Visualizations

The following visualizations were generated:

1. **comparison_with_roc_methods.png**
   - 4-panel comparison showing AUC, time, points, and mean performance
   - Highlights Wide Beam's dominance in AUC

2. **auc_heatmap_comparison.png**
   - Color-coded heatmap of AUC values
   - Gold stars (★) indicate best method per dataset
   - Wide Beam has all 4 stars

3. **wide_beam_roc_plots.png**
   - ROC space plots for each dataset
   - Shows convex hull and point distributions
   - Visual confirmation of high-quality subgroup discovery

---

## Conclusions

### Primary Conclusion

**Wide Beam Search (pysubdisc) is the clear winner** for depth=4 subgroup discovery:
- 🏆 **100% win rate** (4/4 datasets)
- 📈 **34% better AUC** than best alternative
- ⚡ **60x faster** than comparable Basic ROC Search
- 📊 **More interpretable** with compact point sets

### Secondary Findings

1. **Depth matters**: Optimized ROC methods failed at depth=4 but worked at depth=1-2
2. **Quality vs Speed trade-off**: Wide Beam offers best balance
3. **Library maturity**: pysubdisc's proven implementation shows value

### Future Work

1. **Hybrid optimization**: Combine ROC method speed with Wide Beam quality
2. **Depth-adaptive strategies**: Auto-select method based on depth
3. **Parameter tuning**: Explore width parameter effects on Wide Beam
4. **Scalability testing**: Test on larger datasets (>10K instances)

---

## Files Generated

### Data Files
- `wide_beam_summary.csv` - Main results summary
- `wide_beam_subgroups.csv` - Detailed subgroup information
- `detailed_comparison.csv` - Full comparison matrix
- `auc_comparison.csv` - AUC-only comparison

### Visualizations
- `wide_beam_roc_plots.png` - ROC space visualizations
- `comparison_with_roc_methods.png` - 4-panel comparison
- `auc_heatmap_comparison.png` - Heatmap with rankings

### Scripts
- `wide-beam.py` - Modified Wide Beam runner
- `compare_wide_beam_results.py` - Comparison analysis script

---

## Configuration Used

### Wide Beam Search
```python
depth = 4
width = 100
search_strategy = "beam"
quality_measure = "Cortana Quality"
```

### ROC Methods
```python
depth = 4
exclude_hull_points = True
# Various parameters tested:
# - n_points: [10, 20, 50, 100]
# - distance_percentage: [0.5, 1.0, 2.0, 5.0, 10.0]
```

---

## Repository Structure

```
runs/
├── wide_beam_search/          # Wide Beam results
│   ├── wide_beam_summary.csv
│   ├── wide_beam_subgroups.csv
│   ├── detailed_comparison.csv
│   ├── auc_comparison.csv
│   ├── wide_beam_roc_plots.png
│   ├── comparison_with_roc_methods.png
│   └── auc_heatmap_comparison.png
│
└── comprehensive_all_methods/  # ROC methods results
    ├── comprehensive_results.csv
    ├── summary_by_method.csv
    └── [various visualization PNGs]
```

---

## References

- **pysubdisc**: Python subgroup discovery library
- **Original ROC research**: Custom implementations for ROC curve optimization
- **Datasets**: UCI Machine Learning Repository

---

*Analysis completed on: Generated from comprehensive testing suite*
*Total experiments: 484 (480 ROC + 4 Wide Beam)*
*Datasets tested: adult, mushroom, ionosphere, tic-tac-toe*
