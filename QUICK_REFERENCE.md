# Hull Comparison Quick Reference

## Quick Start

### 1. Run Test Script
```bash
python test_hull_comparison.py
```
Output: `./runs/hull_test/` with visualizations and statistics

### 2. Use with True ROC Search
```python
from true_roc_search import true_roc_search, load_data

data = load_data('./tests/adult.txt')
results = true_roc_search(data, 'target', alphas=[0.5], max_depth=3)

# Hull comparisons automatically saved to:
# ./runs/true_roc/alpha_0.5/hull_comparisons/
```

### 3. Standalone Usage
```python
from true_roc_search import demonstrate_hull_comparison
import numpy as np

points = np.random.rand(50, 2)
hull_data = demonstrate_hull_comparison(points, depth=1, output_dir='./results')
```

## Function Quick Reference

### `remove_hull_points_and_recalculate(points, return_details=False)`
```python
# Minimal usage
new_hull = remove_hull_points_and_recalculate(points)

# Detailed usage
hull_data = remove_hull_points_and_recalculate(points, return_details=True)
print(hull_data['reduction_percentage'])
```

### `plot_hull_comparison(hull_data, depth, output_path=None)`
```python
# Display plot
plot_hull_comparison(hull_data, depth=2)

# Save plot
plot_hull_comparison(hull_data, depth=2, 
                    output_path='./results/comparison.png',
                    title_suffix=' (My Dataset)')
```

### `demonstrate_hull_comparison(points, depth=1, output_dir=None)`
```python
# Complete demonstration with visualization
hull_data = demonstrate_hull_comparison(points, depth=1, output_dir='./results')
```

## Output Files

### From `true_roc_search()`
```
output_dir/alpha_X/hull_comparisons/
├── hull_comparison_summary.csv      # Statistics for all depths
├── hull_comparison_depth_1.png      # Depth 1 visualization
├── hull_comparison_depth_2.png      # Depth 2 visualization
└── hull_comparison_depth_3.png      # Depth 3 visualization
```

### From `test_hull_comparison.py`
```
./runs/hull_test/
├── hull_comparison_demo_depth_1.png
├── depth_1_comparison.png
├── depth_2_comparison.png
├── depth_3_comparison.png
└── area_reduction_plot.png
```

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `original_hull_points` | # points on original hull | More = diverse hull |
| `new_hull_points` | # points on new hull | More = good depth diversity |
| `original_hull_area` | Area under original hull | Larger = better ROC curve |
| `new_hull_area` | Area under new hull | Shows remaining potential |
| `hull_area_reduction` | Absolute area difference | Impact of removing hull |
| `reduction_percentage` | % area reduction | >50% = dominated by few points |

## Common Patterns

### Pattern 1: Check Hull Stability
```python
hull_data = remove_hull_points_and_recalculate(points, return_details=True)
stability = hull_data['new_hull_area'] / hull_data['original_hull_area']

if stability > 0.8:
    print("High diversity - many good subgroups")
elif stability > 0.5:
    print("Moderate diversity")
else:
    print("Low diversity - dominated by top performers")
```

### Pattern 2: Track Depth Evolution
```python
results = true_roc_search(data, 'target', alphas=[0.5], max_depth=5)

print("Depth | Hull Points | Area Reduction")
print("-" * 40)
for hull_data in results[0.5]['hull_comparisons']:
    depth = hull_data['depth']
    n_hull = len(hull_data['original_hull'])
    reduction = hull_data.get('reduction_percentage', 0)
    print(f"{depth:5d} | {n_hull:11d} | {reduction:13.1f}%")
```

### Pattern 3: Compare Algorithms
```python
results = true_roc_search(data, 'target', alphas=[0.3, 0.5, 0.7])

for alpha in [0.3, 0.5, 0.7]:
    avg_reduction = np.mean([
        hd.get('reduction_percentage', 0) 
        for hd in results[alpha]['hull_comparisons']
    ])
    print(f"Alpha {alpha}: Avg reduction = {avg_reduction:.1f}%")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No new hull points | Increase `min_coverage` or check point distribution |
| Very small reduction | Normal for well-distributed points |
| Hull calculation error | Need at least 3 points above diagonal |
| Missing visualizations | Check output directory exists |

## Command Line Usage

```bash
# Pure ROC search with hull comparison
python true_roc_search.py --data ./tests/adult.txt --target target --pure-roc --depth 3

# Multiple alphas
python true_roc_search.py --data ./tests/adult.txt --target target --alphas 0.3 0.5 0.7 --depth 3

# Batch mode (all datasets)
python true_roc_search.py --batch --data-dir ./tests --depth 3
```

## Visualization Guide

### Three-Panel Plot
```
┌─────────────┬─────────────┬─────────────┐
│ Original    │ New Hull    │ Comparison  │
│ Hull        │ (After      │ (Both       │
│             │ Removal)    │ Overlaid)   │
└─────────────┴─────────────┴─────────────┘
```

**Red** = Original hull  
**Purple** = New hull  
**Blue** = All points  
**Green** = Remaining points  
**Black dashed** = Diagonal (random classifier)

## Advanced Usage

### Custom Point Analysis
```python
from true_roc_search import remove_hull_points_and_recalculate, plot_hull_comparison

# Your custom points (FPR, TPR)
custom_points = np.array([
    [0.1, 0.3],
    [0.2, 0.6],
    [0.3, 0.7],
    [0.4, 0.85],
    [0.6, 0.9]
])

# Analyze
hull_data = remove_hull_points_and_recalculate(custom_points, return_details=True)
hull_data['depth'] = 'custom'

# Visualize
plot_hull_comparison(hull_data, 'custom', './my_analysis.png')

# Print summary
print(f"Original hull: {len(hull_data['original_hull'])} points")
print(f"New hull: {len(hull_data.get('new_hull', []))} points")
print(f"Area reduction: {hull_data.get('reduction_percentage', 0):.1f}%")
```

## Integration Checklist

- ✅ Import functions from `true_roc_search`
- ✅ Prepare points as numpy array (n, 2) with [FPR, TPR]
- ✅ Ensure points above diagonal (TPR > FPR)
- ✅ Use `return_details=True` for analysis
- ✅ Create output directory before saving
- ✅ Check for at least 3 points for hull calculation

## Additional Resources

- Full documentation: `HULL_COMPARISON_GUIDE.md`
- Implementation details: `IMPLEMENTATION_SUMMARY.md`
- Test examples: `test_hull_comparison.py`
- Main search code: `true_roc_search.py`
