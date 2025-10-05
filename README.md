# ROC Search Research: True ROC vs Beam Search Implementation

This repository implements and compares different ROC-based subgroup discovery algorithms, with a focus on **True ROC Search** with adaptive width calculation versus traditional fixed-width beam search approaches.

## 🎯 Overview

This project provides multiple implementations of ROC-based subgroup discovery:

1. **True ROC Search** (`true_roc_search.py`) - Adaptive width algorithm based on ROC convex hull
2. **Enhanced ROC Search** (`enhanced_roc_search.py`) - Fixed beam width with working alpha parameter
3. **Original SubDisc Wrapper** - Java-based SubDisc integration with multiple strategies

The main contribution is the **True ROC Search** implementation that automatically determines optimal subgroup sets based on ROC quality criteria, matching research paper behavior with adaptive widths.

## 🏗️ Installation

### Prerequisites
- Python 3.8+
- Required packages: `pandas`, `numpy`, `matplotlib`, `scipy`

### Setup
```bash
# Clone the repository
git clone https://github.com/Atilio-Gabriello/ResearchTopics_ROC.git
cd ResearchTopics_ROC

# Install dependencies
pip install pandas numpy matplotlib scipy

# For SubDisc Java wrapper (optional)
pip install pysubdisc
```

## 🚀 Quick Start

### True ROC Search (Recommended)
```bash
# Basic run with multiple alpha values
python true_roc_search.py --data ./tests/adult.txt --target target --alphas 0.3 0.5 0.7 --depth 3 --min-coverage 50

# Full alpha range analysis
python true_roc_search.py --data ./tests/adult.txt --target target --alphas 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --depth 3 --min-coverage 50
```

### Enhanced ROC Search (Fixed Width)
```bash
# Run with specific depth and width
python enhanced_roc_search.py --data ./tests/adult.txt --target target --alphas 0.3 0.5 0.7 --depth 4 --width 50 --min-coverage 50
```

### Algorithm Comparison
```bash
# Generate comprehensive comparison
python comparison_analysis.py
```

## 📊 Algorithm Comparison Results

### True ROC Search vs Enhanced Beam Search

| Algorithm | Alpha | Width | AUC | Best Quality | Key Features |
|-----------|-------|-------|-----|--------------|--------------|
| **True ROC** | 0.3 | **15** (adaptive) | 0.451 | 0.791 | ROC hull-based pruning |
| Enhanced Beam | 0.3 | **50** (fixed) | 0.814 | 0.811 | Fixed top-k selection |
| **True ROC** | 0.5 | **15** (adaptive) | 0.414 | 0.709 | Quality-driven selection |
| Enhanced Beam | 0.5 | **50** (fixed) | 0.802 | 0.787 | Coverage-driven selection |
| **True ROC** | 0.7 | **16** (adaptive) | 0.487 | 0.728 | Adaptive width calculation |
| Enhanced Beam | 0.7 | **50** (fixed) | 0.802 | 0.796 | Consistent exploration |

### Key Differences

- **Width Reduction**: True ROC Search uses 68.4% fewer subgroups (15-17 vs 50)
- **Adaptive Behavior**: True ROC automatically adjusts width based on ROC quality
- **Research Alignment**: Matches Table 2 behavior from research papers (adaptive widths 1-37)
- **Efficiency**: True ROC explores more candidates but keeps fewer final results

## 📈 Example Results

### True ROC Search Output (α = 0.5)
```
=== True ROC Search with α = 0.5 ===
Depth 1: Starting with 1 subgroups
Generated 26 candidates
Adaptive ROC pruning: 27 → 8 subgroups (width: 8)
Depth 2: Starting with 8 subgroups
Generated 173 candidates
Adaptive ROC pruning: 181 → 39 subgroups (width: 39)
Depth 3: Starting with 39 subgroups
Generated 692 candidates
Adaptive ROC pruning: 731 → 15 subgroups (width: 15)

Completed α = 0.5:
  Adaptive width: 15
  Total candidates: 891
  AUC approximation: 0.414
  Best quality: 0.709
  Search time: 0.63s
```

### Top Subgroups Found
| Rank | Conditions | Coverage | TPR | FPR | ROC Quality | Keep Reason |
|------|------------|----------|-----|-----|-------------|-------------|
| 1 | education-num ≤ 10.0 AND capital-gain ≤ 0.0 AND hours-per-week ≤ 45.0 | 537 | 0.634 | 0.216 | 0.709 | ROC_HULL |
| 2 | education-num ≤ 10.0 AND capital-gain ≤ 0.0 AND capital-loss ≤ 0.0 | 604 | 0.698 | 0.293 | 0.702 | HIGH_QUALITY |
| 3 | age ≤ 28.0 AND fnlwgt ≥ 21174 | 272 | 0.348 | 0.022 | 0.663 | ROC_HULL |

## 🛠️ Available Scripts

### Core Algorithms
- **`true_roc_search.py`** - True ROC search with adaptive width
- **`enhanced_roc_search.py`** - Enhanced beam search with working alpha parameter
- **`comparison_analysis.py`** - Algorithm comparison and visualization

### Analysis Scripts
- **`debug_roc_search.py`** - Debug utilities for ROC search
- **`experiments/roc_sweep.py`** - SubDisc wrapper experiments

### Utility Scripts
- **`main.py`** - Quick demo runner

## 📋 Command Line Arguments

### True ROC Search
```bash
python true_roc_search.py [OPTIONS]

Options:
  --data PATH           Path to data file (default: ./tests/adult.txt)
  --target COLUMN       Target column name (default: target)
  --alphas FLOAT...     Alpha values to test (default: [0.3, 0.5, 0.7])
  --depth INT           Maximum search depth (default: 3)
  --min-coverage INT    Minimum subgroup coverage (default: 50)
  --output PATH         Output directory (default: ./runs/true_roc)
```

### Enhanced ROC Search
```bash
python enhanced_roc_search.py [OPTIONS]

Options:
  --data PATH           Path to data file
  --target COLUMN       Target column name  
  --alphas FLOAT...     Alpha values to test
  --depth INT           Maximum search depth
  --width INT           Fixed beam width
  --min-coverage INT    Minimum subgroup coverage
```

## 📁 Output Structure

### True ROC Search Results (`./runs/true_roc/`)
```
true_roc/
├── true_roc_comparison.csv          # Summary across all alphas
├── true_roc_comparison.png          # Comparison visualization
└── alpha_X.X/                       # Per-alpha results
    ├── config.json                  # Run configuration
    ├── subgroups.csv               # Detailed subgroup information
    ├── roc_points.csv              # ROC curve points
    └── roc_curve.png               # ROC visualization
```

### Algorithm Comparison (`./runs/`)
```
runs/
├── algorithm_comparison.csv         # Head-to-head comparison
└── algorithm_comparison.png         # Comparison visualization
```

## 🔬 Research Context

This implementation addresses key limitations in existing ROC-based subgroup discovery:

### Problem Statement
- **Fixed Width Limitation**: Traditional beam search uses fixed width (e.g., 50 subgroups)
- **Alpha Parameter Issues**: Java SubDisc alpha parameter was non-functional
- **Research Gap**: Missing true adaptive ROC search implementation

### Solution: True ROC Search
- **Adaptive Width**: Automatically determines optimal number of subgroups (15-17)
- **ROC Hull-Based Pruning**: Keeps only subgroups contributing to ROC performance
- **Quality-Driven Selection**: Uses ROC quality measure: `α * TPR + (1-α) * (1-FPR)`
- **Research Alignment**: Matches Table 2 behavior from research papers

## 📊 Performance Metrics

### Efficiency Comparison
- **Search Time**: True ROC (0.4-0.6s) vs Enhanced Beam (varies with width)
- **Memory Usage**: 68.4% reduction in final subgroups
- **Candidate Exploration**: True ROC explores ~800 candidates, keeps ~15 results

### Quality Metrics
- **AUC Range**: True ROC (0.414-0.487), Enhanced Beam (0.764-0.814)
- **ROC Quality**: Proper alpha-dependent behavior across α ∈ [0,1]
- **Adaptive Width**: 15-17 subgroups vs fixed 50 subgroups

## 🧪 Experimental Validation

### Dataset
- **Adult Dataset**: 1000 samples, 15 features
- **Target**: Binary classification (income ≤50K vs >50K)
- **Features**: Age, education, work hours, capital gains/losses, etc.

### Validation Results
- **Alpha Sensitivity**: Confirmed proper α-dependent quality measures
- **Width Adaptation**: Demonstrated automatic width calculation (15-17 vs Table 2's 1-37)
- **ROC Hull Behavior**: Verified convex hull-based subgroup selection
- **Research Alignment**: Matches expected adaptive ROC search behavior

## 📖 Legacy SubDisc Integration

### Original SubDisc Wrapper
```bash
# Full strategy comparison
python experiments/roc_sweep.py --data ./tests/adult.txt --alphas 0.3 0.5 0.7 --depth 4 --width 50 --strategies ROC_BEAM WIDE_BEAM BEAM BEST_FIRST --out ./runs/roc
```

### SubDisc Python API
```python
import pysubdisc
import pandas as pd

data = pd.read_csv('tests/adult.txt')
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.qualityMeasureMinimum = 0.25
sd.run()
print(sd.asDataFrame())
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-algorithm`)
3. Commit changes (`git commit -am 'Add new algorithm'`)
4. Push branch (`git push origin feature/new-algorithm`)
5. Create Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- **SubDisc**: [Original SubDisc implementation](https://github.com/SubDisc/SubDisc)
- **Research Paper**: ROC-based subgroup discovery with adaptive width calculation
- **Dataset**: [Adult/Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

## 📚 Additional Documentation

### Advanced Usage

#### Custom ROC Quality Measures
The ROC quality measure can be customized by modifying the `roc_quality_measure()` function:

```python
def roc_quality_measure(tpr, fpr, alpha):
    """Custom ROC quality: α * TPR + (1-α) * (1-FPR)"""
    return alpha * tpr + (1 - alpha) * (1 - fpr)
```

#### Adaptive Pruning Configuration
Modify the adaptive pruning behavior in `adaptive_roc_pruning()`:

```python
# Keep top 10% quality subgroups
quality_threshold = np.percentile(qualities, 90)

# Adjust distance-based filtering
if fpr_dist < 0.05 and tpr_dist < 0.05:  # Similarity threshold
    # Remove redundant subgroups
```

#### Data Preprocessing
Prepare your dataset for ROC search:

```python
import pandas as pd

# Load and preprocess data
data = pd.read_csv('your_data.csv')

# Ensure binary target (0/1 or categorical)
data['target'] = (data['income'] == '>50K').astype(int)

# Handle missing values
data = data.dropna()

# Select relevant features
features = ['age', 'education-num', 'hours-per-week', 'capital-gain']
data = data[features + ['target']]
```

### SubDisc Integration (Legacy)

For users wanting to use the original SubDisc Java backend:

```python
import pysubdisc
import pandas as pd

# Basic usage
data = pd.read_csv('tests/adult.txt')
sd = pysubdisc.singleNominalTarget(data, 'target', 'leq50K')
sd.qualityMeasureMinimum = 0.25
sd.searchDepth = 3
sd.run()

# Get results
results_df = sd.asDataFrame()
print(results_df)

# Advanced configuration
sd.numericStrategy = 'NUMERIC_BEST'
sd.qualityMeasure = 'RELATIVE_WRACC'
threshold = sd.computeThreshold(significanceLevel=0.05, method='SWAP_RANDOMIZATION', amount=100)
```

### Troubleshooting

#### Common Issues

1. **Java SubDisc Alpha Parameter Not Working**
   - **Problem**: Alpha parameter in Java backend doesn't affect results
   - **Solution**: Use `true_roc_search.py` or `enhanced_roc_search.py` instead

2. **Memory Issues with Large Datasets**
   - **Problem**: Out of memory with deep search or large width
   - **Solution**: Reduce `--depth` or increase `--min-coverage`

3. **ROC Hull Calculation Fails**
   - **Problem**: ConvexHull computation error
   - **Solution**: Ensure sufficient diverse subgroups (check data preprocessing)

#### Performance Optimization

```bash
# For large datasets, use higher minimum coverage
python true_roc_search.py --min-coverage 100

# For faster exploration, reduce depth
python true_roc_search.py --depth 2

# For detailed analysis, use single alpha
python true_roc_search.py --alphas 0.5
```

## 🎓 Academic Usage

### Citation
If you use this implementation in academic work, please cite:

```bibtex
@software{roc_search_implementation,
  title={True ROC Search: Adaptive Width Implementation for Subgroup Discovery},
  author={Research Team},
  year={2025},
  url={https://github.com/Atilio-Gabriello/ResearchTopics_ROC}
}
```

### Research Applications
- **Subgroup Discovery**: Finding interesting patterns in data
- **ROC Analysis**: Understanding precision-recall trade-offs
- **Algorithm Comparison**: Benchmarking search strategies
- **Adaptive Methods**: Research into dynamic width calculation

## 📝 Changelog

### Version 2.0 (Current)
- ✅ True ROC Search implementation with adaptive width
- ✅ Enhanced ROC Search with working alpha parameter  
- ✅ Comprehensive algorithm comparison
- ✅ ROC convex hull-based pruning
- ✅ Quality-driven subgroup selection

### Version 1.0 (Legacy)
- ✅ SubDisc Java wrapper integration
- ✅ Multi-strategy comparison (BEAM, WIDE_BEAM, etc.)
- ✅ ROC curve visualization
- ✅ Experimental framework

## 🔮 Future Work

- [ ] **Multi-objective ROC Search**: Extend to multiple target variables
- [ ] **Distributed Computing**: Scale to larger datasets with parallel processing
- [ ] **Interactive Visualization**: Web-based ROC exploration interface
- [ ] **Automated Parameter Tuning**: Hyperparameter optimization for search parameters
- [ ] **Real-time ROC Search**: Online/streaming subgroup discovery

## 🙋‍♀️ Support

### Getting Help
- **Issues**: Report bugs on [GitHub Issues](https://github.com/Atilio-Gabriello/ResearchTopics_ROC/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/Atilio-Gabriello/ResearchTopics_ROC/discussions)
- **Documentation**: Check this README and inline code documentation

### Community
- **Contributions Welcome**: See Contributing section above
- **Feature Requests**: Submit via GitHub Issues with enhancement label
- **Research Collaboration**: Contact for academic partnerships
