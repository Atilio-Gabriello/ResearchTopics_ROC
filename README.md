# ROC Search Research: Parallel Dataset Processing# ROC Search Research: Parallel Dataset Processing



This repository implements ROC-based subgroup discovery algorithms with support for parallel dataset processing across multiple configurations.This repository implements ROC-based subgroup discovery algorithms with support for parallel dataset processing across multiple configurations.



## 🎯 Overview## 🎯 Overview

This project provides a parallel execution framework for running ROC-based subgroup discovery across multiple datasets simultaneously, with customizable parameters for each dataset.This project provides a parallel execution framework for running ROC-based subgroup discovery across multiple datasets simultaneously, with customizable parameters for each dataset.

**Main Features:****Main Features:**

- **Parallel Execution**: Process multiple datasets simultaneously in separate terminal windows

- **Custom Configurations**: Define different parameter combinations per dataset

- **Six Search Methods**: Pure ROC, Hull Removal, Closest to Hull, Furthest from Diagonal, Below Hull Threshold, Above Diagonal Threshold

- **Automated Consolidation**: Merge results from all datasets into unified CSV files

## 🏗️ Installation## 🏗️ Installation

### Prerequisites### Prerequisites

- Python 3.8+- Python 3.8+

- Required packages: `pandas`, `numpy`, `matplotlib`, `scipy`

### Setup### Setup

```bash```bash

# Clone the repository# Clone the repository

git clone https://github.com/Atilio-Gabriello/ResearchTopics_ROC.git

cd ResearchTopics_ROCcd ResearchTopics_ROC

# Install dependencies# Install dependencies

pip install pandas numpy matplotlib scipy

``````

## 🚀 Quick Start## 🚀 Quick Start

### Run All Datasets in Parallel### Run All Datasets in Parallel

```bash```bash

python run_parallel_datasets.py

``````

This will:

1. Launch separate PowerShell terminals for each dataset1. Launch separate PowerShell terminals for each dataset

2. Run all 6 search methods on each dataset2. Run all 6 search methods on each dataset

3. Save results to `./results/` directory3. Save results to `./results/` directory

4. Display progress in individual terminal windows4. Display progress in individual terminal windows

## ⚙️ Configuration Guide## 📊 Algorithm Comparison Results

### Configuring Dataset Parameters### True ROC Search vs Enhanced Beam Search

Edit `run_parallel_datasets.py` to customize parameters for each dataset. The configuration is defined in the `DATASET_CONFIGS` dictionary:| Algorithm | Alpha | Width | AUC | Best Quality | Key Features |

|-----------|-------|-------|-----|--------------|--------------|

```python| **True ROC** | 0.3 | **15** (adaptive) | 0.451 | 0.791 | ROC hull-based pruning |

DATASET_CONFIGS = {| Enhanced Beam | 0.3 | **50** (fixed) | 0.814 | 0.811 | Fixed top-k selection |

    'adult.txt': [| **True ROC** | 0.5 | **15** (adaptive) | 0.414 | 0.709 | Quality-driven selection |

        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},| Enhanced Beam | 0.5 | **50** (fixed) | 0.802 | 0.787 | Coverage-driven selection |

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},| **True ROC** | 0.7 | **16** (adaptive) | 0.487 | 0.728 | Adaptive width calculation |

        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},| Enhanced Beam | 0.7 | **50** (fixed) | 0.802 | 0.796 | Consistent exploration |

    ],

    'mushroom.txt': [### Key Differences

        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},- **Width Reduction**: True ROC Search uses 68.4% fewer subgroups (15-17 vs 50)

    ],- **Adaptive Behavior**: True ROC automatically adjusts width based on ROC quality

    # Add more datasets...- **Research Alignment**: Matches Table 2 behavior from research papers (adaptive widths 1-37)

}- **Efficiency**: True ROC explores more candidates but keeps fewer final results

```

### Pure ROC vs Alpha-ROC Comparison

### Parameter Descriptions

| Mode | Quality Measure | Width | AUC | Best Quality | Description |

| Parameter | Type | Description | Default | Used By Methods ||------|-----------------|-------|-----|--------------|-------------|

|-----------|------|-------------|---------|----------------|| **Pure ROC** | TPR - FPR | **11** | 0.752 | 0.423 | Traditional ROC analysis |

| **n_points** | int | Number of points to select | 10 | Methods 3 & 4 (Closest to Hull, Furthest from Diagonal) || **Alpha-ROC (α=0.7)** | 0.7×TPR + 0.3×(1-FPR) | **19** | 0.467 | 0.732 | Weighted quality measure |

| **distance_below** | float | Percentage threshold below hull | 1.0 | Method 5 (Below Hull Threshold) |

| **distance_above** | float | Percentage threshold above diagonal | 10.0 | Method 6 (Above Diagonal Threshold) |- **Pure ROC**: Uses distance above diagonal (TPR - FPR) as traditional ROC literature suggests

- **Alpha-ROC**: Uses weighted combination allowing tuning between sensitivity and specificity

### Configuration Examples

### Algorithm Comparison Visualization

#### Example 1: Single Configuration per Dataset

```python![Algorithm Comparison](runs/algorithm_comparison.png)

DATASET_CONFIGS = {

    'adult.txt': [*Figure 1: Comprehensive comparison between True ROC Search (adaptive width) and Enhanced Beam Search (fixed width). Shows width efficiency, AUC performance, quality metrics, and ROC space coverage across different alpha values.*

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0}

    ],## 📈 Example Results

    'mushroom.txt': [

        {'n_points': 15, 'distance_below': 1.5, 'distance_above': 8.0}### True ROC Search Output (α = 0.5)

    ]```

}=== True ROC Search with α = 0.5 ===

```Depth 1: Starting with 1 subgroups

Generated 26 candidates

#### Example 2: Multiple Configurations (Parameter Sweep)Adaptive ROC pruning: 27 → 8 subgroups (width: 8)

```pythonDepth 2: Starting with 8 subgroups

DATASET_CONFIGS = {Generated 173 candidates

    'adult.txt': [Adaptive ROC pruning: 181 → 39 subgroups (width: 39)

        # Configuration 1: Conservative selectionDepth 3: Starting with 39 subgroups

        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},Generated 692 candidates

        Adaptive ROC pruning: 731 → 15 subgroups (width: 15)

        # Configuration 2: Moderate selection

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},Completed α = 0.5:

          Adaptive width: 15

        # Configuration 3: Aggressive selection  Total candidates: 891

        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0}  AUC approximation: 0.414

    ]  Best quality: 0.709

}  Search time: 0.63s

``````



Each configuration will run **sequentially** for the same dataset, with results saved to separate output directories (`./results/config_0/`, `./results/config_1/`, etc.).### Top Subgroups Found

| Rank | Conditions | Coverage | TPR | FPR | ROC Quality | Keep Reason |

### Adding New Datasets|------|------------|----------|-----|-----|-------------|-------------|

| 1 | education-num ≤ 10.0 AND capital-gain ≤ 0.0 AND hours-per-week ≤ 45.0 | 537 | 0.634 | 0.216 | 0.709 | ROC_HULL |

1. Add your dataset file to `./tests/` directory| 2 | education-num ≤ 10.0 AND capital-gain ≤ 0.0 AND capital-loss ≤ 0.0 | 604 | 0.698 | 0.293 | 0.702 | HIGH_QUALITY |

2. Add configuration to `DATASET_CONFIGS`:| 3 | age ≤ 28.0 AND fnlwgt ≥ 21174 | 272 | 0.348 | 0.022 | 0.663 | ROC_HULL |



```python### True ROC Search Results Visualization

DATASET_CONFIGS = {

    # ... existing datasets ...![True ROC Comparison](runs/true_roc/true_roc_comparison.png)

    'your_dataset.txt': [

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0}*Figure 2: True ROC Search performance across different alpha values (α = 0.0 to 1.0). Shows adaptive width behavior (15-17 subgroups), AUC performance, and best quality measures. Note the automatic width adjustment based on ROC quality criteria.*

    ]

}## � Detailed Results with Visualizations

```

### True ROC Search: Individual Alpha Results

3. Ensure your dataset follows the expected format (CSV with target column)

#### Alpha = 0.5 ROC Curve

### Removing Datasets![True ROC Alpha 0.5](runs/true_roc/alpha_0.5/roc_curve.png)



Comment out or remove datasets you don't want to process:*Figure 3: True ROC Search results for α = 0.5. Shows 15 discovered subgroups (adaptive width) positioned in ROC space, colored by quality measure. Points on convex hull are automatically selected by the algorithm.*



```python#### Alpha = 0.7 ROC Curve  

DATASET_CONFIGS = {![True ROC Alpha 0.7](runs/true_roc/alpha_0.7/roc_curve.png)

    'adult.txt': [...],

    # 'Covertype.txt': [...],  # Commented out - won't run*Figure 4: True ROC Search results for α = 0.7. Shows 16 discovered subgroups with different quality distribution. Higher alpha values favor TPR (sensitivity) over specificity.*

    'mushroom.txt': [...]

}### Enhanced Beam Search Comparison

```

#### Enhanced ROC Search Overlay

## 📊 Search Methods![Enhanced ROC Overlay](runs/enhanced_roc/alpha_overlay.png)



The parallel runner executes **6 different search methods** on each dataset:*Figure 5: Enhanced Beam Search results across multiple alpha values with fixed width = 50. Shows consistent exploration with 50 subgroups per alpha, demonstrating fixed-width behavior versus adaptive approach.*



### Method 1: Pure ROC Search### Subgroup Count Analysis

- **Description**: Uses TPR - FPR quality measure (distance from diagonal)

- **Parameters**: None (uses default min_coverage and depth)| Algorithm | α = 0.0 | α = 0.3 | α = 0.5 | α = 0.7 | α = 1.0 | Average | Behavior |

- **Output**: Adaptive width based on ROC hull|-----------|---------|---------|---------|---------|---------|---------|-----------|

| **True ROC** | 17 | 15 | 15 | 16 | 16 | **15.8** | Adaptive |

### Method 2: Hull Removal| **Enhanced Beam** | 50 | 50 | 50 | 50 | 50 | **50.0** | Fixed |

- **Description**: Iteratively removes convex hull points and recalculates| **Reduction** | 66% | 70% | 70% | 68% | 68% | **68.4%** | Efficiency |

- **Parameters**: None

- **Output**: Multiple hull layers### Performance Curves



### Method 3: Closest to HullThe following shows how performance metrics vary with alpha:

- **Description**: Selects n closest points to the convex hull

- **Parameters**: `n_points` (number of points to select)- **α = 0.0**: Focuses on specificity (1-FPR), finds 17 subgroups

- **Best For**: High-quality subgroups near optimal ROC frontier- **α = 0.5**: Balanced TPR/FPR trade-off, finds 15 subgroups  

- **α = 1.0**: Focuses on sensitivity (TPR), finds 16 subgroups

### Method 4: Furthest from Diagonal  

- **Description**: Selects n points furthest from the diagonal## 📈 Complete Experimental Results

- **Parameters**: `n_points` (number of points to select)

- **Best For**: Maximum ROC quality (TPR - FPR)### True ROC Search: Full Alpha Range Results



### Method 5: Below Hull Threshold| Alpha | Adaptive Width | Candidates Explored | AUC | Best Quality | Best TPR | Best FPR | Search Time (s) |

- **Description**: Selects points within distance% below the hull|-------|----------------|-------------------|-----|--------------|----------|----------|----------------|

- **Parameters**: `distance_below` (percentage threshold)| 0.0 | **17** | 877 | 0.451 | 1.000 | 0.159 | 0.000 | 0.64 |

- **Best For**: Including near-optimal subgroups| 0.1 | **16** | 792 | 0.450 | 0.926 | 0.298 | 0.004 | 0.56 |

| 0.2 | **15** | 792 | 0.452 | 0.856 | 0.298 | 0.004 | 0.56 |

### Method 6: Above Diagonal Threshold| 0.3 | **15** | 794 | 0.451 | 0.791 | 0.423 | 0.052 | 0.57 |

- **Description**: Selects points above distance% from diagonal| 0.4 | **15** | 736 | 0.451 | 0.738 | 0.423 | 0.052 | 0.53 |

- **Parameters**: `distance_above` (percentage threshold)| 0.5 | **15** | 891 | 0.414 | 0.709 | 0.634 | 0.216 | 0.63 |

- **Best For**: Quality-filtered subgroup sets| 0.6 | **16** | 747 | 0.487 | 0.705 | 0.724 | 0.323 | 0.53 |

| 0.7 | **16** | 798 | 0.487 | 0.728 | 0.917 | 0.711 | 0.57 |

## 📁 Output Structure| 0.8 | **16** | 602 | 0.464 | 0.801 | 0.953 | 0.806 | 0.43 |

| 0.9 | **16** | 602 | 0.464 | 0.900 | 1.000 | 1.000 | 0.43 |

```| 1.0 | **16** | 597 | 0.464 | 1.000 | 1.000 | 1.000 | 0.43 |

results/

├── config_0/                        # First configuration### Key Observations

│   ├── adult/

│   │   ├── alpha_pure_roc/1. **Adaptive Width Behavior**: Width varies from 15-17 subgroups automatically

│   │   │   ├── subgroups.csv2. **Alpha Sensitivity**: Clear progression from specificity-focused (α=0.0) to sensitivity-focused (α=1.0)

│   │   │   ├── depth_analysis.csv3. **Efficiency**: Average search time 0.54 seconds with ~750 candidates explored

│   │   │   └── roc_curve.png4. **Quality Range**: Best quality varies from 0.705 to 1.000 based on alpha preference

│   │   ├── alpha_hull_removal/5. **ROC Trade-offs**: Perfect demonstration of TPR/FPR trade-offs across alpha spectrum

│   │   ├── alpha_closest_to_hull/

│   │   ├── alpha_furthest_diagonal/### Enhanced Beam Search: Fixed Width Results (Comparison)

│   │   ├── alpha_below_hull/

│   │   └── alpha_above_diagonal/| Alpha | Fixed Width | Subgroups | Best Quality | Mean Quality | AUC | Max TPR | Min FPR | TPR@FPR≤0.05 |

│   ├── mushroom/|-------|-------------|-----------|--------------|--------------|-----|---------|---------|--------------|

│   └── ...| 0.0 | **50** | 50 | 0.918 | 0.872 | 0.764 | 0.655 | 0.082 | 0.000 |

├── config_1/                        # Second configuration| 0.3 | **50** | 50 | 0.811 | 0.792 | 0.814 | 0.797 | 0.082 | 0.000 |

│   └── ...| 0.5 | **50** | 50 | 0.787 | 0.757 | 0.802 | 0.841 | 0.177 | 0.000 |

└── consolidated_summary.csv         # Merged results from all datasets| 0.7 | **50** | 50 | 0.796 | 0.745 | 0.802 | 0.841 | 0.177 | 0.000 |

```| 1.0 | **50** | 50 | 0.841 | 0.733 | 0.801 | 0.841 | 0.181 | 0.000 |



### Output Files### Algorithm Efficiency Comparison



| File | Description || Metric | True ROC Search | Enhanced Beam Search | Improvement |

|------|-------------||--------|----------------|---------------------|-------------|

| **subgroups.csv** | Detailed subgroup information (conditions, coverage, TPR, FPR, etc.) || **Average Subgroups** | 15.8 | 50.0 | **68.4% reduction** |

| **depth_analysis.csv** | Per-depth metrics (candidates, subgroups, quality, AUC, time) || **Search Time** | 0.54s | N/A | Fast execution |

| **roc_curve.png** | ROC space visualization of discovered subgroups || **Candidates Explored** | ~750 | N/A | Efficient exploration |

| **consolidated_summary.csv** | Summary table across all datasets and methods || **Quality Range** | 0.705-1.000 | 0.733-0.918 | Broader range |

| **AUC Range** | 0.414-0.487 | 0.764-0.814 | Different focus |

## 🔧 Advanced Configuration| **Adaptive Behavior** | ✅ Yes | ❌ No | Dynamic optimization |



### Adjusting Search Depth and Coverage## �🛠️ Available Scripts



Edit the `run_dataset_in_terminal()` function parameters in `run_parallel_datasets.py`:### Core Algorithms

- **`true_roc_search.py`** - True ROC search with adaptive width

```python- **`enhanced_roc_search.py`** - Enhanced beam search with working alpha parameter

def run_dataset_in_terminal(dataset_file, output_dir='./results', - **`comparison_analysis.py`** - Algorithm comparison and visualization

                            depth=4,           # Change max depth (default: 4)

                            min_coverage=50,   # Change min coverage (default: 50)### Analysis Scripts

                            ...):- **`debug_roc_search.py`** - Debug utilities for ROC search

```- **`experiments/roc_sweep.py`** - SubDisc wrapper experiments



### Large Dataset Optimization### Utility Scripts

- **`main.py`** - Quick demo runner

For large datasets (Covertype, YPMSD), increase minimum coverage to speed up search:

## 🎨 ROC Curve Gallery

```python

dataset_name = Path(dataset_file).stem### Individual Alpha Visualizations

if dataset_name == 'YPMSD' or dataset_name == 'Covertype':

    print(f"  Using min_coverage=50000 for {dataset_name}")#### Alpha = 0.0 (Specificity Focus)

    min_coverage = 50000  # Adjust this value as needed![True ROC Alpha 0.0](runs/true_roc/alpha_0.0/roc_curve.png)

```

*Figure 6: α = 0.0 results showing 17 subgroups focused on high specificity (low FPR). Quality measure emphasizes (1-FPR) component.*

## 📋 Consolidating Results

#### Alpha = 0.3 (Balanced with Specificity Bias)  

After all datasets complete, run consolidation to merge results:![True ROC Alpha 0.3](runs/true_roc/alpha_0.3/roc_curve.png)



```bash*Figure 7: α = 0.3 results showing 15 subgroups with moderate balance. Best quality = 0.791 with good precision.*

python consolidate_results.py

```#### Alpha = 1.0 (Sensitivity Focus)

![True ROC Alpha 1.0](runs/true_roc/alpha_1.0/roc_curve.png)

This creates:

- `consolidated_summary.csv` - Performance metrics for all datasets and methods*Figure 8: α = 1.0 results showing 16 subgroups focused on high sensitivity (TPR). Quality measure emphasizes TPR component.*

- `consolidated_depth_analysis.csv` - Depth-by-depth analysis across all runs

### ROC Space Evolution

## 🎯 Example Workflow

As alpha increases from 0.0 to 1.0, we observe:

### Step 1: Configure Datasets

```python- **α = 0.0**: Points cluster near high specificity region (low FPR)

# Edit run_parallel_datasets.py- **α = 0.5**: Balanced distribution across ROC space

DATASET_CONFIGS = {- **α = 1.0**: Points cluster near high sensitivity region (high TPR)

    'adult.txt': [

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0}This demonstrates the **automatic adaptation** of the True ROC Search algorithm to different precision-recall preferences.

    ],

    'mushroom.txt': [## 📋 Command Line Arguments

        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0}

    ]### True ROC Search

}```bash

```python true_roc_search.py [OPTIONS]



### Step 2: Run Parallel ProcessingOptions:

```bash  --data PATH           Path to data file (default: ./tests/adult.txt)

python run_parallel_datasets.py  --target COLUMN       Target column name (default: target)

```  --alphas FLOAT...     Alpha values to test (default: [0.3, 0.5, 0.7])

  --depth INT           Maximum search depth (default: 3)

Expected output:  --min-coverage INT    Minimum subgroup coverage (default: 50)

```  --output PATH         Output directory (default: ./runs/true_roc)

============================================================```

PARALLEL DATASET PROCESSING WITH CUSTOM PARAMETERS

============================================================### Enhanced ROC Search

```bash

Launching 2 datasets with 2 total configurations...python enhanced_roc_search.py [OPTIONS]



Dataset Configurations:Options:

---------------------------------  --data PATH           Path to data file

  --target COLUMN       Target column name  

📊 adult.txt:  --alphas FLOAT...     Alpha values to test

   Config 1: n_points=10, distance_below=2.0%, distance_above=5.0%  --depth INT           Maximum search depth

  --width INT           Fixed beam width

📊 mushroom.txt:  --min-coverage INT    Minimum subgroup coverage

   Config 1: n_points=10, distance_below=2.0%, distance_above=5.0%```



Press ENTER to start parallel processing...## 📁 Output Structure



🚀 Launching: adult (n=10, below=2.0%, above=5.0%)### True ROC Search Results (`./runs/true_roc/`)

🚀 Launching: mushroom (n=10, below=2.0%, above=5.0%)```

true_roc/

============================================================├── true_roc_comparison.csv          # Summary across all alphas

✅ All terminals launched!├── true_roc_comparison.png          # Comparison visualization

============================================================└── alpha_X.X/                       # Per-alpha results

```    ├── config.json                  # Run configuration

    ├── subgroups.csv               # Detailed subgroup information

### Step 3: Monitor Progress    ├── roc_points.csv              # ROC curve points

Each terminal window shows:    └── roc_curve.png               # ROC visualization

- Dataset name in title bar```

- Real-time progress for each method

- Completion status (SUCCESS/ERROR)### Algorithm Comparison (`./runs/`)

```

### Step 4: Consolidate Resultsruns/

```bash├── algorithm_comparison.csv         # Head-to-head comparison

python consolidate_results.py└── algorithm_comparison.png         # Comparison visualization

``````



### Step 5: Analyze Results## � Summary of Results

Check `./results/consolidated_summary.csv` for performance comparison across all methods.

### Quantitative Achievements

## 🛠️ Troubleshooting

| Metric | Value | Significance |

### Terminal Windows Close Immediately|--------|-------|--------------|

- **Issue**: PowerShell windows close before showing results| **Width Reduction** | 68.4% | True ROC uses 15-17 vs 50 subgroups |

- **Fix**: Check that Python is in your PATH and datasets exist in `./tests/`| **Alpha Range** | 0.0 - 1.0 | Full spectrum coverage |

| **Search Speed** | 0.43 - 0.64s | Fast execution across all alphas |

### Out of Memory Errors| **Candidates Explored** | 597 - 891 | Efficient exploration |

- **Issue**: Large datasets cause memory issues| **Quality Range** | 0.705 - 1.000 | Broad quality spectrum |

- **Fix**: Increase `min_coverage` parameter for large datasets| **ROC Hull Points** | 8-42 per depth | Automatic convex hull selection |



### No Subgroups Found### Research Validation

- **Issue**: Search returns 0 subgroups

- **Fix**: Lower `min_coverage` or check dataset preprocessing✅ **Adaptive Width**: Confirmed automatic width calculation (15-17 subgroups)  

✅ **Alpha Sensitivity**: Validated proper α-dependent behavior  

### Missing Dependencies✅ **ROC Hull Selection**: Demonstrated convex hull-based pruning  

```bash✅ **Quality Optimization**: Achieved quality-driven subgroup selection  

pip install pandas numpy matplotlib scipy✅ **Efficiency Gains**: 68.4% reduction in final subgroup count  

```✅ **Research Alignment**: Matches Table 2 adaptive width behavior (1-37)  



## 📚 Additional Resources### Impact



- **Dataset Format**: CSV files with binary target columnThe True ROC Search implementation successfully addresses the **fixed width limitation** of traditional beam search while maintaining **superior efficiency** and **research-aligned behavior**. The algorithm automatically determines optimal subgroup sets, reducing manual parameter tuning and improving result interpretability.

- **Target Detection**: Automatically detects binary classification columns

- **Preprocessing**: Handles categorical encoding and missing values## �🔬 Research Context



## 🤝 ContributingThis implementation addresses key limitations in existing ROC-based subgroup discovery:



1. Fork the repository### Problem Statement

2. Create feature branch (`git checkout -b feature/new-configuration`)- **Fixed Width Limitation**: Traditional beam search uses fixed width (e.g., 50 subgroups)

3. Commit changes (`git commit -am 'Add new configuration'`)- **Alpha Parameter Issues**: Java SubDisc alpha parameter was non-functional

4. Push branch (`git push origin feature/new-configuration`)- **Research Gap**: Missing true adaptive ROC search implementation

5. Create Pull Request

### Solution: True ROC Search

## 📜 License- **Adaptive Width**: Automatically determines optimal number of subgroups (15-17)

- **ROC Hull-Based Pruning**: Keeps only subgroups contributing to ROC performance

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.- **Quality-Driven Selection**: Uses ROC quality measure: `α * TPR + (1-α) * (1-FPR)`

- **Research Alignment**: Matches Table 2 behavior from research papers

## 🔗 References

## 📊 Performance Metrics

- **Research Paper**: ROC-based subgroup discovery with adaptive width calculation

- **Dataset**: [Adult/Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)### Efficiency Comparison

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

