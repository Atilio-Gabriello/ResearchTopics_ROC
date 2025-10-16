# 🎯 Complete Guide: Multi-Parameter ROC Search System

## 📁 Files Created

1. **`run_all_methods_all_datasets.py`** - Main script (ENHANCED)
2. **`test_multi_params.py`** - Quick testing script
3. **`analyze_multi_param_results.py`** - Results analysis script
4. **`MULTI_PARAM_GUIDE.md`** - Detailed technical guide
5. **`QUICK_START_MULTI_PARAMS.md`** - Ready-to-use examples
6. **`MULTI_PARAM_SUMMARY.md`** - High-level overview
7. **`README_MULTI_PARAMS.md`** - This file

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Multi-Parameter Analysis
```bash
# Test multiple n-points and percentages on adult dataset
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 10 20 30 \
    --distance-percentage 1.0 2.0 5.0 \
    --datasets adult
```

### Step 2: Check Output
```bash
# Results saved to:
./runs/all_methods_comparison/
├── adult/
│   ├── adult_comparison.png          # Visual comparison
│   └── adult_results.json             # Detailed results
└── consolidated_results.csv           # All results in one table
```

### Step 3: Analyze Results
```bash
# Run the analysis script
python analyze_multi_param_results.py
```

---

## 🎨 What You Get

### 1. Multiple Parameter Testing
**Before (old way):**
```bash
# Had to run 3 separate times
python run_all_methods_all_datasets.py --n-points 10
python run_all_methods_all_datasets.py --n-points 20
python run_all_methods_all_datasets.py --n-points 30
```

**After (new way):**
```bash
# Run once with all parameters
python run_all_methods_all_datasets.py --n-points 10 20 30
```

### 2. Comprehensive Output
- ✅ Consolidated CSV with all results
- ✅ Parameter column showing which value was used
- ✅ Dynamic visualizations adapting to number of variations
- ✅ Individual JSON files per dataset
- ✅ Real-time console progress

### 3. Easy Analysis
```python
import pandas as pd
df = pd.read_csv('./runs/all_methods_comparison/consolidated_results.csv')

# Find best parameters instantly
best = df.loc[df.groupby('dataset')['auc'].idxmax()]
print(best[['dataset', 'method', 'parameter', 'auc']])
```

---

## 📊 The 6 Methods

| # | Method | Parameter Type | Example Command |
|---|--------|----------------|-----------------|
| 1 | Basic ROC Search | None (fixed) | Runs automatically |
| 2 | Remove Hull Points | None (fixed) | Runs automatically |
| 3 | Closest Points to Hull | **n-points** | `--n-points 10 20 30` |
| 4 | Furthest from Diagonal | **n-points** | `--n-points 10 20 30` |
| 5 | Points Below Hull | **percentage** | `--distance-percentage 1.0 2.0` |
| 6 | Points Above Diagonal | **percentage** | `--distance-percentage 1.0 2.0` |

---

## 💡 Common Use Cases

### Use Case 1: Find Optimal Compression Ratio
**Goal:** Reduce points while maintaining AUC

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 5 10 15 20 25 30 40 50 \
    --datasets adult \
    --output ./runs/compression_study
```

Then analyze:
```python
import pandas as pd
df = pd.read_csv('./runs/compression_study/consolidated_results.csv')
closest = df[df['method_key'].str.contains('closest')]
closest.plot(x='parameter', y='auc', style='o-')
```

### Use Case 2: Compare Selection Strategies
**Goal:** Which method is best for my data?

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 15 \
    --distance-percentage 2.0 \
    --datasets adult mushroom ionosphere
```

### Use Case 3: Parameter Sensitivity Analysis
**Goal:** How sensitive are results to parameter changes?

```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --distance-percentage 0.1 0.5 1.0 2.0 5.0 10.0 \
    --datasets ionosphere
```

---

## 📈 Example Workflow

### 1. Initial Exploration (Fast)
```bash
python run_all_methods_all_datasets.py --depth 1 --datasets adult
```
*Takes ~30 seconds, gives you a feel for the methods*

### 2. Parameter Sweep (Comprehensive)
```bash
python run_all_methods_all_datasets.py \
    --depth 2 \
    --n-points 5 10 15 20 25 \
    --distance-percentage 0.5 1.0 2.0 5.0 \
    --datasets adult ionosphere
```
*Takes ~5-10 minutes, comprehensive parameter testing*

### 3. Analysis
```bash
python analyze_multi_param_results.py
```
*Generates plots and finds optimal parameters*

### 4. Apply Best Parameters (Production)
Use the best parameters found for your specific dataset in production runs.

---

## 🔧 All Command Line Options

```bash
python run_all_methods_all_datasets.py \
    --data-dir ./tests \                    # Where datasets are
    --depth 2 \                             # Search depth
    --min-coverage 50 \                     # Min subgroup size
    --n-points 10 20 30 \                   # Multiple n-points
    --distance-percentage 1.0 2.0 5.0 \     # Multiple percentages
    --datasets adult mushroom \              # Specific datasets
    --output ./runs/my_analysis              # Output directory
```

### Default Values
- `--data-dir`: `./tests`
- `--depth`: `2`
- `--min-coverage`: `50`
- `--n-points`: `[10]` (single value)
- `--distance-percentage`: `[1.0]` (single value)
- `--datasets`: All datasets in data-dir
- `--output`: `./runs/all_methods_comparison`

---

## 📖 Documentation Reference

| File | Purpose | When to Read |
|------|---------|--------------|
| **QUICK_START_MULTI_PARAMS.md** | Ready-to-use examples | Start here! |
| **MULTI_PARAM_SUMMARY.md** | High-level overview | Quick understanding |
| **MULTI_PARAM_GUIDE.md** | Technical details | Deep dive |
| **README_MULTI_PARAMS.md** | This file | Complete reference |

---

## 🎯 Cheat Sheet

### Quick Commands

```bash
# Minimal test
python run_all_methods_all_datasets.py --depth 1 --datasets adult

# Standard comparison
python run_all_methods_all_datasets.py --depth 2 --n-points 10 20

# Full parameter sweep
python run_all_methods_all_datasets.py \
    --n-points 5 10 15 20 25 \
    --distance-percentage 0.5 1.0 2.0 5.0

# Analyze results
python analyze_multi_param_results.py
```

### Quick Analysis

```python
import pandas as pd

# Load results
df = pd.read_csv('./runs/all_methods_comparison/consolidated_results.csv')

# View summary
print(df.groupby('method')['auc'].describe())

# Best parameters per dataset
best = df.loc[df.groupby(['dataset', 'method'])['auc'].idxmax()]
print(best[['dataset', 'method', 'parameter', 'auc']])

# Plot AUC trends
closest = df[df['method_key'].str.contains('closest')]
closest.groupby('parameter')['auc'].mean().plot(marker='o', title='AUC vs N-Points')
```

---

## ⚡ Performance Tips

1. **Start Small**: Test 1-2 datasets first
2. **Use Depth 1-2**: Deeper searches take much longer
3. **Limit Parameters**: Start with 2-3 values per parameter
4. **Parallel Datasets**: Process datasets in parallel if needed

### Estimated Times

| Configuration | Methods/Dataset | Time/Dataset | Total (8 datasets) |
|---------------|-----------------|--------------|-------------------|
| Minimal (depth 1) | 6 | ~30 sec | ~4 min |
| Standard (depth 2) | 6 | ~1 min | ~8 min |
| 3 params each | 14 | ~3 min | ~24 min |
| 5 params each | 22 | ~5 min | ~40 min |

---

## 🔬 Advanced Features

### Custom Output Directory
```bash
python run_all_methods_all_datasets.py \
    --output ./experiments/run_$(date +%Y%m%d_%H%M%S)
```

### Test Specific Datasets
```bash
python run_all_methods_all_datasets.py \
    --datasets adult ionosphere wisconsin
```

### Fine-Grained Parameter Testing
```bash
python run_all_methods_all_datasets.py \
    --n-points 5 7 10 12 15 18 20 25 30 \
    --distance-percentage 0.1 0.25 0.5 0.75 1.0 1.5 2.0 3.0 5.0
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Install required packages
pip install pandas numpy scipy matplotlib
```

### Issue: "Dataset not found"
```bash
# Check datasets exist
ls tests/*.txt

# Verify path
python run_all_methods_all_datasets.py --data-dir ./tests
```

### Issue: "Results file not found" (when analyzing)
```bash
# Run the main script first
python run_all_methods_all_datasets.py --depth 2

# Then analyze
python analyze_multi_param_results.py
```

### Issue: Script runs too long
```bash
# Use fewer parameters
python run_all_methods_all_datasets.py --depth 1 --n-points 10 20

# Or test fewer datasets
python run_all_methods_all_datasets.py --datasets adult
```

---

## 📦 Output Structure

```
runs/all_methods_comparison/
├── consolidated_results.csv           # Main results file
│   └── Columns: dataset, method, parameter, auc, points, time, etc.
│
├── adult/
│   ├── adult_comparison.png          # Visual comparison
│   └── adult_results.json             # Detailed results
│
├── ionosphere/
│   ├── ionosphere_comparison.png
│   └── ionosphere_results.json
│
└── ... (one folder per dataset)
```

---

## ✅ Summary

**What Changed:**
- ✅ Support for multiple `--n-points` values
- ✅ Support for multiple `--distance-percentage` values
- ✅ Parameter-aware result keys and CSV columns
- ✅ Dynamic visualizations for any number of variations
- ✅ Comprehensive analysis scripts

**Backward Compatible:**
- ✅ Old commands still work with defaults
- ✅ Default: `n-points=[10]`, `distance-percentage=[1.0]`

**New Capabilities:**
- ✅ Test multiple parameters in single run
- ✅ Easy parameter optimization
- ✅ Systematic method comparison
- ✅ Comprehensive result analysis

**Ready to Use:**
```bash
python run_all_methods_all_datasets.py --depth 2 --n-points 10 20 30
```

---

## 🎓 Learning Path

1. **Start**: Read `QUICK_START_MULTI_PARAMS.md`
2. **Run**: Try Example 5 (Quick Test)
3. **Explore**: Run with multiple parameters
4. **Analyze**: Use `analyze_multi_param_results.py`
5. **Deep Dive**: Read `MULTI_PARAM_GUIDE.md` for details

---

## 📞 Need Help?

1. Check the documentation files
2. Run with `--help` flag
3. Start with simple examples
4. Test on small datasets first

---

**Happy Parameter Optimization! 🚀**
