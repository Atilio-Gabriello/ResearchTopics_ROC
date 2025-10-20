"""
Script to run all 6 search methods on all datasets
Depth = 4, Min Coverage = 50
"""

import subprocess
import sys

def main():
    print("="*80)
    print("Running All Search Methods on All Datasets")
    print("="*80)
    print("Parameters:")
    print("  - Depth: 4")
    print("  - Min Coverage: 50")
    print("  - Output: ./runs/all_methods_depth4")
    print("="*80)
    
    # Run the batch analysis
    cmd = [
        sys.executable,
        "true_roc_search.py",
        "--batch",
        "--depth", "4",
        "--min-coverage", "50",
        "--data-dir", "./tests",
        "--output", "./runs/all_methods_depth4",
        "--pure-roc"
    ]
    
    print(f"\nExecuting: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n" + "="*80)
        print("SUCCESS: All methods completed!")
        print("="*80)
        print("Results saved to: ./runs/all_methods_depth4")
        print("\nCheck the following files:")
        print("  - consolidated_summary.csv: Summary of all methods and datasets")
        print("  - consolidated_depth_analysis.csv: Depth-by-depth analysis")
        print("  - Individual dataset folders with detailed results")
    else:
        print("\n" + "="*80)
        print("ERROR: Batch analysis failed!")
        print("="*80)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
