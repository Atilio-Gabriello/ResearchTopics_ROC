#!/usr/bin/env python3
"""
Debug script to understand ROC search behavior and alpha parameter effect.
"""

import pysubdisc
import pandas as pd

def test_roc_search_alpha():
    """Test ROC search with different alpha values to understand their effect."""
    
    # Load data
    data = pd.read_csv('./tests/adult.txt')
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution: {data['target'].value_counts()}")
    
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}
    
    for alpha in alphas:
        print(f"\n=== Testing alpha = {alpha} ===")
        
        # Create subgroup discovery object
        sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
        sd.searchStrategy = 'ROC_BEAM'
        sd.searchDepth = 3  # smaller depth for faster testing
        sd.searchStrategyWidth = 20  # smaller width for faster testing
        sd.maximumSubgroups = 100  # limit subgroups
        
        # Set alpha parameter
        try:
            sd.alpha = float(alpha)
            print(f"Set alpha = {sd.alpha}")
        except Exception as e:
            print(f"Failed to set alpha: {e}")
        
        # Set other ROC parameters for comparison
        try:
            sd.beta = 1.0
            sd.postProcessingDoAutoRun = False
            sd.nrThreads = 1
        except Exception as e:
            print(f"Could not set additional ROC params: {e}")
        
        # Run and collect results
        sd.run()
        df = sd.asDataFrame()
        
        results[alpha] = {
            'num_subgroups': len(df),
            'best_quality': df['Quality'].max() if not df.empty else 0,
            'mean_quality': df['Quality'].mean() if not df.empty else 0,
            'mean_coverage': df['Coverage'].mean() if not df.empty else 0,
            'top_3_conditions': df['Conditions'].head(3).tolist() if len(df) >= 3 else df['Conditions'].tolist()
        }
        
        print(f"  Subgroups found: {results[alpha]['num_subgroups']}")
        print(f"  Best quality: {results[alpha]['best_quality']:.4f}")
        print(f"  Mean quality: {results[alpha]['mean_quality']:.4f}")
        print(f"  Top conditions: {results[alpha]['top_3_conditions'][:2]}")  # Show first 2
    
    # Compare results
    print(f"\n=== COMPARISON ===")
    print("Alpha\tSubgroups\tBest_Q\t\tMean_Q\t\tMean_Cov")
    for alpha in alphas:
        r = results[alpha]
        print(f"{alpha}\t{r['num_subgroups']}\t\t{r['best_quality']:.4f}\t\t{r['mean_quality']:.4f}\t\t{r['mean_coverage']:.1f}")
    
    # Check if results are identical (indicating alpha isn't working)
    all_identical = True
    first_result = results[alphas[0]]
    for alpha in alphas[1:]:
        if (results[alpha]['num_subgroups'] != first_result['num_subgroups'] or
            abs(results[alpha]['best_quality'] - first_result['best_quality']) > 1e-6):
            all_identical = False
            break
    
    if all_identical:
        print("\n⚠️  WARNING: All results are identical! Alpha parameter may not be working.")
        print("This suggests the ROC search implementation might need adjustment.")
    else:
        print("\n✅ Results vary with alpha - ROC search is working as expected.")
    
    return results

def explain_roc_search():
    """Explain the ROC search algorithm and width decision mechanism."""
    
    print("\n" + "="*60)
    print("ROC SEARCH ALGORITHM EXPLANATION")
    print("="*60)
    
    print("""
ROC Search (Receiver Operating Characteristic Search) is a subgroup discovery
algorithm designed to find subgroups that perform well in ROC space.

KEY CONCEPTS:
1. **ROC Space**: Each subgroup is evaluated as a binary classifier
   - True Positive Rate (TPR) = TP / (TP + FN) - sensitivity
   - False Positive Rate (FPR) = FP / (FP + TN) - 1 - specificity
   - Each subgroup becomes a point (FPR, TPR) in ROC space

2. **Alpha Parameter**: Controls the trade-off in ROC space
   - α ∈ [0,1] weights the importance of TPR vs FPR
   - α = 0: Focus on minimizing FPR (high precision)
   - α = 1: Focus on maximizing TPR (high recall)
   - α = 0.5: Balanced trade-off

3. **Width Decision**: Beam search maintains top-k candidates
   - Width = number of partial subgroups kept at each level
   - Larger width = more exploration but slower search
   - ROC search may adaptively adjust width based on ROC performance

4. **Quality Measure**: ROC-based quality function
   - Typically combines TPR and FPR with alpha weighting
   - May use AUC, distance from ideal point (0,1), or custom metrics

5. **Beam Search Process**:
   - Level 0: Start with empty subgroup
   - Level i: Extend best 'width' subgroups from level i-1
   - Evaluate each extension in ROC space
   - Keep top 'width' candidates based on ROC quality
   - Continue until max depth reached

EXPECTED BEHAVIOR WITH ALPHA:
- Different alpha values should yield different subgroup sets
- α=0.0 should favor high-precision subgroups (low FPR)
- α=1.0 should favor high-recall subgroups (high TPR)
- α=0.5 should balance precision and recall
""")

if __name__ == '__main__':
    explain_roc_search()
    test_roc_search_alpha()