"""
Comparison Analysis: True ROC Search vs Enhanced Beam Search

This script compares the results from our true ROC search implementation 
(with adaptive width) against the enhanced beam search (with fixed width).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_comparison_analysis():
    """Create comprehensive comparison between True ROC Search and Enhanced Beam Search."""
    
    # Load results from both approaches
    true_roc_results = pd.read_csv('./runs/true_roc/true_roc_comparison.csv')
    beam_search_results = pd.read_csv('./runs/enhanced_roc/alpha_comparison.csv')
    
    # Filter beam search results for comparison (keep only relevant alphas)
    common_alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    beam_filtered = beam_search_results[beam_search_results['alpha'].isin(common_alphas)]
    true_filtered = true_roc_results[true_roc_results['alpha'].isin(common_alphas)]
    
    print("=== COMPARISON: True ROC Search vs Enhanced Beam Search ===\n")
    
    # Create comparison table
    comparison_data = []
    for alpha in common_alphas:
        true_row = true_filtered[true_filtered['alpha'] == alpha].iloc[0]
        beam_row = beam_filtered[beam_filtered['alpha'] == alpha].iloc[0]
        
        comparison_data.append({
            'Alpha': alpha,
            'True_ROC_Width': true_row['adaptive_width'],
            'Beam_Search_Width': beam_row['width'],
            'True_ROC_AUC': true_row['auc_approx'],
            'Beam_Search_AUC': beam_row['AUC'],
            'True_ROC_Quality': true_row['best_quality'],
            'Beam_Search_Quality': beam_row['best_quality'],
            'True_ROC_Time': true_row['search_time'],
            'Beam_Search_Candidates': beam_row['num_subgroups'],
            'True_ROC_Candidates': true_row['total_candidates']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print comparison table
    print("ALGORITHM COMPARISON:")
    print("=" * 120)
    print(f"{'Alpha':<6} {'True ROC':<15} {'Beam Search':<15} {'True ROC':<12} {'Beam Search':<12} {'True ROC':<12} {'Beam Search':<12}")
    print(f"{'':>6} {'Width':<15} {'Width':<15} {'AUC':<12} {'AUC':<12} {'Quality':<12} {'Quality':<12}")
    print("=" * 120)
    
    for _, row in comparison_df.iterrows():
        print(f"{row['Alpha']:<6.1f} {int(row['True_ROC_Width']):<15d} {int(row['Beam_Search_Width']):<15d} "
              f"{row['True_ROC_AUC']:<12.3f} {row['Beam_Search_AUC']:<12.3f} "
              f"{row['True_ROC_Quality']:<12.3f} {row['Beam_Search_Quality']:<12.3f}")
    
    print("=" * 120)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 50)
    
    avg_true_width = comparison_df['True_ROC_Width'].mean()
    avg_beam_width = comparison_df['Beam_Search_Width'].mean()
    
    print(f"1. ADAPTIVE WIDTH:")
    print(f"   - True ROC Search: {avg_true_width:.1f} subgroups (adaptive, range: {comparison_df['True_ROC_Width'].min()}-{comparison_df['True_ROC_Width'].max()})")
    print(f"   - Beam Search: {avg_beam_width:.1f} subgroups (fixed width)")
    print(f"   - Width Reduction: {(1 - avg_true_width/avg_beam_width)*100:.1f}% fewer subgroups with adaptive approach")
    
    print(f"\n2. SEARCH EFFICIENCY:")
    avg_true_candidates = comparison_df['True_ROC_Candidates'].mean()
    avg_beam_candidates = comparison_df['Beam_Search_Candidates'].mean()
    print(f"   - True ROC Search: {avg_true_candidates:.0f} candidates explored")
    print(f"   - Beam Search: {avg_beam_candidates:.0f} final subgroups kept")
    print(f"   - True ROC explores more candidates but keeps fewer final results")
    
    print(f"\n3. PERFORMANCE COMPARISON:")
    true_auc_range = f"{comparison_df['True_ROC_AUC'].min():.3f}-{comparison_df['True_ROC_AUC'].max():.3f}"
    beam_auc_range = f"{comparison_df['Beam_Search_AUC'].min():.3f}-{comparison_df['Beam_Search_AUC'].max():.3f}"
    print(f"   - True ROC AUC range: {true_auc_range}")
    print(f"   - Beam Search AUC range: {beam_auc_range}")
    
    print(f"\n4. ALGORITHMIC DIFFERENCES:")
    print(f"   - True ROC Search: ROC convex hull-based pruning (quality-driven)")
    print(f"   - Beam Search: Fixed top-k selection (coverage-driven)")
    print(f"   - True ROC Search mimics Table 2 behavior (adaptive width 1-37)")
    print(f"   - Beam Search provides consistent exploration (fixed width 30-50)")
    
    # Create visualization
    create_comparison_plots(comparison_df)
    
    # Save comparison
    comparison_df.to_csv('./runs/algorithm_comparison.csv', index=False)
    print(f"\n5. RESULTS SAVED:")
    print(f"   - Comparison table: ./runs/algorithm_comparison.csv")
    print(f"   - Visualization: ./runs/algorithm_comparison.png")
    
    return comparison_df

def create_comparison_plots(comparison_df):
    """Create comparison visualization."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    alphas = comparison_df['Alpha']
    
    # Plot 1: Width comparison
    ax1.plot(alphas, comparison_df['True_ROC_Width'], 'bo-', linewidth=2, markersize=8, label='True ROC (Adaptive)')
    ax1.plot(alphas, comparison_df['Beam_Search_Width'], 'rs-', linewidth=2, markersize=8, label='Beam Search (Fixed)')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Width (Number of Subgroups)')
    ax1.set_title('Width Comparison: Adaptive vs Fixed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC comparison
    ax2.plot(alphas, comparison_df['True_ROC_AUC'], 'bo-', linewidth=2, markersize=8, label='True ROC')
    ax2.plot(alphas, comparison_df['Beam_Search_AUC'], 'rs-', linewidth=2, markersize=8, label='Beam Search')
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('AUC')
    ax2.set_title('AUC Performance Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Quality comparison
    ax3.plot(alphas, comparison_df['True_ROC_Quality'], 'bo-', linewidth=2, markersize=8, label='True ROC')
    ax3.plot(alphas, comparison_df['Beam_Search_Quality'], 'rs-', linewidth=2, markersize=8, label='Beam Search')
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Best Quality')
    ax3.set_title('Best Quality Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency comparison (candidates vs final subgroups)
    width_ratio = comparison_df['True_ROC_Width'] / comparison_df['Beam_Search_Width']
    ax4.bar(alphas, width_ratio, alpha=0.7, color='green')
    ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Parity')
    ax4.set_xlabel('Alpha')
    ax4.set_ylabel('Width Ratio (True ROC / Beam Search)')
    ax4.set_title('Width Efficiency: True ROC vs Beam Search')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add text annotation
    avg_ratio = width_ratio.mean()
    ax4.text(0.5, 0.8, f'Avg Reduction: {(1-avg_ratio)*100:.1f}%', 
             transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('./runs/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    comparison_df = create_comparison_analysis()