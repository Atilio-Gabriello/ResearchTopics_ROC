"""
Create a comprehensive metrics comparison table showing original vs new hull metrics.
"""

import numpy as np
import pandas as pd
from true_roc_search import remove_hull_points_and_recalculate
import matplotlib.pyplot as plt
from pathlib import Path

def generate_sample_points(n=50, seed=42):
    """Generate sample ROC points."""
    np.random.seed(seed)
    fpr = np.random.beta(2, 5, n)
    tpr = np.random.beta(5, 2, n)
    tpr = np.maximum(tpr, fpr + np.random.uniform(0.05, 0.2, n))
    return np.column_stack([np.clip(fpr, 0, 1), np.clip(tpr, 0, 1)])

def create_metrics_comparison_table():
    """Create a comprehensive comparison table with all metrics."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS COMPARISON TABLE")
    print("=" * 80)
    
    # Generate data for multiple depths
    depths = [1, 2, 3]
    all_comparisons = []
    
    for depth in depths:
        n_points = 30 + (depth * 20)
        points = generate_sample_points(n=n_points, seed=depth)
        
        # Get detailed comparison
        hull_data = remove_hull_points_and_recalculate(points, return_details=True)
        
        comparison = {
            'Depth': depth,
            'Total Points': len(hull_data.get('all_points', [])),
            
            # Original Hull
            'Orig Hull Subgroups': hull_data.get('original_num_subgroups', 0),
            'Orig AUC': hull_data.get('original_auc', 0),
            'Orig Best Quality': hull_data.get('original_max_quality', 0),
            'Orig Avg Quality': hull_data.get('original_avg_quality', 0),
            'Orig Best TPR': hull_data.get('original_best_tpr', 0),
            'Orig Best FPR': hull_data.get('original_best_fpr', 0),
            
            # New Hull
            'New Hull Subgroups': hull_data.get('new_num_subgroups', 0),
            'New AUC': hull_data.get('new_auc', 0),
            'New Best Quality': hull_data.get('new_max_quality', 0),
            'New Avg Quality': hull_data.get('new_avg_quality', 0),
            'New Best TPR': hull_data.get('new_best_tpr', 0),
            'New Best FPR': hull_data.get('new_best_fpr', 0),
            
            # Reductions
            'Subgroups Removed': hull_data.get('subgroups_removed', 0),
            'AUC Reduction': hull_data.get('auc_reduction', 0),
            'AUC Reduction %': hull_data.get('auc_reduction_percentage', 0),
            'Quality Reduction': hull_data.get('quality_reduction', 0),
            'Area Reduction %': hull_data.get('reduction_percentage', 0)
        }
        
        all_comparisons.append(comparison)
    
    # Create DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Save to CSV
    output_dir = Path('./runs/hull_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'comprehensive_metrics_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comprehensive comparison to: {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 80)
    print("ORIGINAL HULL vs NEW HULL METRICS")
    print("=" * 80)
    print()
    
    # Print comparison by depth
    for _, row in df.iterrows():
        print(f"{'='*80}")
        print(f"DEPTH {int(row['Depth'])} - {int(row['Total Points'])} Total Points")
        print(f"{'='*80}")
        print()
        
        print(f"{'Metric':<30} {'Original Hull':>20} {'New Hull':>20} {'Reduction':>10}")
        print(f"{'-'*80}")
        print(f"{'Number of Subgroups':<30} {int(row['Orig Hull Subgroups']):>20} {int(row['New Hull Subgroups']):>20} {int(row['Subgroups Removed']):>10}")
        print(f"{'AUC':<30} {row['Orig AUC']:>20.4f} {row['New AUC']:>20.4f} {row['AUC Reduction %']:>9.1f}%")
        print(f"{'Best Quality (TPR-FPR)':<30} {row['Orig Best Quality']:>20.4f} {row['New Best Quality']:>20.4f} {row['Quality Reduction']:>10.4f}")
        print(f"{'Average Quality':<30} {row['Orig Avg Quality']:>20.4f} {row['New Avg Quality']:>20.4f} {'-':>10}")
        print(f"{'Best TPR':<30} {row['Orig Best TPR']:>20.4f} {row['New Best TPR']:>20.4f} {'-':>10}")
        print(f"{'Best FPR':<30} {row['Orig Best FPR']:>20.4f} {row['New Best FPR']:>20.4f} {'-':>10}")
        print(f"{'Hull Area Reduction':<30} {'-':>20} {'-':>20} {row['Area Reduction %']:>9.1f}%")
        print()
    
    # Create visualization
    create_metrics_visualization(df, output_dir)
    
    return df

def create_metrics_visualization(df, output_dir):
    """Create visualization of metrics comparison."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Original Hull vs New Hull - Comprehensive Metrics Comparison', 
                 fontsize=16, fontweight='bold')
    
    depths = df['Depth'].values
    
    # Plot 1: Number of Subgroups
    ax = axes[0, 0]
    width = 0.35
    x = np.arange(len(depths))
    ax.bar(x - width/2, df['Orig Hull Subgroups'], width, label='Original Hull', color='red', alpha=0.7)
    ax.bar(x + width/2, df['New Hull Subgroups'], width, label='New Hull', color='purple', alpha=0.7)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Number of Subgroups')
    ax.set_title('Subgroups on Hull')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: AUC Comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, df['Orig AUC'], width, label='Original Hull', color='red', alpha=0.7)
    ax.bar(x + width/2, df['New AUC'], width, label='New Hull', color='purple', alpha=0.7)
    ax.set_xlabel('Depth')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: AUC Reduction Percentage
    ax = axes[0, 2]
    ax.bar(depths, df['AUC Reduction %'], color='orange', alpha=0.7)
    ax.set_xlabel('Depth')
    ax.set_ylabel('AUC Reduction (%)')
    ax.set_title('AUC Loss from Removing Hull Points')
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Best Quality Comparison
    ax = axes[1, 0]
    ax.bar(x - width/2, df['Orig Best Quality'], width, label='Original Hull', color='red', alpha=0.7)
    ax.bar(x + width/2, df['New Best Quality'], width, label='New Hull', color='purple', alpha=0.7)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Quality (TPR - FPR)')
    ax.set_title('Best Quality Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(depths)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Best TPR vs FPR (scatter)
    ax = axes[1, 1]
    ax.scatter(df['Orig Best FPR'], df['Orig Best TPR'], s=200, c='red', 
              marker='*', label='Original Hull', edgecolors='black', linewidths=1.5, zorder=5)
    ax.scatter(df['New Best FPR'], df['New Best TPR'], s=200, c='purple', 
              marker='s', label='New Hull', edgecolors='black', linewidths=1.5, zorder=5)
    
    # Add depth labels
    for i, depth in enumerate(depths):
        ax.annotate(f'D{int(depth)}', (df['Orig Best FPR'].iloc[i], df['Orig Best TPR'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')
        ax.annotate(f'D{int(depth)}', (df['New Best FPR'].iloc[i], df['New Best TPR'].iloc[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, color='purple')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax.set_xlabel('Best FPR')
    ax.set_ylabel('Best TPR')
    ax.set_title('Best Subgroup Position (ROC Space)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, max(df['Orig Best FPR'].max(), df['New Best FPR'].max()) + 0.1)
    ax.set_ylim(min(df['Orig Best TPR'].min(), df['New Best TPR'].min()) - 0.1, 1.05)
    
    # Plot 6: Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "SUMMARY\n" + "="*40 + "\n\n"
    summary_text += f"Avg AUC Reduction: {df['AUC Reduction %'].mean():.1f}%\n"
    summary_text += f"Avg Subgroups Removed: {df['Subgroups Removed'].mean():.1f}\n"
    summary_text += f"Avg Quality Reduction: {df['Quality Reduction'].mean():.4f}\n\n"
    
    summary_text += "INTERPRETATION:\n"
    avg_auc_reduction = df['AUC Reduction %'].mean()
    if avg_auc_reduction > 5:
        summary_text += "• High AUC loss\n  Hull points dominate\n  performance"
    elif avg_auc_reduction > 2:
        summary_text += "• Moderate AUC loss\n  Balanced hull\n  contribution"
    else:
        summary_text += "• Low AUC loss\n  Good point\n  diversity"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / 'comprehensive_metrics_visualization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved metrics visualization to: {plot_path}")
    
    plt.show()
    plt.close()

if __name__ == '__main__':
    df = create_metrics_comparison_table()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • comprehensive_metrics_comparison.csv")
    print("  • comprehensive_metrics_visualization.png")
    print("\nLocation: ./runs/hull_test/")
