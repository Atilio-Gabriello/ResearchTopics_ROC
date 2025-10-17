"""
Compare Wide Beam Search results with Comprehensive ROC Search Methods

This script loads both result sets and creates comparison tables and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load both result sets."""
    # Load wide beam results
    wide_beam_path = Path('./runs/wide_beam_search/wide_beam_summary.csv')
    wide_beam_df = pd.read_csv(wide_beam_path)
    wide_beam_df['method'] = 'Wide Beam (pysubdisc)'
    
    # Load comprehensive ROC results (best configuration for each method)
    roc_path = Path('./runs/comprehensive_all_methods/comprehensive_results.csv')
    roc_df = pd.read_csv(roc_path)
    
    # Filter for depth=4 only
    roc_df = roc_df[roc_df['depth'] == 4].copy()
    
    # Get best configuration for each method per dataset
    best_roc_df = roc_df.sort_values('auc', ascending=False).groupby(['dataset', 'method']).first().reset_index()
    
    return wide_beam_df, best_roc_df, roc_df


def create_comparison_table(wide_beam_df, best_roc_df):
    """Create a comparison table showing all methods."""
    # Common datasets
    common_datasets = set(wide_beam_df['dataset']) & set(best_roc_df['dataset'])
    
    print("="*100)
    print("WIDE BEAM SEARCH vs ROC SEARCH METHODS COMPARISON (Depth=4)")
    print("="*100)
    
    # Create comparison for each dataset
    comparison_data = []
    
    for dataset in sorted(common_datasets):
        wb_row = wide_beam_df[wide_beam_df['dataset'] == dataset].iloc[0]
        
        dataset_data = {
            'Dataset': dataset,
            'Wide Beam AUC': wb_row['auc'],
            'Wide Beam Points': wb_row['num_roc_points'],
            'Wide Beam Time(s)': wb_row['time_seconds']
        }
        
        # Get all ROC method results for this dataset
        roc_methods = best_roc_df[best_roc_df['dataset'] == dataset]
        
        for _, roc_row in roc_methods.iterrows():
            method_name = roc_row['method']
            dataset_data[f'{method_name} AUC'] = roc_row['auc']
            # Use total_points if available, or num_candidates, or num_subgroups
            points_col = 'total_points' if 'total_points' in roc_row and pd.notna(roc_row['total_points']) else 'num_subgroups'
            dataset_data[f'{method_name} Points'] = roc_row[points_col] if pd.notna(roc_row[points_col]) else 0
            dataset_data[f'{method_name} Time(s)'] = roc_row['time_seconds']
        
        comparison_data.append(dataset_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display simplified comparison - AUC only
    print("\n" + "="*100)
    print("AUC COMPARISON BY DATASET")
    print("="*100)
    
    auc_columns = ['Dataset', 'Wide Beam AUC'] + [col for col in comparison_df.columns if col.endswith(' AUC') and col != 'Wide Beam AUC']
    auc_df = comparison_df[auc_columns].copy()
    
    # Rename columns for better display
    auc_df.columns = auc_df.columns.str.replace(' AUC', '')
    
    print(auc_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Find best method per dataset
    print("\n" + "="*100)
    print("BEST METHOD PER DATASET (by AUC)")
    print("="*100)
    
    for _, row in auc_df.iterrows():
        dataset = row['Dataset']
        auc_values = row.drop('Dataset')
        best_method = auc_values.idxmax()
        best_auc = auc_values.max()
        wb_auc = row['Wide Beam']
        
        if best_method == 'Wide Beam':
            status = "🏆 WINNER"
        elif wb_auc >= best_auc - 0.01:  # Within 1% of best
            status = "✓ Competitive"
        else:
            diff = best_auc - wb_auc
            status = f"△ -{diff:.4f}"
        
        print(f"{dataset:15s} | Best: {best_method:30s} (AUC={best_auc:.4f}) | Wide Beam: {status}")
    
    # Overall statistics
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    
    wb_mean_auc = wide_beam_df['auc'].mean()
    roc_mean_aucs = best_roc_df.groupby('method')['auc'].mean().sort_values(ascending=False)
    
    print(f"\nMean AUC Across Datasets:")
    print(f"  Wide Beam Search:    {wb_mean_auc:.4f}")
    print(f"\nTop ROC Methods:")
    for i, (method, auc) in enumerate(roc_mean_aucs.head(3).items(), 1):
        diff = auc - wb_mean_auc
        symbol = "+" if diff > 0 else ""
        print(f"  {i}. {method:27s} {auc:.4f} ({symbol}{diff:.4f})")
    
    # Time comparison
    print(f"\n{'='*100}")
    print("EXECUTION TIME COMPARISON")
    print("="*100)
    
    wb_mean_time = wide_beam_df['time_seconds'].mean()
    roc_mean_times = best_roc_df.groupby('method')['time_seconds'].mean().sort_values()
    
    print(f"\nMean Execution Time:")
    print(f"  Wide Beam Search:    {wb_mean_time:.4f}s")
    print(f"\nROC Methods:")
    for method, time in roc_mean_times.items():
        speedup = wb_mean_time / time if time > 0 else float('inf')
        print(f"  {method:27s} {time:.4f}s ({speedup:.1f}x faster than WB)" if speedup > 1 
              else f"  {method:27s} {time:.4f}s ({1/speedup:.1f}x slower than WB)")
    
    # ROC points comparison
    print(f"\n{'='*100}")
    print("ROC POINTS GENERATED")
    print("="*100)
    
    wb_mean_points = wide_beam_df['num_roc_points'].mean()
    # Use total_points if available, otherwise use num_subgroups
    points_col = 'total_points' if 'total_points' in best_roc_df.columns else 'num_subgroups'
    roc_mean_points = best_roc_df.groupby('method')[points_col].mean().sort_values(ascending=False)
    
    print(f"\nMean ROC Points:")
    print(f"  Wide Beam Search:    {wb_mean_points:.0f}")
    print(f"\nROC Methods:")
    for method, points in roc_mean_points.items():
        print(f"  {method:27s} {points:.0f}")
    
    return comparison_df, auc_df


def create_comparison_plots(wide_beam_df, best_roc_df):
    """Create visual comparisons."""
    common_datasets = set(wide_beam_df['dataset']) & set(best_roc_df['dataset'])
    
    # 1. AUC Comparison Bar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for plotting
    methods = sorted(best_roc_df['method'].unique())
    datasets = sorted(common_datasets)
    
    # AUC comparison
    ax = axes[0, 0]
    x = np.arange(len(datasets))
    width = 0.12
    
    # Plot Wide Beam
    wb_aucs = [wide_beam_df[wide_beam_df['dataset'] == ds]['auc'].values[0] for ds in datasets]
    ax.bar(x - width*3.5, wb_aucs, width, label='Wide Beam', color='#1f77b4', alpha=0.8)
    
    # Plot ROC methods
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for i, method in enumerate(methods):
        method_aucs = []
        for ds in datasets:
            ds_data = best_roc_df[(best_roc_df['dataset'] == ds) & (best_roc_df['method'] == method)]
            if len(ds_data) > 0:
                method_aucs.append(ds_data['auc'].values[0])
            else:
                method_aucs.append(0)
        ax.bar(x - width*3.5 + width*(i+1), method_aucs, width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=11, fontweight='bold')
    ax.set_title('AUC Comparison Across Datasets (Depth=4)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Time comparison
    ax = axes[0, 1]
    wb_times = [wide_beam_df[wide_beam_df['dataset'] == ds]['time_seconds'].values[0] for ds in datasets]
    ax.bar(x - width*3.5, wb_times, width, label='Wide Beam', color='#1f77b4', alpha=0.8)
    
    for i, method in enumerate(methods):
        method_times = []
        for ds in datasets:
            ds_data = best_roc_df[(best_roc_df['dataset'] == ds) & (best_roc_df['method'] == method)]
            if len(ds_data) > 0:
                method_times.append(ds_data['time_seconds'].values[0])
            else:
                method_times.append(0)
        ax.bar(x - width*3.5 + width*(i+1), method_times, width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # ROC Points comparison
    ax = axes[1, 0]
    wb_points = [wide_beam_df[wide_beam_df['dataset'] == ds]['num_roc_points'].values[0] for ds in datasets]
    ax.bar(x - width*3.5, wb_points, width, label='Wide Beam', color='#1f77b4', alpha=0.8)
    
    points_col = 'total_points' if 'total_points' in best_roc_df.columns else 'num_subgroups'
    for i, method in enumerate(methods):
        method_points = []
        for ds in datasets:
            ds_data = best_roc_df[(best_roc_df['dataset'] == ds) & (best_roc_df['method'] == method)]
            if len(ds_data) > 0:
                val = ds_data[points_col].values[0]
                method_points.append(val if pd.notna(val) else 0)
            else:
                method_points.append(0)
        ax.bar(x - width*3.5 + width*(i+1), method_points, width, label=method, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of ROC Points', fontsize=11, fontweight='bold')
    ax.set_title('ROC Points Generated', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Mean AUC comparison
    ax = axes[1, 1]
    all_methods = ['Wide Beam'] + methods
    mean_aucs = [wide_beam_df['auc'].mean()]
    mean_aucs.extend([best_roc_df[best_roc_df['method'] == m]['auc'].mean() for m in methods])
    
    colors_bar = ['#1f77b4'] + [colors[i] for i in range(len(methods))]
    bars = ax.barh(all_methods, mean_aucs, color=colors_bar, alpha=0.8)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', 
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Mean AUC', fontsize=11, fontweight='bold')
    ax.set_title('Mean AUC Across All Datasets', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1)
    
    # Highlight best
    best_idx = np.argmax(mean_aucs)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    output_path = Path('./runs/wide_beam_search/comparison_with_roc_methods.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()
    
    # 2. AUC Ranking Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create matrix of AUC values
    auc_matrix = []
    row_labels = []
    
    for dataset in datasets:
        row = []
        wb_auc = wide_beam_df[wide_beam_df['dataset'] == dataset]['auc'].values[0]
        row.append(wb_auc)
        
        for method in methods:
            ds_data = best_roc_df[(best_roc_df['dataset'] == dataset) & (best_roc_df['method'] == method)]
            if len(ds_data) > 0:
                row.append(ds_data['auc'].values[0])
            else:
                row.append(0)
        
        auc_matrix.append(row)
        row_labels.append(dataset)
    
    auc_matrix = np.array(auc_matrix)
    col_labels = ['Wide Beam'] + methods
    
    # Create heatmap
    im = ax.imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right')
    ax.set_yticklabels(row_labels)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AUC', rotation=270, labelpad=20, fontweight='bold')
    
    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = auc_matrix[i, j]
            # Find best in row
            best_in_row = auc_matrix[i].max()
            text_color = 'white' if value < 0.5 else 'black'
            text_weight = 'bold' if value == best_in_row else 'normal'
            
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", 
                          color=text_color, fontsize=9,
                          fontweight=text_weight)
            
            # Add star for best in row
            if value == best_in_row:
                ax.text(j, i-0.35, '★',
                       ha="center", va="center", 
                       color='gold', fontsize=12)
    
    ax.set_title('AUC Heatmap: Wide Beam vs ROC Methods (★ = Best per Dataset)', 
                 fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    output_path = Path('./runs/wide_beam_search/auc_heatmap_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved AUC heatmap to: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("Loading results...")
    wide_beam_df, best_roc_df, full_roc_df = load_results()
    
    print(f"Loaded Wide Beam results: {len(wide_beam_df)} datasets")
    print(f"Loaded ROC results: {len(best_roc_df)} configurations\n")
    
    # Create comparison table
    comparison_df, auc_df = create_comparison_table(wide_beam_df, best_roc_df)
    
    # Save comparison
    output_dir = Path('./runs/wide_beam_search')
    comparison_df.to_csv(output_dir / 'detailed_comparison.csv', index=False)
    auc_df.to_csv(output_dir / 'auc_comparison.csv', index=False)
    print(f"\n{'='*100}")
    print(f"Saved detailed comparison to: {output_dir / 'detailed_comparison.csv'}")
    print(f"Saved AUC comparison to: {output_dir / 'auc_comparison.csv'}")
    
    # Create plots
    create_comparison_plots(wide_beam_df, best_roc_df)
    
    print(f"\n{'='*100}")
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
