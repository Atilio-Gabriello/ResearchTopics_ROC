"""
Additional Visualizations for Comprehensive ROC Test Results

Creates advanced visualizations including:
1. Heatmaps of AUC by method and depth
2. Parameter sensitivity detailed plots
3. Dataset-specific comparisons
4. ROC quality distribution plots
5. Time complexity analysis
6. AUC reduction analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_data(results_dir='./runs/comprehensive_all_methods'):
    """Load all result files."""
    results_dir = Path(results_dir)
    
    df_full = pd.read_csv(results_dir / 'comprehensive_results.csv')
    df_method = pd.read_csv(results_dir / 'summary_by_method.csv')
    df_dataset = pd.read_csv(results_dir / 'summary_by_dataset.csv')
    df_depth = pd.read_csv(results_dir / 'summary_by_depth.csv')
    df_param = pd.read_csv(results_dir / 'parameter_analysis.csv')
    
    return df_full, df_method, df_dataset, df_depth, df_param


def create_auc_heatmap(df, output_dir):
    """Create heatmap of mean AUC by method and depth."""
    print("Creating AUC heatmap...")
    
    # Filter to methods that have AUC values
    df_with_auc = df[df['auc'].notna() | df['new_auc'].notna()].copy()
    
    # Use new_auc if auc is not available
    df_with_auc['auc_value'] = df_with_auc['auc'].fillna(df_with_auc['new_auc'])
    
    # Create pivot table
    pivot = df_with_auc.pivot_table(
        values='auc_value', 
        index='method', 
        columns='depth', 
        aggfunc='mean'
    )
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.values[i, j]):
                text = ax.text(j, i, f'{pivot.values[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean AUC', fontsize=12)
    
    ax.set_title('Mean AUC by Method and Depth', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Depth', fontsize=14)
    ax.set_ylabel('Method', fontsize=14)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'auc_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_auc_reduction_analysis(df, output_dir):
    """Analyze AUC reduction patterns."""
    print("Creating AUC reduction analysis...")
    
    # Filter methods that have AUC reduction
    df_reduction = df[df['auc_reduction'].notna()].copy()
    
    if df_reduction.empty:
        print("No AUC reduction data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC reduction by method
    ax1 = axes[0, 0]
    method_reduction = df_reduction.groupby('method')['auc_reduction'].agg(['mean', 'std'])
    method_reduction.plot(kind='bar', ax=ax1, color=['#e74c3c', '#95a5a6'])
    ax1.set_title('AUC Reduction by Method', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('AUC Reduction', fontsize=12)
    ax1.legend(['Mean', 'Std Dev'])
    ax1.grid(True, alpha=0.3)
    
    # 2. AUC reduction by depth
    ax2 = axes[0, 1]
    for method in df_reduction['method'].unique():
        method_data = df_reduction[df_reduction['method'] == method]
        depth_reduction = method_data.groupby('depth')['auc_reduction'].mean()
        ax2.plot(depth_reduction.index, depth_reduction.values, 'o-', 
                label=method, linewidth=2, markersize=8)
    ax2.set_title('AUC Reduction by Depth', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Mean AUC Reduction', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. AUC reduction distribution
    ax3 = axes[1, 0]
    df_reduction.boxplot(column='auc_reduction', by='method', ax=ax3)
    ax3.set_title('AUC Reduction Distribution by Method', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=12)
    ax3.set_ylabel('AUC Reduction', fontsize=12)
    plt.suptitle('')  # Remove default title
    
    # 4. Original vs New AUC scatter
    ax4 = axes[1, 1]
    for method in df_reduction['method'].unique():
        method_data = df_reduction[df_reduction['method'] == method]
        ax4.scatter(method_data['original_auc'], method_data['new_auc'], 
                   alpha=0.6, s=60, label=method)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='No change')
    ax4.set_title('Original vs New AUC', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Original AUC', fontsize=12)
    ax4.set_ylabel('New AUC', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, 1.0)
    ax4.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'auc_reduction_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_quality_metrics_comparison(df, output_dir):
    """Compare quality metrics across methods."""
    print("Creating quality metrics comparison...")
    
    # Filter data with quality metrics
    df_quality = df[df['max_quality'].notna()].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Max quality by method and depth
    ax1 = axes[0, 0]
    for method in df_quality['method'].unique():
        method_data = df_quality[df_quality['method'] == method]
        depth_quality = method_data.groupby('depth')['max_quality'].mean()
        ax1.plot(depth_quality.index, depth_quality.values, 'o-', 
                label=method, linewidth=2, markersize=8)
    ax1.set_title('Maximum Quality by Depth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Depth', fontsize=12)
    ax1.set_ylabel('Max Quality (TPR - FPR)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Quality distribution
    ax2 = axes[0, 1]
    quality_data = []
    labels = []
    for method in df_quality['method'].unique():
        method_data = df_quality[df_quality['method'] == method]
        quality_data.append(method_data['max_quality'].dropna())
        labels.append(method)
    ax2.boxplot(quality_data, labels=labels)
    ax2.set_title('Quality Distribution by Method', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Max Quality', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Quality reduction for methods that have it
    ax3 = axes[1, 0]
    df_qual_red = df[(df['original_max_quality'].notna()) & (df['new_max_quality'].notna())].copy()
    if not df_qual_red.empty:
        df_qual_red['quality_reduction'] = df_qual_red['original_max_quality'] - df_qual_red['new_max_quality']
        qual_red_by_method = df_qual_red.groupby('method')['quality_reduction'].agg(['mean', 'std'])
        qual_red_by_method.plot(kind='bar', ax=ax3, color=['#3498db', '#95a5a6'])
        ax3.set_title('Quality Reduction by Method', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Method', fontsize=12)
        ax3.set_ylabel('Quality Reduction', fontsize=12)
        ax3.legend(['Mean', 'Std Dev'])
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No quality reduction data', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. AUC vs Quality scatter
    ax4 = axes[1, 1]
    for method in df_quality['method'].unique():
        method_data = df_quality[df_quality['method'] == method]
        if 'auc' in method_data.columns:
            auc_col = 'auc'
        else:
            auc_col = 'new_auc'
        valid_data = method_data[[auc_col, 'max_quality']].dropna()
        ax4.scatter(valid_data['max_quality'], valid_data[auc_col], 
                   alpha=0.6, s=60, label=method)
    ax4.set_title('AUC vs Max Quality', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Max Quality (TPR - FPR)', fontsize=12)
    ax4.set_ylabel('AUC', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'quality_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_dataset_specific_analysis(df, output_dir):
    """Create dataset-specific comparison plots."""
    print("Creating dataset-specific analysis...")
    
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, dataset in enumerate(datasets):
        if idx >= 8:
            break
        
        ax = axes[idx]
        dataset_data = df[df['dataset'] == dataset]
        
        # Plot AUC by method for this dataset
        for method in dataset_data['method'].unique():
            method_data = dataset_data[dataset_data['method'] == method]
            
            # Get AUC column
            if 'auc' in method_data.columns and method_data['auc'].notna().any():
                auc_col = 'auc'
            elif 'new_auc' in method_data.columns:
                auc_col = 'new_auc'
            else:
                continue
            
            depth_auc = method_data.groupby('depth')[auc_col].mean()
            if not depth_auc.empty:
                ax.plot(depth_auc.index, depth_auc.values, 'o-', 
                       label=method[:15], linewidth=2, markersize=6, alpha=0.7)
        
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Depth', fontsize=10)
        ax.set_ylabel('Mean AUC', fontsize=10)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)
    
    # Hide unused subplots
    for idx in range(n_datasets, 8):
        axes[idx].axis('off')
    
    plt.suptitle('AUC by Depth for Each Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_dir) / 'dataset_specific_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_parameter_sensitivity_detailed(df, output_dir):
    """Create detailed parameter sensitivity plots."""
    print("Creating detailed parameter sensitivity plots...")
    
    # Filter parameter-based methods
    df_param = df[df['param_type'].notna()].copy()
    
    if df_param.empty:
        print("No parameter data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. n_points methods (closest and furthest)
    ax1 = axes[0, 0]
    for method in ['closest_points_to_hull', 'furthest_from_diagonal']:
        method_data = df_param[df_param['method'] == method]
        if not method_data.empty:
            param_auc = method_data.groupby('param_value')['new_auc'].agg(['mean', 'std'])
            ax1.errorbar(param_auc.index, param_auc['mean'], yerr=param_auc['std'],
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        label=method.replace('_', ' ').title())
    ax1.set_title('AUC vs n_points Parameter', fontsize=14, fontweight='bold')
    ax1.set_xlabel('n_points', fontsize=12)
    ax1.set_ylabel('Mean AUC', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. distance_percentage methods (below hull and above diagonal)
    ax2 = axes[0, 1]
    for method in ['below_hull_percentage', 'above_diagonal_percentage']:
        method_data = df_param[df_param['method'] == method]
        if not method_data.empty:
            param_auc = method_data.groupby('param_value')['new_auc'].agg(['mean', 'std'])
            ax2.errorbar(param_auc.index, param_auc['mean'], yerr=param_auc['std'],
                        marker='o', linewidth=2, markersize=8, capsize=5,
                        label=method.replace('_', ' ').title())
    ax2.set_title('AUC vs distance_percentage Parameter', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance Percentage (%)', fontsize=12)
    ax2.set_ylabel('Mean AUC', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Selected points by parameter
    ax3 = axes[1, 0]
    for method in df_param['method'].unique():
        method_data = df_param[df_param['method'] == method]
        if 'selected_points' in method_data.columns and method_data['selected_points'].notna().any():
            param_selected = method_data.groupby('param_value')['selected_points'].mean()
            ax3.plot(param_selected.index, param_selected.values, 'o-',
                    linewidth=2, markersize=8, label=method[:20])
    ax3.set_title('Mean Selected Points by Parameter', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Parameter Value', fontsize=12)
    ax3.set_ylabel('Mean Selected Points', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. AUC reduction by parameter
    ax4 = axes[1, 1]
    for method in df_param['method'].unique():
        method_data = df_param[df_param['method'] == method]
        if 'auc_reduction' in method_data.columns and method_data['auc_reduction'].notna().any():
            param_reduction = method_data.groupby('param_value')['auc_reduction'].mean()
            ax4.plot(param_reduction.index, param_reduction.values, 'o-',
                    linewidth=2, markersize=8, label=method[:20])
    ax4.set_title('Mean AUC Reduction by Parameter', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Parameter Value', fontsize=12)
    ax4.set_ylabel('Mean AUC Reduction', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'parameter_sensitivity_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_time_complexity_analysis(df, output_dir):
    """Analyze computational time complexity."""
    print("Creating time complexity analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Time by method
    ax1 = axes[0, 0]
    method_time = df.groupby('method')['time_seconds'].agg(['mean', 'std', 'sum'])
    method_time['mean'].plot(kind='barh', ax=ax1, color='#3498db', xerr=method_time['std'])
    ax1.set_title('Mean Execution Time by Method', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('Method', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # 2. Time by depth
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        depth_time = method_data.groupby('depth')['time_seconds'].mean()
        if not depth_time.empty:
            ax2.plot(depth_time.index, depth_time.values, 'o-',
                    label=method[:15], linewidth=2, markersize=8, alpha=0.7)
    ax2.set_title('Mean Time by Depth', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Total time by method
    ax3 = axes[1, 0]
    method_time['sum'].plot(kind='bar', ax=ax3, color='#e74c3c')
    ax3.set_title('Total Execution Time by Method', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Method', fontsize=12)
    ax3.set_ylabel('Total Time (seconds)', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Time vs number of points
    ax4 = axes[1, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        if 'total_points' in method_data.columns and method_data['total_points'].notna().any():
            valid_data = method_data[['total_points', 'time_seconds']].dropna()
            if not valid_data.empty:
                ax4.scatter(valid_data['total_points'], valid_data['time_seconds'],
                           alpha=0.6, s=60, label=method[:15])
    ax4.set_title('Time vs Number of Points', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Total Points', fontsize=12)
    ax4.set_ylabel('Time (seconds)', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'time_complexity_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_hull_area_analysis(df, output_dir):
    """Analyze convex hull area metrics."""
    print("Creating hull area analysis...")
    
    # Filter data with area metrics
    df_area = df[(df['original_area'].notna()) & (df['new_area'].notna())].copy()
    
    if df_area.empty:
        print("No hull area data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Area reduction by method
    ax1 = axes[0, 0]
    area_red_method = df_area.groupby('method')['area_reduction'].agg(['mean', 'std'])
    area_red_method.plot(kind='bar', ax=ax1, color=['#e67e22', '#95a5a6'])
    ax1.set_title('Hull Area Reduction by Method', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Area Reduction', fontsize=12)
    ax1.legend(['Mean', 'Std Dev'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Area reduction by depth
    ax2 = axes[0, 1]
    for method in df_area['method'].unique():
        method_data = df_area[df_area['method'] == method]
        depth_area = method_data.groupby('depth')['area_reduction'].mean()
        if not depth_area.empty:
            ax2.plot(depth_area.index, depth_area.values, 'o-',
                    label=method, linewidth=2, markersize=8)
    ax2.set_title('Area Reduction by Depth', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Depth', fontsize=12)
    ax2.set_ylabel('Mean Area Reduction', fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Original vs New Area
    ax3 = axes[1, 0]
    ax3.scatter(df_area['original_area'], df_area['new_area'], alpha=0.6, s=60, c=df_area['depth'], cmap='viridis')
    ax3.plot([0, df_area['original_area'].max()], [0, df_area['original_area'].max()], 
            'k--', alpha=0.3, linewidth=2, label='No change')
    ax3.set_title('Original vs New Hull Area', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Original Area', fontsize=12)
    ax3.set_ylabel('New Area', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Depth')
    
    # 4. Area reduction distribution
    ax4 = axes[1, 1]
    df_area.boxplot(column='area_reduction', by='method', ax=ax4)
    ax4.set_title('Area Reduction Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Method', fontsize=12)
    ax4.set_ylabel('Area Reduction', fontsize=12)
    plt.suptitle('')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'hull_area_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Create all additional visualizations."""
    print("="*80)
    print("Creating Additional Visualizations")
    print("="*80)
    
    results_dir = './runs/comprehensive_all_methods'
    output_dir = Path(results_dir)
    
    # Load data
    print("\nLoading data...")
    df_full, df_method, df_dataset, df_depth, df_param = load_data(results_dir)
    print(f"Loaded {len(df_full)} total results")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_auc_heatmap(df_full, output_dir)
    create_auc_reduction_analysis(df_full, output_dir)
    create_quality_metrics_comparison(df_full, output_dir)
    create_dataset_specific_analysis(df_full, output_dir)
    create_parameter_sensitivity_detailed(df_full, output_dir)
    create_time_complexity_analysis(df_full, output_dir)
    create_hull_area_analysis(df_full, output_dir)
    
    print("\n" + "="*80)
    print("All additional visualizations created successfully!")
    print(f"Output directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
