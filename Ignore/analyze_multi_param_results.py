"""
Example script to analyze results from multi-parameter ROC search runs.

This demonstrates how to load and analyze the consolidated results CSV
to find optimal parameters and compare methods across datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(results_path='./runs/all_methods_comparison/consolidated_results.csv'):
    """Load the consolidated results CSV."""
    if not Path(results_path).exists():
        print(f"Error: Results file not found at {results_path}")
        print("Please run the multi-parameter script first!")
        return None
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} results from {len(df['dataset'].unique())} datasets")
    print(f"Methods tested: {df['method_key'].nunique()}")
    return df


def analyze_npoints_effect(df, method_prefix='closest_points'):
    """Analyze the effect of n-points parameter on a specific method."""
    print(f"\n{'='*70}")
    print(f"N-Points Analysis for {method_prefix}")
    print(f"{'='*70}")
    
    # Filter to the specific method
    method_df = df[df['method_key'].str.contains(method_prefix)].copy()
    
    if len(method_df) == 0:
        print(f"No results found for method: {method_prefix}")
        return
    
    # Convert parameter to numeric
    method_df['n_points'] = pd.to_numeric(method_df['parameter'], errors='coerce')
    method_df = method_df.dropna(subset=['n_points'])
    
    # Group by dataset and n_points
    pivot = method_df.pivot_table(
        values='auc',
        index='dataset',
        columns='n_points',
        aggfunc='first'
    )
    
    print("\nAUC by Dataset and N-Points:")
    print(pivot.round(4))
    
    # Find best n-points per dataset
    print("\nBest N-Points per Dataset:")
    best = method_df.loc[method_df.groupby('dataset')['auc'].idxmax()]
    for _, row in best.iterrows():
        print(f"  {row['dataset']}: n={int(row['n_points'])}, AUC={row['auc']:.4f}")
    
    # Plot trends
    plt.figure(figsize=(12, 6))
    for dataset in method_df['dataset'].unique():
        data = method_df[method_df['dataset'] == dataset].sort_values('n_points')
        plt.plot(data['n_points'], data['auc'], marker='o', label=dataset, linewidth=2)
    
    plt.xlabel('N Points', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'AUC vs N-Points - {method_prefix.replace("_", " ").title()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{method_prefix}_npoints_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {method_prefix}_npoints_analysis.png")
    plt.show()
    plt.close()


def analyze_percentage_effect(df, method_prefix='below_hull'):
    """Analyze the effect of percentage parameter on a specific method."""
    print(f"\n{'='*70}")
    print(f"Percentage Analysis for {method_prefix}")
    print(f"{'='*70}")
    
    # Filter to the specific method
    method_df = df[df['method_key'].str.contains(method_prefix)].copy()
    
    if len(method_df) == 0:
        print(f"No results found for method: {method_prefix}")
        return
    
    # Convert parameter to numeric
    method_df['percentage'] = pd.to_numeric(method_df['parameter'], errors='coerce')
    method_df = method_df.dropna(subset=['percentage'])
    
    # Group by dataset and percentage
    pivot = method_df.pivot_table(
        values='auc',
        index='dataset',
        columns='percentage',
        aggfunc='first'
    )
    
    print("\nAUC by Dataset and Percentage:")
    print(pivot.round(4))
    
    # Find best percentage per dataset
    print("\nBest Percentage per Dataset:")
    best = method_df.loc[method_df.groupby('dataset')['auc'].idxmax()]
    for _, row in best.iterrows():
        print(f"  {row['dataset']}: {row['percentage']:.1f}%, AUC={row['auc']:.4f}")
    
    # Plot trends
    plt.figure(figsize=(12, 6))
    for dataset in method_df['dataset'].unique():
        data = method_df[method_df['dataset'] == dataset].sort_values('percentage')
        plt.plot(data['percentage'], data['auc'], marker='o', label=dataset, linewidth=2)
    
    plt.xlabel('Distance Percentage (%)', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.title(f'AUC vs Percentage - {method_prefix.replace("_", " ").title()}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{method_prefix}_percentage_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {method_prefix}_percentage_analysis.png")
    plt.show()
    plt.close()


def compare_all_methods(df, dataset='adult'):
    """Compare all methods for a specific dataset."""
    print(f"\n{'='*70}")
    print(f"Method Comparison for {dataset}")
    print(f"{'='*70}")
    
    dataset_df = df[df['dataset'] == dataset].copy()
    
    if len(dataset_df) == 0:
        print(f"No results found for dataset: {dataset}")
        return
    
    # Sort by AUC
    dataset_df = dataset_df.sort_values('auc', ascending=False)
    
    print("\nAll Methods Ranked by AUC:")
    print(f"{'Rank':<6} {'Method':<45} {'Param':<10} {'AUC':<10} {'Points':<10}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(dataset_df.iterrows(), 1):
        method = row['method'][:43]  # Truncate if too long
        param = str(row['parameter'])[:8] if pd.notna(row['parameter']) else 'N/A'
        auc = row['auc']
        points = row['num_hull_points']
        print(f"{i:<6} {method:<45} {param:<10} {auc:<10.4f} {points:<10}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: AUC comparison
    methods = dataset_df['method'].values
    aucs = dataset_df['auc'].values
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    
    ax1.barh(range(len(methods)), aucs, color=colors)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels([m[:30] for m in methods], fontsize=8)
    ax1.set_xlabel('AUC', fontsize=12)
    ax1.set_title(f'AUC Comparison - {dataset}', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: AUC vs Number of Points
    ax2.scatter(dataset_df['num_hull_points'], dataset_df['auc'], 
               c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Number of Hull Points', fontsize=12)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title(f'AUC vs Points - {dataset}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add method labels to points
    for _, row in dataset_df.iterrows():
        method_short = row['method'].split()[0]  # First word of method
        ax2.annotate(method_short, 
                    (row['num_hull_points'], row['auc']),
                    fontsize=7, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{dataset}_method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {dataset}_method_comparison.png")
    plt.show()
    plt.close()


def find_pareto_optimal(df):
    """Find Pareto-optimal solutions (highest AUC for given number of points)."""
    print(f"\n{'='*70}")
    print("Pareto-Optimal Solutions")
    print(f"{'='*70}")
    
    for dataset in df['dataset'].unique():
        print(f"\n{dataset}:")
        dataset_df = df[df['dataset'] == dataset].copy()
        
        # Sort by number of points
        dataset_df = dataset_df.sort_values('num_hull_points')
        
        pareto = []
        max_auc_seen = -1
        
        for _, row in dataset_df.iterrows():
            if row['auc'] > max_auc_seen:
                max_auc_seen = row['auc']
                pareto.append(row)
        
        print(f"  {'Points':<10} {'AUC':<10} {'Method':<40} {'Param':<10}")
        print("  " + "-" * 70)
        for row in pareto:
            points = int(row['num_hull_points'])
            auc = row['auc']
            method = row['method'][:38]
            param = str(row['parameter'])[:8] if pd.notna(row['parameter']) else 'N/A'
            print(f"  {points:<10} {auc:<10.4f} {method:<40} {param:<10}")


def generate_summary_report(df):
    """Generate a comprehensive summary report."""
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SUMMARY REPORT")
    print(f"{'='*70}")
    
    print(f"\nDatasets analyzed: {df['dataset'].nunique()}")
    print(f"Total method variations: {df['method_key'].nunique()}")
    print(f"Total results: {len(df)}")
    
    print("\n--- Overall Best Performance ---")
    best_overall = df.loc[df['auc'].idxmax()]
    print(f"Dataset: {best_overall['dataset']}")
    print(f"Method: {best_overall['method']}")
    print(f"Parameter: {best_overall['parameter']}")
    print(f"AUC: {best_overall['auc']:.4f}")
    print(f"Hull Points: {int(best_overall['num_hull_points'])}")
    
    print("\n--- Average Performance by Method Type ---")
    method_types = {
        'Basic': df[df['method_key'] == 'basic'],
        'Remove Hull': df[df['method_key'] == 'remove_hull'],
        'Closest Points': df[df['method_key'].str.contains('closest')],
        'Furthest Points': df[df['method_key'].str.contains('furthest')],
        'Below Hull': df[df['method_key'].str.contains('below_hull')],
        'Above Diagonal': df[df['method_key'].str.contains('above_diagonal')]
    }
    
    for method_name, method_df in method_types.items():
        if len(method_df) > 0:
            avg_auc = method_df['auc'].mean()
            avg_points = method_df['num_hull_points'].mean()
            print(f"{method_name:<20} - Avg AUC: {avg_auc:.4f}, Avg Points: {avg_points:.1f}")
    
    print("\n--- Dataset-Specific Best Methods ---")
    for dataset in df['dataset'].unique():
        dataset_df = df[df['dataset'] == dataset]
        best = dataset_df.loc[dataset_df['auc'].idxmax()]
        print(f"{dataset:<15} - {best['method'][:30]} (AUC: {best['auc']:.4f})")


def main():
    """Main analysis workflow."""
    print("="*70)
    print("MULTI-PARAMETER ROC SEARCH RESULTS ANALYZER")
    print("="*70)
    
    # Load results
    df = load_results()
    if df is None:
        return
    
    # Generate comprehensive report
    generate_summary_report(df)
    
    # Analyze n-points effect (methods 3 & 4)
    if any(df['method_key'].str.contains('closest_points')):
        analyze_npoints_effect(df, 'closest_points')
    
    if any(df['method_key'].str.contains('furthest_points')):
        analyze_npoints_effect(df, 'furthest_points')
    
    # Analyze percentage effect (methods 5 & 6)
    if any(df['method_key'].str.contains('below_hull')):
        analyze_percentage_effect(df, 'below_hull')
    
    if any(df['method_key'].str.contains('above_diagonal')):
        analyze_percentage_effect(df, 'above_diagonal')
    
    # Compare all methods for first dataset
    first_dataset = df['dataset'].iloc[0]
    compare_all_methods(df, first_dataset)
    
    # Find Pareto-optimal solutions
    find_pareto_optimal(df)
    
    print("\n" + "="*70)
    print("Analysis complete! Check the generated plots.")
    print("="*70)


if __name__ == '__main__':
    main()
