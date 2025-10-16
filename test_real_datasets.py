"""
Test all hull manipulation functions using REAL datasets from your existing results.
Tests across multiple datasets (adult, ionosphere, mushroom) and depths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import all hull manipulation functions
from true_roc_search import (
    remove_hull_points_and_recalculate,
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal,
    select_points_below_hull,
    select_points_above_diagonal
)


def load_roc_points_from_csv(csv_path, depth=None):
    """
    Load ROC points from existing CSV results.
    
    Args:
        csv_path: Path to CSV file with depth, fpr, tpr columns
        depth: Optional - filter to specific depth (if None, use all depths)
    
    Returns:
        Dictionary mapping depth to ROC points array
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if 'fpr' not in df.columns or 'tpr' not in df.columns:
            print(f"Error: CSV must have 'fpr' and 'tpr' columns")
            return None
        
        # If depth column exists, group by depth
        if 'depth' in df.columns:
            if depth is not None:
                # Filter to specific depth
                depth_df = df[df['depth'] == depth]
                points = depth_df[['fpr', 'tpr']].values
                return {depth: points}
            else:
                # Return all depths
                depth_dict = {}
                for d in sorted(df['depth'].unique()):
                    depth_df = df[df['depth'] == d]
                    points = depth_df[['fpr', 'tpr']].values
                    depth_dict[d] = points
                return depth_dict
        else:
            # No depth column, return all points as depth 1
            points = df[['fpr', 'tpr']].values
            return {1: points}
            
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def test_all_functions_on_dataset(dataset_name, csv_path, depths_to_test=None, n_points_list=[10, 20]):
    """
    Test all 5 hull manipulation functions on a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'adult', 'ionosphere')
        csv_path: Path to the CSV file with ROC results
        depths_to_test: List of depths to test (if None, test all available)
        n_points_list: List of N values for selection functions
    """
    print(f"\n{'='*80}")
    print(f"Testing Dataset: {dataset_name.upper()}")
    print(f"Source: {csv_path}")
    print(f"{'='*80}\n")
    
    # Load data
    depth_points = load_roc_points_from_csv(csv_path)
    
    if depth_points is None:
        print(f"Failed to load data for {dataset_name}")
        return
    
    # Filter depths if specified
    if depths_to_test is not None:
        depth_points = {d: pts for d, pts in depth_points.items() if d in depths_to_test}
    
    # Create output directory for this dataset
    output_dir = Path(f'./runs/real_data_tests/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    
    # Test each depth
    for depth, points in sorted(depth_points.items()):
        print(f"\n{'─'*80}")
        print(f"Depth {depth}: {len(points)} ROC points")
        print(f"{'─'*80}")
        
        if len(points) < 3:
            print(f"Skipping depth {depth} - not enough points (need at least 3)")
            continue
        
        # Test 1: Remove hull points and recalculate
        print(f"\n--- Test 1: Remove Hull Points ---")
        hull_removal_result = remove_hull_points_and_recalculate(points, return_details=True)
        
        print(f"Original hull: {hull_removal_result['original_num_subgroups']} points, "
              f"AUC={hull_removal_result['original_auc']:.4f}")
        print(f"New hull: {hull_removal_result['new_num_subgroups']} points, "
              f"AUC={hull_removal_result['new_auc']:.4f}")
        print(f"AUC reduction: {hull_removal_result['auc_reduction_percentage']:.2f}%")
        
        all_results.append({
            'dataset': dataset_name,
            'depth': depth,
            'method': 'remove_hull',
            'n_points': hull_removal_result['original_num_subgroups'],
            'original_auc': hull_removal_result['original_auc'],
            'new_auc': hull_removal_result['new_auc'],
            'auc_change_pct': hull_removal_result['auc_reduction_percentage'],
            'new_hull_points': hull_removal_result['new_num_subgroups']
        })
        
        # Test 2-5: Selection functions with different N values
        for n_points in n_points_list:
            if n_points >= len(points):
                print(f"\nSkipping N={n_points} (need {n_points} but only have {len(points)} points)")
                continue
            
            print(f"\n--- Testing with N={n_points} points ---")
            
            # Test with exclude_hull=True for clearer results
            exclude_hull = True
            
            # Closest to hull
            print(f"  • Closest to Hull (exclude_hull={exclude_hull})...")
            closest = select_closest_points_to_hull(points, n_points, return_details=True, 
                                                    exclude_hull_points=exclude_hull)
            print(f"    AUC change: {closest.get('auc_reduction_percentage', 0):.2f}%, "
                  f"New hull: {len(closest['new_hull'])} points")
            
            all_results.append({
                'dataset': dataset_name,
                'depth': depth,
                'method': f'closest_to_hull (exclude={exclude_hull})',
                'n_points': n_points,
                'original_auc': closest.get('original_auc', 0),
                'new_auc': closest.get('new_auc', 0),
                'auc_change_pct': closest.get('auc_reduction_percentage', 0),
                'new_hull_points': len(closest['new_hull'])
            })
            
            # Create individual plot for Closest to Hull
            create_individual_method_plots(
                dataset_name, depth, n_points, points,
                'Closest to Hull', closest, output_dir
            )
            
            # Furthest from diagonal
            print(f"  • Furthest from Diagonal (exclude_hull={exclude_hull})...")
            furthest = select_furthest_points_from_diagonal(points, n_points, return_details=True,
                                                           exclude_hull_points=exclude_hull)
            print(f"    AUC change: {furthest.get('auc_reduction_percentage', 0):.2f}%, "
                  f"New hull: {len(furthest['new_hull'])} points")
            
            all_results.append({
                'dataset': dataset_name,
                'depth': depth,
                'method': f'furthest_from_diagonal (exclude={exclude_hull})',
                'n_points': n_points,
                'original_auc': furthest.get('original_auc', 0),
                'new_auc': furthest.get('new_auc', 0),
                'auc_change_pct': furthest.get('auc_reduction_percentage', 0),
                'new_hull_points': len(furthest['new_hull'])
            })
            
            # Create individual plot for Furthest from Diagonal
            create_individual_method_plots(
                dataset_name, depth, n_points, points,
                'Furthest from Diagonal', furthest, output_dir
            )
            
            # Below hull
            print(f"  • Below Hull (exclude_hull={exclude_hull})...")
            below = select_points_below_hull(points, n_points, return_details=True,
                                            exclude_hull_points=exclude_hull)
            print(f"    AUC change: {below.get('auc_reduction_percentage', 0):.2f}%, "
                  f"New hull: {len(below['new_hull'])} points")
            print(f"    Avg vertical distance: {below.get('avg_vertical_distance', 0):.4f}")
            
            all_results.append({
                'dataset': dataset_name,
                'depth': depth,
                'method': f'below_hull (exclude={exclude_hull})',
                'n_points': n_points,
                'original_auc': below.get('original_auc', 0),
                'new_auc': below.get('new_auc', 0),
                'auc_change_pct': below.get('auc_reduction_percentage', 0),
                'new_hull_points': len(below['new_hull']),
                'avg_vertical_distance': below.get('avg_vertical_distance', 0)
            })
            
            # Create individual plot for Below Hull
            create_individual_method_plots(
                dataset_name, depth, n_points, points,
                'Below Hull', below, output_dir
            )
            
            # Above diagonal
            print(f"  • Above Diagonal (exclude_hull={exclude_hull})...")
            above = select_points_above_diagonal(points, n_points, return_details=True,
                                                exclude_hull_points=exclude_hull)
            print(f"    AUC change: {above.get('auc_reduction_percentage', 0):.2f}%, "
                  f"New hull: {len(above['new_hull'])} points")
            print(f"    Avg TPR: {above.get('avg_tpr_selected', 0):.4f}")
            
            all_results.append({
                'dataset': dataset_name,
                'depth': depth,
                'method': f'above_diagonal (exclude={exclude_hull})',
                'n_points': n_points,
                'original_auc': above.get('original_auc', 0),
                'new_auc': above.get('new_auc', 0),
                'auc_change_pct': above.get('auc_reduction_percentage', 0),
                'new_hull_points': len(above['new_hull']),
                'avg_tpr_selected': above.get('avg_tpr_selected', 0)
            })
            
            # Create individual plot for Above Diagonal
            create_individual_method_plots(
                dataset_name, depth, n_points, points,
                'Above Diagonal', above, output_dir
            )
            
            # Create 4x3 comparison visualization for this depth and N
            create_comparison_plot(
                dataset_name, depth, n_points, points,
                closest, furthest, below, above,
                output_dir
            )
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / f'{dataset_name}_all_methods_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved results to: {results_path}")
    
    # Count plots created
    plot_files = list(output_dir.glob('*.png'))
    print(f"✓ Created {len(plot_files)} visualization plots")
    
    return results_df


def create_individual_method_plots(dataset_name, depth, n_points, original_points,
                                  method_name, result, output_dir):
    """
    Create individual 3-panel plot for a single method (like the original test scripts).
    Shows: Original Hull | Selected Points | Overlay Comparison
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    original_hull = result['original_hull']
    selected = result['selected_points']
    new_hull = result['new_hull']
    
    # Panel 1: Original hull with all points
    axes[0].plot(original_points[:, 0], original_points[:, 1], 'bo', alpha=0.5, 
                label='All points', markersize=6)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[0].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2, 
                    label='Original hull')
        axes[0].scatter(original_hull[:, 0], original_hull[:, 1], c='red', s=100, 
                       marker='*', zorder=5, label='Hull points', edgecolors='black')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title(f'Original Hull\n{dataset_name.upper()} - Depth {depth}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # Panel 2: Selected points and new hull
    axes[1].plot(selected[:, 0], selected[:, 1], 'go', alpha=0.5, 
                label=f'{n_points} selected', markersize=6)
    if len(new_hull) > 0:
        hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
        axes[1].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', linewidth=2, 
                    label='New hull')
        axes[1].scatter(new_hull[:, 0], new_hull[:, 1], c='purple', s=100, 
                       marker='*', zorder=5, label='New hull points', edgecolors='black')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1].set_xlabel('FPR')
    axes[1].set_ylabel('TPR')
    axes[1].set_title(f'{method_name}\n{n_points} Points Selected')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    # Panel 3: Overlay comparison
    axes[2].plot(original_points[:, 0], original_points[:, 1], 'bo', alpha=0.3, 
                label='All points', markersize=4)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2, 
                    alpha=0.7, label='Original hull')
    if len(new_hull) > 0:
        hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
        axes[2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', 
                    linewidth=2, linestyle='--', alpha=0.7, label='New hull')
    axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[2].set_xlabel('FPR')
    axes[2].set_ylabel('TPR')
    
    # Add statistics to title
    auc_change = result.get('auc_reduction_percentage', 0)
    orig_auc = result.get('original_auc', 0)
    new_auc = result.get('new_auc', 0)
    axes[2].set_title(f'Overlay Comparison\n'
                     f'AUC: {orig_auc:.4f} → {new_auc:.4f} ({auc_change:+.1f}%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    
    plt.suptitle(f'{method_name}: {dataset_name.upper()} - Depth {depth} - N={n_points}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    method_slug = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    output_path = output_dir / f'{dataset_name}_depth{depth}_n{n_points}_{method_slug}_individual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def create_comparison_plot(dataset_name, depth, n_points, original_points,
                          closest_result, furthest_result, below_result, above_result,
                          output_dir):
    """
    Create a 4x3 comparison plot for all methods on real data.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    
    # Get original hull from any result (they're all the same)
    original_hull = closest_result['original_hull']
    
    method_configs = [
        ('Closest to Hull', closest_result, 'green', 'lime'),
        ('Furthest from Diagonal', furthest_result, 'purple', 'violet'),
        ('Below Hull', below_result, 'darkblue', 'cyan'),
        ('Above Diagonal', above_result, 'darkred', 'orange')
    ]
    
    for row, (method_name, result, hull_color, point_color) in enumerate(method_configs):
        selected = result['selected_points']
        new_hull = result['new_hull']
        
        # Column 1: Original hull
        axes[row, 0].plot(original_points[:, 0], original_points[:, 1], 'bo', 
                         alpha=0.3, markersize=4, label='All points')
        if len(original_hull) > 0:
            hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
            axes[row, 0].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                            linewidth=2, label='Original hull')
            axes[row, 0].scatter(original_hull[:, 0], original_hull[:, 1], 
                               c='red', s=100, marker='*', zorder=5, edgecolors='black')
        axes[row, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
        axes[row, 0].set_xlabel('FPR')
        axes[row, 0].set_ylabel('TPR')
        axes[row, 0].set_title('Original Hull')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_ylim(0, 1)
        
        # Column 2: Selected points
        axes[row, 1].plot(selected[:, 0], selected[:, 1], 'o', color=point_color,
                         alpha=0.6, markersize=6, label=f'{n_points} selected')
        if len(new_hull) > 0:
            hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
            axes[row, 1].plot(hull_sorted[:, 0], hull_sorted[:, 1], color=hull_color,
                            linewidth=2, label='New hull')
            axes[row, 1].scatter(new_hull[:, 0], new_hull[:, 1], 
                               c=hull_color, s=100, marker='*', zorder=5, edgecolors='black')
        axes[row, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
        axes[row, 1].set_xlabel('FPR')
        axes[row, 1].set_ylabel('TPR')
        axes[row, 1].set_title(f'{method_name}: {n_points} Points')
        axes[row, 1].legend(fontsize=8)
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_ylim(0, 1)
        
        # Column 3: Overlay
        axes[row, 2].plot(original_points[:, 0], original_points[:, 1], 'bo', 
                         alpha=0.2, markersize=3, label='All points')
        if len(original_hull) > 0:
            hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
            axes[row, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                            linewidth=2, alpha=0.7, label='Original')
        if len(new_hull) > 0:
            hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
            axes[row, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], color=hull_color,
                            linewidth=2, linestyle='--', alpha=0.7, label=f'New')
        axes[row, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
        axes[row, 2].set_xlabel('FPR')
        axes[row, 2].set_ylabel('TPR')
        
        auc_change = result.get('auc_reduction_percentage', 0)
        orig_auc = result.get('original_auc', 0)
        new_auc = result.get('new_auc', 0)
        axes[row, 2].set_title(f'{method_name} Overlay\n'
                              f'AUC: {orig_auc:.4f}→{new_auc:.4f} ({auc_change:+.1f}%)')
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 2].set_xlim(0, 1)
        axes[row, 2].set_ylim(0, 1)
    
    plt.suptitle(f'All Methods Comparison: {dataset_name.upper()} - Depth {depth} - N={n_points}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / f'{dataset_name}_depth{depth}_n{n_points}_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved plot: {output_path.name}")
    
    plt.close()


def create_summary_report(all_datasets_results):
    """
    Create a comprehensive summary report across all datasets.
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE SUMMARY REPORT - ALL DATASETS")
    print(f"{'='*80}\n")
    
    # Combine all results
    combined_df = pd.concat(all_datasets_results.values(), ignore_index=True)
    
    # Summary by method
    print("\n--- Summary by Method (Average AUC Change %) ---")
    method_summary = combined_df.groupby('method')['auc_change_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
    method_summary = method_summary.sort_values('mean', ascending=False)
    print(method_summary.to_string())
    
    # Summary by dataset
    print("\n--- Summary by Dataset (Average AUC Change %) ---")
    dataset_summary = combined_df.groupby('dataset')['auc_change_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
    dataset_summary = dataset_summary.sort_values('mean', ascending=False)
    print(dataset_summary.to_string())
    
    # Summary by depth
    print("\n--- Summary by Depth (Average AUC Change %) ---")
    depth_summary = combined_df.groupby('depth')['auc_change_pct'].agg(['mean', 'std', 'min', 'max', 'count'])
    depth_summary = depth_summary.sort_values('depth')
    print(depth_summary.to_string())
    
    # Save comprehensive results
    output_path = Path('./runs/real_data_tests/comprehensive_summary.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved comprehensive results to: {output_path}")
    
    return combined_df


def main():
    """
    Main test function - tests all datasets.
    """
    print("="*80)
    print("TESTING ALL HULL MANIPULATION FUNCTIONS ON REAL DATASETS")
    print("="*80)
    
    # Define datasets to test - using the ROC points files
    datasets = {
        'adult': 'runs/all_datasets_complete/adult/alpha_pure_roc/roc_points.csv',
        'ionosphere': 'runs/all_datasets_complete/ionosphere/alpha_pure_roc/roc_points.csv',
        'mushroom': 'runs/all_datasets_complete/mushroom/alpha_pure_roc/roc_points.csv'
    }
    
    # Test parameters
    depths_to_test = None  # Use all points (no depth filtering since file doesn't have depth column)
    n_points_list = [5, 10]    # Test with these N values
    
    # Storage for all results
    all_datasets_results = {}
    
    # Test each dataset
    for dataset_name, csv_path in datasets.items():
        # Check if file exists
        if not Path(csv_path).exists():
            print(f"\n⚠ Warning: {csv_path} not found, skipping {dataset_name}")
            continue
        
        # Test all functions on this dataset
        results_df = test_all_functions_on_dataset(
            dataset_name, 
            csv_path, 
            depths_to_test=depths_to_test,
            n_points_list=n_points_list
        )
        
        all_datasets_results[dataset_name] = results_df
    
    # Create comprehensive summary
    if all_datasets_results:
        combined_df = create_summary_report(all_datasets_results)
        
        print(f"\n{'='*80}")
        print("ALL TESTS COMPLETED!")
        print(f"{'='*80}")
        print(f"\nTested {len(all_datasets_results)} datasets")
        print(f"Total experiments: {len(combined_df)}")
        print(f"\nResults saved to: ./runs/real_data_tests/")
        print(f"\nDatasets tested: {', '.join(all_datasets_results.keys())}")
        print(f"Depths tested: {depths_to_test}")
        print(f"N values tested: {n_points_list}")
    else:
        print("\n⚠ No datasets were successfully tested!")


if __name__ == "__main__":
    main()
