"""
Test the new percentage-based threshold approach for functions 4 and 5.

NEW APPROACH:
- Both functions use the FURTHEST point from the diagonal as the reference
- Function 4: threshold = max_diagonal_distance × (percentage / 100), select hull_distance ≤ threshold
- Function 5: threshold = max_diagonal_distance × ((100 - percentage) / 100), select diagonal_distance ≥ threshold

Example with 1% parameter and max_diagonal_distance = 0.8:
- Function 4: threshold = 0.8 × 0.01 = 0.008, select points ≤ 0.008 from hull
- Function 5: threshold = 0.8 × 0.99 = 0.792, select points ≥ 0.792 from diagonal
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from true_roc_search import (
    load_data, 
    preprocess_categorical_data,
    calculate_subgroup_stats,
    generate_candidates,
    select_points_below_hull,
    select_points_above_diagonal
)

def run_test_on_dataset(data_path, target_col, percentages=[0.5, 1.0, 2.0, 5.0, 10.0], output_dir='./runs/new_percentage_tests'):
    """Test new percentage approach on a dataset."""
    
    print(f"\n{'='*80}")
    print(f"Testing dataset: {data_path}")
    print(f"Target column: {target_col}")
    print(f"{'='*80}\n")
    
    # Load and preprocess data
    data = load_data(data_path)
    if data is None:
        print(f"Failed to load {data_path}")
        return
    
    data = preprocess_categorical_data(data)
    
    # Generate candidates at depth 2
    print("Generating candidate subgroups at depth 2...")
    candidates, hull_comparison = generate_candidates(
        data, target_col, 
        [calculate_subgroup_stats(data, [], target_col)], 
        depth=2, min_coverage=10
    )
    
    if not candidates:
        print("No candidates generated!")
        return
    
    print(f"Generated {len(candidates)} candidates")
    
    # Extract ROC points
    all_points = []
    for candidate in candidates:
        if 'tpr' in candidate and 'fpr' in candidate:
            all_points.append([candidate['fpr'], candidate['tpr']])
    
    all_points = np.array(all_points)
    print(f"Total ROC points: {len(all_points)}")
    
    # Filter points above diagonal
    above_diagonal = all_points[all_points[:, 1] > all_points[:, 0]]
    print(f"Points above diagonal: {len(above_diagonal)}")
    
    if len(above_diagonal) < 3:
        print("Not enough points above diagonal!")
        return
    
    # Calculate max diagonal distance for reference
    diagonal_distances = above_diagonal[:, 1] - above_diagonal[:, 0]
    max_diagonal_distance = np.max(diagonal_distances)
    print(f"\nMax diagonal distance (reference): {max_diagonal_distance:.4f}")
    
    # Create output directory
    dataset_name = Path(data_path).stem
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test results storage
    results_below_hull = []
    results_above_diagonal = []
    
    # Test each percentage
    for pct in percentages:
        print(f"\n{'─'*80}")
        print(f"Testing {pct}% threshold")
        print(f"{'─'*80}")
        
        # Calculate expected thresholds
        threshold_below_hull = max_diagonal_distance * (pct / 100.0)
        threshold_above_diagonal = max_diagonal_distance * ((100.0 - pct) / 100.0)
        
        print(f"\nFunction 4 (below hull):")
        print(f"  Threshold: {threshold_below_hull:.6f} ({pct}% of {max_diagonal_distance:.4f})")
        print(f"  Select points with hull_distance ≤ {threshold_below_hull:.6f}")
        
        print(f"\nFunction 5 (above diagonal):")
        print(f"  Threshold: {threshold_above_diagonal:.6f} ({100-pct}% of {max_diagonal_distance:.4f})")
        print(f"  Select points with diagonal_distance ≥ {threshold_above_diagonal:.6f}")
        
        # Test function 4: select_points_below_hull
        try:
            result_below = select_points_below_hull(
                all_points, 
                distance_percentage=pct, 
                return_details=True,
                exclude_hull_points=True
            )
            
            n_selected_below = result_below.get('n_selected', 0)
            auc_below = result_below.get('new_auc', 0)
            
            print(f"\n✓ Function 4 Results:")
            print(f"  Points selected: {n_selected_below}")
            print(f"  New hull points: {len(result_below.get('new_hull', []))}")
            print(f"  New AUC: {auc_below:.4f}")
            print(f"  AUC reduction: {result_below.get('auc_reduction', 0):.4f} ({result_below.get('auc_reduction_percentage', 0):.2f}%)")
            
            results_below_hull.append({
                'percentage': pct,
                'threshold': threshold_below_hull,
                'n_selected': n_selected_below,
                'n_hull_points': len(result_below.get('new_hull', [])),
                'new_auc': auc_below,
                'auc_reduction': result_below.get('auc_reduction', 0),
                'auc_reduction_pct': result_below.get('auc_reduction_percentage', 0)
            })
            
        except Exception as e:
            print(f"\n✗ Function 4 Error: {e}")
            results_below_hull.append({
                'percentage': pct,
                'threshold': threshold_below_hull,
                'error': str(e)
            })
        
        # Test function 5: select_points_above_diagonal
        try:
            result_above = select_points_above_diagonal(
                all_points,
                distance_percentage=pct,
                return_details=True,
                exclude_hull_points=True
            )
            
            n_selected_above = result_above.get('n_selected', 0)
            auc_above = result_above.get('new_auc', 0)
            
            print(f"\n✓ Function 5 Results:")
            print(f"  Points selected: {n_selected_above}")
            print(f"  New hull points: {len(result_above.get('new_hull', []))}")
            print(f"  New AUC: {auc_above:.4f}")
            print(f"  AUC reduction: {result_above.get('auc_reduction', 0):.4f} ({result_above.get('auc_reduction_percentage', 0):.2f}%)")
            
            results_above_diagonal.append({
                'percentage': pct,
                'threshold': threshold_above_diagonal,
                'n_selected': n_selected_above,
                'n_hull_points': len(result_above.get('new_hull', [])),
                'new_auc': auc_above,
                'auc_reduction': result_above.get('auc_reduction', 0),
                'auc_reduction_pct': result_above.get('auc_reduction_percentage', 0)
            })
            
        except Exception as e:
            print(f"\n✗ Function 5 Error: {e}")
            results_above_diagonal.append({
                'percentage': pct,
                'threshold': threshold_above_diagonal,
                'error': str(e)
            })
    
    # Save results to CSV
    df_below = pd.DataFrame(results_below_hull)
    df_above = pd.DataFrame(results_above_diagonal)
    
    df_below.to_csv(output_path / 'below_hull_results.csv', index=False)
    df_above.to_csv(output_path / 'above_diagonal_results.csv', index=False)
    
    print(f"\n✓ Saved results to {output_path}")
    
    # Create visualization
    create_comparison_plot(results_below_hull, results_above_diagonal, max_diagonal_distance, output_path / 'comparison.png')
    
    return results_below_hull, results_above_diagonal


def create_comparison_plot(results_below, results_above, max_diagonal_dist, output_path):
    """Create comparison visualization."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    percentages_below = [r['percentage'] for r in results_below if 'n_selected' in r]
    n_selected_below = [r['n_selected'] for r in results_below if 'n_selected' in r]
    auc_reduction_below = [r.get('auc_reduction_pct', 0) for r in results_below if 'n_selected' in r]
    
    percentages_above = [r['percentage'] for r in results_above if 'n_selected' in r]
    n_selected_above = [r['n_selected'] for r in results_above if 'n_selected' in r]
    auc_reduction_above = [r.get('auc_reduction_pct', 0) for r in results_above if 'n_selected' in r]
    
    # Plot 1: Number of selected points
    axes[0, 0].plot(percentages_below, n_selected_below, 'o-', label='Below Hull', linewidth=2, markersize=8)
    axes[0, 0].plot(percentages_above, n_selected_above, 's-', label='Above Diagonal', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Percentage Parameter (%)')
    axes[0, 0].set_ylabel('Number of Points Selected')
    axes[0, 0].set_title('Points Selected vs Percentage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: AUC reduction
    axes[0, 1].plot(percentages_below, auc_reduction_below, 'o-', label='Below Hull', linewidth=2, markersize=8)
    axes[0, 1].plot(percentages_above, auc_reduction_above, 's-', label='Above Diagonal', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Percentage Parameter (%)')
    axes[0, 1].set_ylabel('AUC Reduction (%)')
    axes[0, 1].set_title('AUC Reduction vs Percentage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Threshold values
    thresholds_below = [max_diagonal_dist * (p / 100.0) for p in percentages_below]
    thresholds_above = [max_diagonal_dist * ((100.0 - p) / 100.0) for p in percentages_above]
    
    axes[1, 0].plot(percentages_below, thresholds_below, 'o-', label='Below Hull Threshold', linewidth=2, markersize=8)
    axes[1, 0].plot(percentages_above, thresholds_above, 's-', label='Above Diagonal Threshold', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=max_diagonal_dist, color='r', linestyle='--', label=f'Max Diagonal Dist = {max_diagonal_dist:.4f}')
    axes[1, 0].set_xlabel('Percentage Parameter (%)')
    axes[1, 0].set_ylabel('Threshold Value')
    axes[1, 0].set_title('Threshold Values vs Percentage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    axes[1, 1].axis('off')
    
    summary_text = f"""
    NEW PERCENTAGE APPROACH SUMMARY
    
    Reference Distance (Max Diagonal):
    {max_diagonal_dist:.6f}
    
    Function 4 (Below Hull):
    - Threshold = max_dist × (pct / 100)
    - Selects: hull_distance ≤ threshold
    - Example (1%): {max_diagonal_dist * 0.01:.6f}
    
    Function 5 (Above Diagonal):
    - Threshold = max_dist × ((100-pct) / 100)
    - Selects: diagonal_distance ≥ threshold
    - Example (1%): {max_diagonal_dist * 0.99:.6f}
    
    Both use the SAME reference point:
    the furthest point from the diagonal.
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('New Percentage-Based Threshold Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved visualization to {output_path}")


if __name__ == '__main__':
    # Test on multiple datasets
    datasets = [
        ('tests/adult.txt', 'target'),
        ('tests/ionosphere.txt', 'Attribute35'),
        ('tests/mushroom.txt', 'poisonous')
    ]
    
    print("\n" + "="*80)
    print("TESTING NEW PERCENTAGE-BASED THRESHOLD APPROACH")
    print("="*80)
    print("\nKEY CHANGES:")
    print("  1. Both functions use MAX diagonal distance as reference")
    print("  2. Function 4: threshold = max_dist × (pct/100), select ≤ threshold")
    print("  3. Function 5: threshold = max_dist × ((100-pct)/100), select ≥ threshold")
    print("\nTesting percentages: 0.5%, 1.0%, 2.0%, 5.0%, 10.0%")
    print("="*80)
    
    all_results = {}
    
    for data_path, target_col in datasets:
        try:
            results = run_test_on_dataset(
                data_path, 
                target_col,
                percentages=[0.5, 1.0, 2.0, 5.0, 10.0]
            )
            all_results[Path(data_path).stem] = results
        except Exception as e:
            print(f"\n✗ Error testing {data_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: runs/new_percentage_tests/")
    print("\nCheck the CSV files and PNG visualizations for detailed results.")
