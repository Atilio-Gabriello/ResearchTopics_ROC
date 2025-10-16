"""
Enhanced test script with visualizations for the new percentage-based threshold approach.

Includes detailed plots showing:
1. All points, original hull, selected points, and new hull
2. Distance distributions and thresholds
3. Comparison between functions 4 and 5
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


def plot_selection_visualization(result, method_name, percentage, max_diagonal_dist, output_path):
    """
    Create detailed visualization of point selection.
    
    Shows:
    - All points (gray)
    - Original hull (red)
    - Selected points (blue)
    - New hull (green)
    - Threshold lines
    - Distance distributions
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    all_points = result.get('all_points', np.array([]))
    original_hull = result.get('original_hull', np.array([]))
    selected_points = result.get('selected_points', np.array([]))
    new_hull = result.get('new_hull', np.array([]))
    threshold = result.get('threshold_distance', 0)
    reference_dist = result.get('reference_distance', 0)
    n_selected = result.get('n_selected', 0)
    
    # Main ROC space plot
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Plot all points
    if len(all_points) > 0:
        ax1.scatter(all_points[:, 0], all_points[:, 1], 
                   c='lightgray', s=30, alpha=0.5, label='All points', zorder=1)
    
    # Plot original hull
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        ax1.plot(hull_sorted[:, 0], hull_sorted[:, 1], 
                'r-', linewidth=2.5, alpha=0.7, label='Original hull', zorder=3)
        ax1.scatter(original_hull[:, 0], original_hull[:, 1], 
                   c='red', s=150, marker='*', edgecolors='darkred', 
                   linewidths=1.5, label='Original hull points', zorder=4)
    
    # Plot selected points
    if len(selected_points) > 0:
        ax1.scatter(selected_points[:, 0], selected_points[:, 1], 
                   c='blue', s=100, alpha=0.7, edgecolors='darkblue',
                   linewidths=1.5, label=f'Selected points ({n_selected})', zorder=5)
    
    # Plot new hull
    if len(new_hull) > 0:
        new_hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
        ax1.plot(new_hull_sorted[:, 0], new_hull_sorted[:, 1], 
                'g--', linewidth=2.5, alpha=0.7, label='New hull', zorder=6)
        ax1.scatter(new_hull[:, 0], new_hull[:, 1], 
                   c='green', s=150, marker='s', edgecolors='darkgreen',
                   linewidths=1.5, label='New hull points', zorder=7)
    
    # Plot diagonal
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Diagonal', zorder=2)
    
    # Add threshold annotation
    threshold_text = f"Threshold: {threshold:.6f}\nReference: {reference_dist:.6f}"
    ax1.text(0.02, 0.98, threshold_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax1.set_title(f'{method_name} - {percentage}%\n{n_selected} points selected', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    
    # Distance distribution plot
    ax2 = fig.add_subplot(gs[0, 2])
    
    if method_name == 'Below Hull':
        # Show vertical distances below hull
        if 'avg_vertical_distance' in result and len(selected_points) > 0:
            # Calculate distances for all candidate points
            selected_dists = []
            if len(selected_points) > 0:
                for point in selected_points:
                    fpr, tpr = point
                    # Simple approximation
                    selected_dists.append(result.get('avg_vertical_distance', 0))
            
            if selected_dists:
                ax2.hist(selected_dists, bins=20, alpha=0.7, color='blue', edgecolor='black')
                ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
                ax2.set_xlabel('Vertical Distance Below Hull', fontsize=9)
                ax2.set_ylabel('Frequency', fontsize=9)
                ax2.set_title('Selected Points\nDistance Distribution', fontsize=10)
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
    else:  # Above Diagonal
        # Show diagonal distances
        if len(selected_points) > 0:
            diagonal_dists = selected_points[:, 1] - selected_points[:, 0]
            ax2.hist(diagonal_dists, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
            ax2.axvline(max_diagonal_dist, color='green', linestyle='--', linewidth=2, label=f'Max: {max_diagonal_dist:.6f}')
            ax2.set_xlabel('Diagonal Distance (TPR - FPR)', fontsize=9)
            ax2.set_ylabel('Frequency', fontsize=9)
            ax2.set_title('Selected Points\nDiagonal Distance', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
    
    # AUC comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    original_auc = result.get('original_auc', 0)
    new_auc = result.get('new_auc', 0)
    auc_reduction = result.get('auc_reduction', 0)
    auc_reduction_pct = result.get('auc_reduction_percentage', 0)
    
    categories = ['Original\nHull', 'New\nHull']
    values = [original_auc, new_auc]
    colors = ['red', 'green']
    
    bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('AUC', fontsize=10)
    ax3.set_title(f'AUC Comparison\nReduction: {auc_reduction:.4f} ({auc_reduction_pct:.2f}%)', 
                 fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Point counts comparison
    ax4 = fig.add_subplot(gs[1, 1])
    
    original_points = len(original_hull)
    new_points = len(new_hull)
    total_points = len(all_points)
    
    categories = ['All\nPoints', 'Original\nHull', 'Selected', 'New\nHull']
    values = [total_points, original_points, n_selected, new_points]
    colors = ['lightgray', 'red', 'blue', 'green']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Number of Points', fontsize=10)
    ax4.set_title('Point Counts', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom', fontsize=9)
    
    # Quality metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    metrics_text = f"""
    SELECTION METRICS
    ─────────────────────────
    Method: {method_name}
    Percentage: {percentage}%
    
    REFERENCE & THRESHOLD
    ─────────────────────────
    Max Diagonal Distance:
      {reference_dist:.6f}
    
    Threshold:
      {threshold:.6f}
    
    POINTS
    ─────────────────────────
    Total: {total_points}
    Original Hull: {original_points}
    Selected: {n_selected}
    New Hull: {new_points}
    
    AUC
    ─────────────────────────
    Original: {original_auc:.4f}
    New: {new_auc:.4f}
    Reduction: {auc_reduction:.4f}
    Reduction %: {auc_reduction_pct:.2f}%
    
    QUALITY
    ─────────────────────────
    Original Max: {result.get('original_max_quality', 0):.4f}
    New Max: {result.get('new_max_quality', 0):.4f}
    """
    
    ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle(f'{method_name} Selection Visualization - {percentage}% Threshold', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved visualization: {output_path.name}")


def run_enhanced_test(data_path, target_col, percentages=[1.0, 5.0, 10.0, 20.0], 
                     output_dir='./runs/enhanced_percentage_tests'):
    """Enhanced test with visualizations."""
    
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
    
    # Generate candidates
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
    
    # Calculate max diagonal distance
    diagonal_distances = above_diagonal[:, 1] - above_diagonal[:, 0]
    max_diagonal_distance = np.max(diagonal_distances)
    print(f"Max diagonal distance (reference): {max_diagonal_distance:.4f}")
    
    # Create output directory
    dataset_name = Path(data_path).stem
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test each percentage
    for pct in percentages:
        print(f"\n{'─'*80}")
        print(f"Testing {pct}% threshold")
        print(f"{'─'*80}")
        
        # Test function 4: Below Hull
        print(f"\n  Function 4 (Below Hull):")
        try:
            result_below = select_points_below_hull(
                all_points, 
                distance_percentage=pct, 
                return_details=True,
                exclude_hull_points=True
            )
            
            n_selected = result_below.get('n_selected', 0)
            print(f"    Selected: {n_selected} points")
            print(f"    New hull: {len(result_below.get('new_hull', []))} points")
            print(f"    AUC reduction: {result_below.get('auc_reduction_percentage', 0):.2f}%")
            
            # Create visualization
            viz_path = output_path / f'below_hull_{pct}pct.png'
            plot_selection_visualization(
                result_below, 'Below Hull', pct, 
                max_diagonal_distance, viz_path
            )
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        # Test function 5: Above Diagonal
        print(f"\n  Function 5 (Above Diagonal):")
        try:
            result_above = select_points_above_diagonal(
                all_points,
                distance_percentage=pct,
                return_details=True,
                exclude_hull_points=True
            )
            
            n_selected = result_above.get('n_selected', 0)
            print(f"    Selected: {n_selected} points")
            print(f"    New hull: {len(result_above.get('new_hull', []))} points")
            print(f"    AUC reduction: {result_above.get('auc_reduction_percentage', 0):.2f}%")
            
            # Create visualization
            viz_path = output_path / f'above_diagonal_{pct}pct.png'
            plot_selection_visualization(
                result_above, 'Above Diagonal', pct, 
                max_diagonal_distance, viz_path
            )
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print(f"\n✓ All visualizations saved to: {output_path}")


if __name__ == '__main__':
    # Test on multiple datasets
    datasets = [
        ('tests/adult.txt', 'target'),
        ('tests/ionosphere.txt', 'Attribute35'),
        ('tests/mushroom.txt', 'poisonous')
    ]
    
    print("\n" + "="*80)
    print("ENHANCED PERCENTAGE-BASED THRESHOLD TESTING")
    print("WITH VISUALIZATIONS")
    print("="*80)
    print("\nTesting percentages: 1%, 5%, 10%, 20%")
    print("="*80)
    
    for data_path, target_col in datasets:
        try:
            run_enhanced_test(
                data_path, 
                target_col,
                percentages=[1.0, 5.0, 10.0, 20.0]
            )
        except Exception as e:
            print(f"\n✗ Error testing {data_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: runs/enhanced_percentage_tests/")
    print("Check the PNG visualizations for detailed analysis.")
