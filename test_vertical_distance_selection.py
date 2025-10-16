"""
Test script for vertical distance-based point selection functions:
- select_points_below_hull
- select_points_above_diagonal
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the functions from true_roc_search
from true_roc_search import (
    select_points_below_hull,
    select_points_above_diagonal,
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal
)


def generate_test_points(n_points=50, seed=42):
    """Generate test ROC points for demonstration."""
    np.random.seed(seed)
    
    # Generate points above diagonal with varying quality
    fpr = np.random.uniform(0, 0.8, n_points)
    tpr = fpr + np.random.uniform(0.1, 0.4, n_points)
    
    # Clip to valid ROC space
    tpr = np.clip(tpr, 0, 1)
    
    points = np.column_stack([fpr, tpr])
    
    # Filter to only points above diagonal
    points = points[points[:, 1] > points[:, 0]]
    
    return points


def plot_vertical_distance_comparison(original_data, below_hull_data, above_diag_data, n_points, output_dir):
    """
    Create a comprehensive comparison plot showing both vertical distance selection methods.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    all_points = original_data['all_points']
    original_hull = original_data['original_hull']
    
    # Below hull data
    below_selected = below_hull_data['selected_points']
    below_hull = below_hull_data['new_hull']
    
    # Above diagonal data
    above_selected = above_diag_data['selected_points']
    above_hull = above_diag_data['new_hull']
    
    # Row 1: Below Hull Selection
    # Plot 1: Original hull
    axes[0, 0].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.3, 
                    label='All points', markersize=4)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[0, 0].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, label='Original hull')
        axes[0, 0].scatter(original_hull[:, 0], original_hull[:, 1], 
                          c='red', s=100, marker='*', zorder=5, edgecolors='black')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0, 0].set_xlabel('FPR')
    axes[0, 0].set_ylabel('TPR')
    axes[0, 0].set_title('Original Hull')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Selected points below hull
    axes[0, 1].plot(below_selected[:, 0], below_selected[:, 1], 'go', 
                    alpha=0.5, label=f'{n_points} below hull', markersize=6)
    if len(below_hull) > 0:
        hull_sorted = below_hull[np.argsort(below_hull[:, 0])]
        axes[0, 1].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', 
                       linewidth=2, label='New hull')
        axes[0, 1].scatter(below_hull[:, 0], below_hull[:, 1], 
                          c='purple', s=100, marker='*', zorder=5, edgecolors='black')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 1].set_title(f'{n_points} Points Below Hull')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Overlay comparison for below hull
    axes[0, 2].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.2, 
                    label='All points', markersize=3)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[0, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, alpha=0.7, label='Original hull')
    if len(below_hull) > 0:
        hull_sorted = below_hull[np.argsort(below_hull[:, 0])]
        axes[0, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', 
                       linewidth=2, linestyle='--', alpha=0.7, label='New hull (below)')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0, 2].set_xlabel('FPR')
    axes[0, 2].set_ylabel('TPR')
    auc_change = below_hull_data.get('auc_reduction_percentage', 0)
    axes[0, 2].set_title(f'Overlay (Below Hull)\nAUC change: {auc_change:.1f}%')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    
    # Row 2: Above Diagonal Selection
    # Plot 4: Original hull (same as plot 1)
    axes[1, 0].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.3, 
                    label='All points', markersize=4)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[1, 0].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, label='Original hull')
        axes[1, 0].scatter(original_hull[:, 0], original_hull[:, 1], 
                          c='red', s=100, marker='*', zorder=5, edgecolors='black')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1, 0].set_xlabel('FPR')
    axes[1, 0].set_ylabel('TPR')
    axes[1, 0].set_title('Original Hull')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 5: Selected points above diagonal
    axes[1, 1].plot(above_selected[:, 0], above_selected[:, 1], 'mo', 
                    alpha=0.5, label=f'{n_points} above diagonal', markersize=6)
    if len(above_hull) > 0:
        hull_sorted = above_hull[np.argsort(above_hull[:, 0])]
        axes[1, 1].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'orange', 
                       linewidth=2, label='New hull')
        axes[1, 1].scatter(above_hull[:, 0], above_hull[:, 1], 
                          c='orange', s=100, marker='*', zorder=5, edgecolors='black')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1, 1].set_xlabel('FPR')
    axes[1, 1].set_ylabel('TPR')
    axes[1, 1].set_title(f'{n_points} Points Highest Above Diagonal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    
    # Plot 6: Overlay comparison for above diagonal
    axes[1, 2].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.2, 
                    label='All points', markersize=3)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[1, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, alpha=0.7, label='Original hull')
    if len(above_hull) > 0:
        hull_sorted = above_hull[np.argsort(above_hull[:, 0])]
        axes[1, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'orange', 
                       linewidth=2, linestyle='--', alpha=0.7, label='New hull (above diag)')
    axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1, 2].set_xlabel('FPR')
    axes[1, 2].set_ylabel('TPR')
    auc_change = above_diag_data.get('auc_reduction_percentage', 0)
    axes[1, 2].set_title(f'Overlay (Above Diagonal)\nAUC change: {auc_change:.1f}%')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    
    plt.suptitle(f'Vertical Distance Selection Comparison: {n_points} Points', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'vertical_distance_comparison_{n_points}pts.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    plt.show()
    plt.close()


def test_vertical_distance_selection():
    """Test both vertical distance selection methods."""
    
    print("=== Testing Vertical Distance Selection Functions ===\n")
    
    # Generate test data
    points = generate_test_points(n_points=50, seed=42)
    print(f"Generated {len(points)} test points above diagonal\n")
    
    # Create output directory
    output_dir = Path('./runs/vertical_distance_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with different numbers of selected points
    test_configs = [5, 10, 15, 20]
    # 1, 2, 5, 10 %s
    # 100 - (1, 5, 10, 15) %s 
    for n_points in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing with {n_points} selected points")
        print(f"{'='*70}")
        
        # Test 1: Points below hull (DEFAULT)
        print(f"\n--- {n_points} Points Below Hull (include_hull=True, default) ---")
        below_incl = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=False)
        
        print(f"Original hull: {len(below_incl['original_hull'])} points")
        print(f"Selected points: {len(below_incl['selected_points'])} points")
        print(f"New hull: {len(below_incl['new_hull'])} points")
        print(f"Original AUC: {below_incl.get('original_auc', 0):.4f}")
        print(f"New AUC: {below_incl.get('new_auc', 0):.4f}")
        print(f"AUC change: {below_incl.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Avg vertical distance: {below_incl.get('avg_vertical_distance', 0):.4f}")
        
        # Test 1b: Points below hull (EXCLUDE hull)
        print(f"\n--- {n_points} Points Below Hull (exclude_hull=True, forced change) ---")
        below_excl = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"Original hull: {len(below_excl['original_hull'])} points")
        print(f"Selected points: {len(below_excl['selected_points'])} points")
        print(f"New hull: {len(below_excl['new_hull'])} points")
        print(f"Original AUC: {below_excl.get('original_auc', 0):.4f}")
        print(f"New AUC: {below_excl.get('new_auc', 0):.4f}")
        print(f"AUC change: {below_excl.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Avg vertical distance: {below_excl.get('avg_vertical_distance', 0):.4f}")
        
        # Test 2: Points above diagonal (DEFAULT)
        print(f"\n--- {n_points} Points Above Diagonal (include_hull=True, default) ---")
        above_incl = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=False)
        
        print(f"Original hull: {len(above_incl['original_hull'])} points")
        print(f"Selected points: {len(above_incl['selected_points'])} points")
        print(f"New hull: {len(above_incl['new_hull'])} points")
        print(f"Original AUC: {above_incl.get('original_auc', 0):.4f}")
        print(f"New AUC: {above_incl.get('new_auc', 0):.4f}")
        print(f"AUC change: {above_incl.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Avg TPR selected: {above_incl.get('avg_tpr_selected', 0):.4f}")
        print(f"Max TPR selected: {above_incl.get('max_tpr_selected', 0):.4f}")
        
        # Test 2b: Points above diagonal (EXCLUDE hull)
        print(f"\n--- {n_points} Points Above Diagonal (exclude_hull=True, forced change) ---")
        above_excl = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"Original hull: {len(above_excl['original_hull'])} points")
        print(f"Selected points: {len(above_excl['selected_points'])} points")
        print(f"New hull: {len(above_excl['new_hull'])} points")
        print(f"Original AUC: {above_excl.get('original_auc', 0):.4f}")
        print(f"New AUC: {above_excl.get('new_auc', 0):.4f}")
        print(f"AUC change: {above_excl.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Avg TPR selected: {above_excl.get('avg_tpr_selected', 0):.4f}")
        
        # Create comparison visualization (use exclude_hull=True for clearer differences)
        original_data = {
            'all_points': points,
            'original_hull': below_incl['original_hull']
        }
        
        plot_vertical_distance_comparison(
            original_data,
            below_excl,  # Use excluded version
            above_excl,  # Use excluded version
            n_points,
            output_dir
        )
    
    # Create summary comparison table
    print(f"\n{'='*70}")
    print("Summary Comparison Table - Vertical Distance Selection")
    print(f"{'='*70}")
    print(f"{'Method':<45} {'N':<5} {'New Hull':<10} {'AUC':<8} {'AUC Δ%':<10} {'Quality Δ':<12}")
    print("-" * 95)
    
    for n_points in test_configs:
        below_incl = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=False)
        below_excl = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=True)
        above_incl = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=False)
        above_excl = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"{'Below Hull (include hull)':<45} {n_points:<5} "
              f"{len(below_incl['new_hull']):<10} "
              f"{below_incl.get('new_auc', 0):<8.4f} "
              f"{below_incl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{below_incl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Below Hull (exclude hull)':<45} {n_points:<5} "
              f"{len(below_excl['new_hull']):<10} "
              f"{below_excl.get('new_auc', 0):<8.4f} "
              f"{below_excl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{below_excl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Above Diagonal (include hull)':<45} {n_points:<5} "
              f"{len(above_incl['new_hull']):<10} "
              f"{above_incl.get('new_auc', 0):<8.4f} "
              f"{above_incl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{above_incl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Above Diagonal (exclude hull)':<45} {n_points:<5} "
              f"{len(above_excl['new_hull']):<10} "
              f"{above_excl.get('new_auc', 0):<8.4f} "
              f"{above_excl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{above_excl.get('quality_reduction', 0):<12.4f}")
        
        print("-" * 95)
    
    print(f"\nAll tests completed! Results saved to: {output_dir}")
    print(f"\nNOTE: Plots show 'exclude_hull=True' versions for clearer curve differences.")
    print(f"      - Below Hull: Selects points with smallest vertical distance below hull")
    print(f"      - Above Diagonal: Selects points with highest TPR (largest y-distance above diagonal)")


if __name__ == "__main__":
    test_vertical_distance_selection()
