"""
Test script for the new point selection functions:
- select_closest_points_to_hull
- select_furthest_points_from_diagonal
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import the functions from true_roc_search
from true_roc_search import (
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal,
    plot_hull_comparison
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


def plot_selection_comparison(original_data, closest_data, furthest_data, n_points, output_dir):
    """
    Create a comprehensive comparison plot showing both selection methods.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data for closest points selection
    all_points = original_data['all_points']
    original_hull = original_data['original_hull']
    closest_selected = closest_data['selected_points']
    closest_hull = closest_data['new_hull']
    
    # Extract data for furthest points selection
    furthest_selected = furthest_data['selected_points']
    furthest_hull = furthest_data['new_hull']
    
    # Row 1: Closest points to hull
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
    
    # Plot 2: Selected closest points
    axes[0, 1].plot(closest_selected[:, 0], closest_selected[:, 1], 'go', 
                    alpha=0.5, label=f'{n_points} closest points', markersize=6)
    if len(closest_hull) > 0:
        hull_sorted = closest_hull[np.argsort(closest_hull[:, 0])]
        axes[0, 1].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', 
                       linewidth=2, label='New hull')
        axes[0, 1].scatter(closest_hull[:, 0], closest_hull[:, 1], 
                          c='purple', s=100, marker='*', zorder=5, edgecolors='black')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 1].set_title(f'Closest {n_points} Points to Hull')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: Overlay comparison for closest
    axes[0, 2].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.2, 
                    label='All points', markersize=3)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[0, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, alpha=0.7, label='Original hull')
    if len(closest_hull) > 0:
        hull_sorted = closest_hull[np.argsort(closest_hull[:, 0])]
        axes[0, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'purple', 
                       linewidth=2, linestyle='--', alpha=0.7, label='New hull (closest)')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[0, 2].set_xlabel('FPR')
    axes[0, 2].set_ylabel('TPR')
    auc_change = closest_data.get('auc_reduction_percentage', 0)
    axes[0, 2].set_title(f'Overlay (Closest)\nAUC change: {auc_change:.1f}%')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    
    # Row 2: Furthest points from diagonal
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
    
    # Plot 5: Selected furthest points
    axes[1, 1].plot(furthest_selected[:, 0], furthest_selected[:, 1], 'mo', 
                    alpha=0.5, label=f'{n_points} furthest points', markersize=6)
    if len(furthest_hull) > 0:
        hull_sorted = furthest_hull[np.argsort(furthest_hull[:, 0])]
        axes[1, 1].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'orange', 
                       linewidth=2, label='New hull')
        axes[1, 1].scatter(furthest_hull[:, 0], furthest_hull[:, 1], 
                          c='orange', s=100, marker='*', zorder=5, edgecolors='black')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1, 1].set_xlabel('FPR')
    axes[1, 1].set_ylabel('TPR')
    axes[1, 1].set_title(f'Furthest {n_points} Points from Diagonal')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    
    # Plot 6: Overlay comparison for furthest
    axes[1, 2].plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.2, 
                    label='All points', markersize=3)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        axes[1, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                       linewidth=2, alpha=0.7, label='Original hull')
    if len(furthest_hull) > 0:
        hull_sorted = furthest_hull[np.argsort(furthest_hull[:, 0])]
        axes[1, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'orange', 
                       linewidth=2, linestyle='--', alpha=0.7, label='New hull (furthest)')
    axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    axes[1, 2].set_xlabel('FPR')
    axes[1, 2].set_ylabel('TPR')
    auc_change = furthest_data.get('auc_reduction_percentage', 0)
    axes[1, 2].set_title(f'Overlay (Furthest)\nAUC change: {auc_change:.1f}%')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    
    plt.suptitle(f'Point Selection Comparison: {n_points} Points', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / f'point_selection_comparison_{n_points}pts.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    plt.show()
    plt.close()


def test_point_selection():
    """Test both point selection methods."""
    
    print("=== Testing Point Selection Functions ===\n")
    
    # Generate test data
    points = generate_test_points(n_points=50, seed=42)
    print(f"Generated {len(points)} test points above diagonal\n")
    
    # Create output directory
    output_dir = Path('./runs/point_selection_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test with different numbers of selected points
    test_configs = [5, 10, 20, 50]
    
    for n_points in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing with {n_points} selected points")
        print(f"{'='*70}")
        
        # Test 1: Closest points to hull (DEFAULT - may include hull points)
        print(f"\n--- Closest {n_points} Points to Hull (include_hull=True, default) ---")
        closest_data = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=False)
        
        print(f"Original hull: {len(closest_data['original_hull'])} points")
        print(f"Selected points: {len(closest_data['selected_points'])} points")
        print(f"New hull: {len(closest_data['new_hull'])} points")
        print(f"Original AUC: {closest_data.get('original_auc', 0):.4f}")
        print(f"New AUC: {closest_data.get('new_auc', 0):.4f}")
        print(f"AUC change: {closest_data.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Quality change: {closest_data.get('quality_reduction', 0):.4f}")
        
        # Test 1b: Closest points to hull (EXCLUDE hull points - forces change)
        print(f"\n--- Closest {n_points} Points to Hull (exclude_hull=True, forced change) ---")
        closest_excl_data = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"Original hull: {len(closest_excl_data['original_hull'])} points")
        print(f"Selected points: {len(closest_excl_data['selected_points'])} points")
        print(f"New hull: {len(closest_excl_data['new_hull'])} points")
        print(f"Original AUC: {closest_excl_data.get('original_auc', 0):.4f}")
        print(f"New AUC: {closest_excl_data.get('new_auc', 0):.4f}")
        print(f"AUC change: {closest_excl_data.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Quality change: {closest_excl_data.get('quality_reduction', 0):.4f}")
        
        # Test 2: Furthest points from diagonal (DEFAULT)
        print(f"\n--- Furthest {n_points} Points from Diagonal (include_hull=True, default) ---")
        furthest_data = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=False)
        
        print(f"Original hull: {len(furthest_data['original_hull'])} points")
        print(f"Selected points: {len(furthest_data['selected_points'])} points")
        print(f"New hull: {len(furthest_data['new_hull'])} points")
        print(f"Original AUC: {furthest_data.get('original_auc', 0):.4f}")
        print(f"New AUC: {furthest_data.get('new_auc', 0):.4f}")
        print(f"AUC change: {furthest_data.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Quality change: {furthest_data.get('quality_reduction', 0):.4f}")
        print(f"Avg distance from diagonal: {furthest_data.get('avg_distance_from_diagonal', 0):.4f}")
        print(f"Max distance from diagonal: {furthest_data.get('max_distance_from_diagonal', 0):.4f}")
        
        # Test 2b: Furthest points from diagonal (EXCLUDE hull points)
        print(f"\n--- Furthest {n_points} Points from Diagonal (exclude_hull=True, forced change) ---")
        furthest_excl_data = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"Original hull: {len(furthest_excl_data['original_hull'])} points")
        print(f"Selected points: {len(furthest_excl_data['selected_points'])} points")
        print(f"New hull: {len(furthest_excl_data['new_hull'])} points")
        print(f"Original AUC: {furthest_excl_data.get('original_auc', 0):.4f}")
        print(f"New AUC: {furthest_excl_data.get('new_auc', 0):.4f}")
        print(f"AUC change: {furthest_excl_data.get('auc_reduction_percentage', 0):.2f}%")
        print(f"Quality change: {furthest_excl_data.get('quality_reduction', 0):.4f}")
        print(f"Avg distance from diagonal: {furthest_excl_data.get('avg_distance_from_diagonal', 0):.4f}")
        
        # Create comparison visualization (use exclude_hull=True versions for clearer differences)
        # Use the first test as the "original" reference
        original_data = {
            'all_points': points,
            'original_hull': closest_data['original_hull']
        }
        
        plot_selection_comparison(
            original_data, 
            closest_excl_data,  # Use excluded version for clearer visualization
            furthest_excl_data,  # Use excluded version for clearer visualization
            n_points, 
            output_dir
        )
    
    # Create summary comparison table
    print(f"\n{'='*70}")
    print("Summary Comparison Table")
    print(f"{'='*70}")
    print(f"{'Method':<40} {'N':<5} {'New Hull':<10} {'AUC':<8} {'AUC Δ%':<10} {'Quality Δ':<12}")
    print("-" * 90)
    
    for n_points in test_configs:
        closest_incl = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=False)
        closest_excl = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=True)
        furthest_incl = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=False)
        furthest_excl = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=True)
        
        print(f"{'Closest to Hull (include hull)':<40} {n_points:<5} "
              f"{len(closest_incl['new_hull']):<10} "
              f"{closest_incl.get('new_auc', 0):<8.4f} "
              f"{closest_incl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{closest_incl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Closest to Hull (exclude hull)':<40} {n_points:<5} "
              f"{len(closest_excl['new_hull']):<10} "
              f"{closest_excl.get('new_auc', 0):<8.4f} "
              f"{closest_excl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{closest_excl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Furthest from Diagonal (include hull)':<40} {n_points:<5} "
              f"{len(furthest_incl['new_hull']):<10} "
              f"{furthest_incl.get('new_auc', 0):<8.4f} "
              f"{furthest_incl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{furthest_incl.get('quality_reduction', 0):<12.4f}")
        
        print(f"{'Furthest from Diagonal (exclude hull)':<40} {n_points:<5} "
              f"{len(furthest_excl['new_hull']):<10} "
              f"{furthest_excl.get('new_auc', 0):<8.4f} "
              f"{furthest_excl.get('auc_reduction_percentage', 0):<10.2f} "
              f"{furthest_excl.get('quality_reduction', 0):<12.4f}")
        
        print("-" * 90)
    
    print(f"\nAll tests completed! Results saved to: {output_dir}")
    print(f"\nNOTE: Plots show 'exclude_hull=True' versions for clearer curve differences.")
    print(f"      Use 'exclude_hull_points=True' to force different curves.")
    print(f"      Use 'exclude_hull_points=False' (default) to allow hull points in selection.")


if __name__ == "__main__":
    test_point_selection()
