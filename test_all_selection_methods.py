"""
Comprehensive comparison of all four point selection methods:
1. Closest to Hull (Euclidean distance - horizontal)
2. Furthest from Diagonal (TPR-FPR distance - diagonal)
3. Below Hull (Vertical distance below hull)
4. Above Diagonal (TPR magnitude - vertical)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import all selection functions
from true_roc_search import (
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal,
    select_points_below_hull,
    select_points_above_diagonal
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


def plot_all_methods_comparison(points, n_points, exclude_hull, output_dir):
    """
    Create a comprehensive 4x3 grid comparing all selection methods.
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    
    # Get results from all methods
    results = {
        'closest': select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull),
        'furthest': select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull),
        'below': select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull),
        'above': select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
    }
    
    original_hull = results['closest']['original_hull']
    
    method_configs = [
        ('closest', 'Closest to Hull', 'green', 'lime'),
        ('furthest', 'Furthest from Diagonal', 'purple', 'violet'),
        ('below', 'Below Hull', 'darkblue', 'cyan'),
        ('above', 'Above Diagonal', 'darkred', 'orange')
    ]
    
    for row, (method_key, method_name, hull_color, point_color) in enumerate(method_configs):
        result = results[method_key]
        selected = result['selected_points']
        new_hull = result['new_hull']
        
        # Column 1: Original hull
        axes[row, 0].plot(points[:, 0], points[:, 1], 'bo', alpha=0.3, 
                         label='All points', markersize=4)
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
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_xlim(0, 1)
        axes[row, 0].set_ylim(0, 1)
        
        # Column 2: Selected points
        axes[row, 1].plot(selected[:, 0], selected[:, 1], 'o', color=point_color,
                         alpha=0.6, label=f'{n_points} selected', markersize=6)
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
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.3)
        axes[row, 1].set_xlim(0, 1)
        axes[row, 1].set_ylim(0, 1)
        
        # Column 3: Overlay
        axes[row, 2].plot(points[:, 0], points[:, 1], 'bo', alpha=0.2, 
                         label='All points', markersize=3)
        if len(original_hull) > 0:
            hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
            axes[row, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', 
                            linewidth=2, alpha=0.7, label='Original')
        if len(new_hull) > 0:
            hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
            axes[row, 2].plot(hull_sorted[:, 0], hull_sorted[:, 1], color=hull_color,
                            linewidth=2, linestyle='--', alpha=0.7, label=f'New ({method_name})')
        axes[row, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
        axes[row, 2].set_xlabel('FPR')
        axes[row, 2].set_ylabel('TPR')
        
        auc_change = result.get('auc_reduction_percentage', 0)
        orig_auc = result.get('original_auc', 0)
        new_auc = result.get('new_auc', 0)
        axes[row, 2].set_title(f'{method_name} Overlay\n'
                              f'AUC: {orig_auc:.4f}→{new_auc:.4f} ({auc_change:+.1f}%)')
        axes[row, 2].legend()
        axes[row, 2].grid(True, alpha=0.3)
        axes[row, 2].set_xlim(0, 1)
        axes[row, 2].set_ylim(0, 1)
    
    exclude_text = "exclude_hull=True" if exclude_hull else "include_hull=True"
    plt.suptitle(f'All Selection Methods Comparison: {n_points} Points ({exclude_text})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    suffix = "exclude" if exclude_hull else "include"
    output_path = Path(output_dir) / f'all_methods_comparison_{n_points}pts_{suffix}.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    plt.show()
    plt.close()


def test_all_methods():
    """Test all four selection methods and compare results."""
    
    print("=== Comprehensive Comparison of All Selection Methods ===\n")
    
    # Generate test data
    points = generate_test_points(n_points=50, seed=42)
    print(f"Generated {len(points)} test points above diagonal\n")
    
    # Create output directory
    output_dir = Path('./runs/all_methods_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    test_n = [10, 20]
    
    for n_points in test_n:
        print(f"\n{'='*80}")
        print(f"Testing with {n_points} selected points")
        print(f"{'='*80}")
        
        # Test both include and exclude modes
        for exclude_hull in [False, True]:
            mode_text = "EXCLUDE" if exclude_hull else "INCLUDE"
            print(f"\n{'─'*80}")
            print(f"Mode: {mode_text} hull points")
            print(f"{'─'*80}")
            
            # Get results from all methods
            closest = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            furthest = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            below = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            above = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            
            print(f"\n{'Method':<30} {'Selected':<10} {'New Hull':<10} {'AUC Change %':<15} {'Quality Δ':<12}")
            print("-" * 80)
            print(f"{'Closest to Hull':<30} {len(closest['selected_points']):<10} "
                  f"{len(closest['new_hull']):<10} "
                  f"{closest.get('auc_reduction_percentage', 0):<15.2f} "
                  f"{closest.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Furthest from Diagonal':<30} {len(furthest['selected_points']):<10} "
                  f"{len(furthest['new_hull']):<10} "
                  f"{furthest.get('auc_reduction_percentage', 0):<15.2f} "
                  f"{furthest.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Below Hull':<30} {len(below['selected_points']):<10} "
                  f"{len(below['new_hull']):<10} "
                  f"{below.get('auc_reduction_percentage', 0):<15.2f} "
                  f"{below.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Above Diagonal':<30} {len(above['selected_points']):<10} "
                  f"{len(above['new_hull']):<10} "
                  f"{above.get('auc_reduction_percentage', 0):<15.2f} "
                  f"{above.get('quality_reduction', 0):<12.4f}")
            
            # Create visualization
            plot_all_methods_comparison(points, n_points, exclude_hull, output_dir)
    
    # Create final summary table
    print(f"\n{'='*80}")
    print("Final Summary: All Methods at 10 Points")
    print(f"{'='*80}")
    print(f"{'Method':<35} {'Mode':<20} {'AUC Δ%':<10} {'Hull Pts':<10} {'Quality Δ':<12}")
    print("-" * 80)
    
    for n_points in [10]:
        for exclude_hull in [False, True]:
            mode = "Exclude Hull" if exclude_hull else "Include Hull"
            
            closest = select_closest_points_to_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            furthest = select_furthest_points_from_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            below = select_points_below_hull(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            above = select_points_above_diagonal(points, n_points, return_details=True, exclude_hull_points=exclude_hull)
            
            print(f"{'Closest to Hull':<35} {mode:<20} "
                  f"{closest.get('auc_reduction_percentage', 0):<10.2f} "
                  f"{len(closest['new_hull']):<10} "
                  f"{closest.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Furthest from Diagonal':<35} {mode:<20} "
                  f"{furthest.get('auc_reduction_percentage', 0):<10.2f} "
                  f"{len(furthest['new_hull']):<10} "
                  f"{furthest.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Below Hull':<35} {mode:<20} "
                  f"{below.get('auc_reduction_percentage', 0):<10.2f} "
                  f"{len(below['new_hull']):<10} "
                  f"{below.get('quality_reduction', 0):<12.4f}")
            
            print(f"{'Above Diagonal':<35} {mode:<20} "
                  f"{above.get('auc_reduction_percentage', 0):<10.2f} "
                  f"{len(above['new_hull']):<10} "
                  f"{above.get('quality_reduction', 0):<12.4f}")
            
            print("-" * 80)
    
    print(f"\n✓ All tests completed!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"\nSelection Method Characteristics:")
    print(f"  • Closest to Hull: Euclidean distance (horizontal) - preserves hull shape")
    print(f"  • Furthest from Diagonal: TPR-FPR distance - emphasizes quality difference")
    print(f"  • Below Hull: Vertical distance below - selects worst performers")
    print(f"  • Above Diagonal: TPR magnitude - selects best performers")


if __name__ == "__main__":
    test_all_methods()
