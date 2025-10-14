"""
Test script for hull comparison functionality.

This demonstrates how to use the new hull comparison features
to analyze convex hull changes when removing hull points.
"""

import numpy as np
import matplotlib.pyplot as plt
from true_roc_search import (
    remove_hull_points_and_recalculate,
    plot_hull_comparison,
    demonstrate_hull_comparison
)

def generate_sample_roc_points(n_points=50, seed=42):
    """Generate sample ROC points for testing."""
    np.random.seed(seed)
    
    # Generate points with some structure
    # Most points above diagonal for ROC curve
    fpr = np.random.beta(2, 5, n_points)
    tpr = np.random.beta(5, 2, n_points)
    
    # Ensure some points are clearly above diagonal
    tpr = np.maximum(tpr, fpr + np.random.uniform(0.05, 0.2, n_points))
    
    # Clip to [0, 1]
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    
    points = np.column_stack([fpr, tpr])
    return points

def test_basic_hull_comparison():
    """Test basic hull comparison functionality."""
    print("=" * 60)
    print("TEST 1: Basic Hull Comparison")
    print("=" * 60)
    
    # Generate sample points
    points = generate_sample_roc_points(n_points=50)
    
    # Run hull comparison
    hull_data = demonstrate_hull_comparison(points, depth=1, output_dir='./runs/hull_test')
    
    return hull_data

def test_multiple_depths():
    """Test hull comparison across multiple depths (simulated)."""
    print("\n" + "=" * 60)
    print("TEST 2: Multiple Depth Comparison")
    print("=" * 60)
    
    depths = [1, 2, 3]
    all_hull_data = []
    
    for depth in depths:
        # Generate different point sets for each depth
        n_points = 30 + (depth * 20)  # More points at greater depth
        points = generate_sample_roc_points(n_points=n_points, seed=depth)
        
        print(f"\n--- Depth {depth} ({n_points} points) ---")
        hull_data = remove_hull_points_and_recalculate(points, return_details=True)
        hull_data['depth'] = depth
        all_hull_data.append(hull_data)
        
        # Create visualization
        plot_hull_comparison(hull_data, depth, 
                           output_path=f'./runs/hull_test/depth_{depth}_comparison.png')
    
    # Create summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: Hull Area Reduction Across Depths")
    print("=" * 60)
    print(f"{'Depth':<8} {'Original Area':<15} {'New Area':<15} {'Reduction %':<15}")
    print("-" * 60)
    
    for hull_data in all_hull_data:
        depth = hull_data['depth']
        orig_area = hull_data.get('original_hull_area', 0)
        new_area = hull_data.get('new_hull_area', 0)
        reduction = hull_data.get('reduction_percentage', 0)
        print(f"{depth:<8} {orig_area:<15.4f} {new_area:<15.4f} {reduction:<15.1f}")
    
    return all_hull_data

def test_edge_cases():
    """Test edge cases for hull comparison."""
    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)
    
    # Case 1: Very few points
    print("\nCase 1: Only 3 points")
    points_few = np.array([[0.1, 0.3], [0.4, 0.7], [0.6, 0.9]])
    hull_data_few = remove_hull_points_and_recalculate(points_few, return_details=True)
    print(f"  Original hull points: {len(hull_data_few['original_hull'])}")
    print(f"  New hull points: {len(hull_data_few.get('new_hull', []))}")
    
    # Case 2: Points mostly below diagonal
    print("\nCase 2: Points below diagonal")
    fpr = np.random.uniform(0.5, 1.0, 20)
    tpr = np.random.uniform(0.0, 0.5, 20)
    points_below = np.column_stack([fpr, tpr])
    hull_data_below = remove_hull_points_and_recalculate(points_below, return_details=True)
    print(f"  Points above diagonal: {len(hull_data_below.get('all_points', []))}")
    print(f"  Original hull points: {len(hull_data_below['original_hull'])}")
    
    # Case 3: Many points (stress test)
    print("\nCase 3: Many points (200)")
    points_many = generate_sample_roc_points(n_points=200, seed=123)
    hull_data_many = remove_hull_points_and_recalculate(points_many, return_details=True)
    print(f"  Total points: {len(points_many)}")
    print(f"  Points above diagonal: {len(hull_data_many.get('all_points', []))}")
    print(f"  Original hull points: {len(hull_data_many['original_hull'])}")
    print(f"  New hull points: {len(hull_data_many.get('new_hull', []))}")
    print(f"  Area reduction: {hull_data_many.get('reduction_percentage', 0):.1f}%")

def create_comparison_visualization(all_hull_data, output_path='./runs/hull_test/area_reduction_plot.png'):
    """Create a visualization comparing hull areas across depths."""
    depths = [hd['depth'] for hd in all_hull_data]
    original_areas = [hd.get('original_hull_area', 0) for hd in all_hull_data]
    new_areas = [hd.get('new_hull_area', 0) for hd in all_hull_data]
    reductions = [hd.get('reduction_percentage', 0) for hd in all_hull_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Hull areas
    width = 0.35
    x = np.arange(len(depths))
    
    ax1.bar(x - width/2, original_areas, width, label='Original Hull', alpha=0.8, color='red')
    ax1.bar(x + width/2, new_areas, width, label='New Hull', alpha=0.8, color='purple')
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Hull Area')
    ax1.set_title('Convex Hull Area Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(depths)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reduction percentage
    ax2.bar(depths, reductions, color='green', alpha=0.7)
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('Reduction (%)')
    ax2.set_title('Hull Area Reduction Percentage')
    ax2.set_xticks(depths)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure directory exists
    from pathlib import Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved area reduction plot to: {output_path}")
    plt.close()

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HULL COMPARISON FUNCTIONALITY TESTS")
    print("=" * 60)
    
    # Test 1: Basic functionality
    test_basic_hull_comparison()
    
    # Test 2: Multiple depths
    all_hull_data = test_multiple_depths()
    
    # Test 3: Edge cases
    test_edge_cases()
    
    # Create summary visualization
    if all_hull_data:
        create_comparison_visualization(all_hull_data)
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\nResults saved to: ./runs/hull_test/")
    print("\nKey files:")
    print("  - hull_comparison_demo_depth_*.png : Individual depth comparisons")
    print("  - depth_*_comparison.png : Detailed depth visualizations")
    print("  - area_reduction_plot.png : Summary comparison")

if __name__ == '__main__':
    main()
