"""
Visual demonstration of hull comparison functionality.
Creates a simple example showing the concept.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def create_concept_diagram():
    """Create a visual diagram explaining the hull comparison concept."""
    
    # Generate sample ROC points
    np.random.seed(42)
    n_points = 30
    fpr = np.random.beta(2, 5, n_points)
    tpr = np.random.beta(5, 2, n_points)
    tpr = np.maximum(tpr, fpr + np.random.uniform(0.05, 0.2, n_points))
    fpr = np.clip(fpr, 0, 1)
    tpr = np.clip(tpr, 0, 1)
    
    # Filter to points above diagonal
    points = np.column_stack([fpr, tpr])
    above_diagonal = points[points[:, 1] > points[:, 0]]
    
    # Calculate original hull
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    hull = ConvexHull(extended_points)
    hull_indices = set(hull.vertices)
    original_hull_indices = [i - 1 for i in hull_indices if 1 <= i <= len(above_diagonal)]
    original_hull = above_diagonal[original_hull_indices]
    
    # Remove hull points
    remaining_indices = [i for i in range(len(above_diagonal)) if i not in original_hull_indices]
    remaining_points = above_diagonal[remaining_indices]
    
    # Calculate new hull
    if len(remaining_points) >= 3:
        new_extended = np.vstack([[0, 0], remaining_points, [1, 1]])
        new_hull = ConvexHull(new_extended)
        new_hull_indices = set(new_hull.vertices)
        new_hull_points_indices = [i - 1 for i in new_hull_indices if 1 <= i <= len(remaining_points)]
        new_hull_points = remaining_points[new_hull_points_indices]
    else:
        new_hull_points = np.array([])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Title
    fig.suptitle('Hull Comparison: Concept Overview', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Original situation
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(above_diagonal[:, 0], above_diagonal[:, 1], c='blue', s=80, alpha=0.6, label='All points')
    ax1.scatter(original_hull[:, 0], original_hull[:, 1], c='red', s=150, marker='*', 
               edgecolors='black', linewidths=1.5, label='Hull points', zorder=5)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
    ax1.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2, alpha=0.7)
    ax1.set_xlabel('FPR', fontsize=11)
    ax1.set_ylabel('TPR', fontsize=11)
    ax1.set_title('Step 1: Original Points & Hull', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.text(0.5, -0.2, f'{len(above_diagonal)} total points\n{len(original_hull)} on hull', 
            ha='center', transform=ax1.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Remove hull points (arrow diagram)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(above_diagonal[:, 0], above_diagonal[:, 1], c='lightgray', s=80, alpha=0.3)
    ax2.scatter(original_hull[:, 0], original_hull[:, 1], c='red', s=150, marker='*', 
               edgecolors='black', linewidths=1.5, zorder=5)
    # Draw X marks on hull points
    for point in original_hull:
        ax2.plot(point[0], point[1], 'kx', markersize=20, markeredgewidth=3)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('FPR', fontsize=11)
    ax2.set_ylabel('TPR', fontsize=11)
    ax2.set_title('Step 2: Remove Hull Points', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.text(0.5, 0.5, 'REMOVE\nHULL POINTS', ha='center', va='center', 
            fontsize=18, fontweight='bold', color='red', alpha=0.7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.5, -0.2, f'Removing {len(original_hull)} points', 
            ha='center', transform=ax2.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Plot 3: Remaining points
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(remaining_points[:, 0], remaining_points[:, 1], c='green', s=80, 
               alpha=0.6, label='Remaining points')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax3.set_xlabel('FPR', fontsize=11)
    ax3.set_ylabel('TPR', fontsize=11)
    ax3.set_title('Step 3: Remaining Points', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.text(0.5, -0.2, f'{len(remaining_points)} remaining points', 
            ha='center', transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 4: New hull calculation
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(remaining_points[:, 0], remaining_points[:, 1], c='green', s=80, alpha=0.6)
    if len(new_hull_points) > 0:
        ax4.scatter(new_hull_points[:, 0], new_hull_points[:, 1], c='purple', s=150, 
                   marker='*', edgecolors='black', linewidths=1.5, label='New hull points', zorder=5)
        new_hull_sorted = new_hull_points[np.argsort(new_hull_points[:, 0])]
        ax4.plot(new_hull_sorted[:, 0], new_hull_sorted[:, 1], 'purple', linewidth=2, 
                alpha=0.7, label='New hull')
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax4.set_xlabel('FPR', fontsize=11)
    ax4.set_ylabel('TPR', fontsize=11)
    ax4.set_title('Step 4: Calculate New Hull', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-0.05, 1.05)
    ax4.set_ylim(-0.05, 1.05)
    ax4.text(0.5, -0.2, f'{len(new_hull_points)} new hull points', 
            ha='center', transform=ax4.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.5))
    
    # Plot 5: Comparison
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(above_diagonal[:, 0], above_diagonal[:, 1], c='blue', s=50, alpha=0.3)
    hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
    ax5.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=3, alpha=0.7, label='Original hull')
    if len(new_hull_points) > 0:
        new_hull_sorted = new_hull_points[np.argsort(new_hull_points[:, 0])]
        ax5.plot(new_hull_sorted[:, 0], new_hull_sorted[:, 1], 'purple', linewidth=3, 
                linestyle='--', alpha=0.7, label='New hull')
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax5.set_xlabel('FPR', fontsize=11)
    ax5.set_ylabel('TPR', fontsize=11)
    ax5.set_title('Step 5: Compare Hulls', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-0.05, 1.05)
    ax5.set_ylim(-0.05, 1.05)
    
    # Calculate areas
    orig_area = hull.volume
    new_area = new_hull.volume if len(new_hull_points) > 0 else 0
    reduction = ((orig_area - new_area) / orig_area * 100) if orig_area > 0 else 0
    
    ax5.text(0.5, -0.2, f'Area reduction: {reduction:.1f}%', 
            ha='center', transform=ax5.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot 6: Statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
HULL COMPARISON SUMMARY

Original Hull:
  • Points: {len(original_hull)}
  • Area: {orig_area:.4f}

After Removal:
  • Remaining: {len(remaining_points)}
  • New hull points: {len(new_hull_points)}
  • New area: {new_area:.4f}

Analysis:
  • Area reduction: {orig_area - new_area:.4f}
  • Reduction %: {reduction:.1f}%
  • Hull efficiency: {(len(original_hull)/len(above_diagonal)*100):.1f}%

Interpretation:
  {interpret_reduction(reduction)}
"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    from pathlib import Path
    output_path = Path('./runs/hull_test/concept_diagram.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved concept diagram to: {output_path}")
    plt.show()
    plt.close()

def interpret_reduction(reduction_pct):
    """Interpret the reduction percentage."""
    if reduction_pct > 60:
        return "High reduction:\n  Hull dominated by few key points"
    elif reduction_pct > 30:
        return "Moderate reduction:\n  Balanced hull diversity"
    else:
        return "Low reduction:\n  Many points contribute to hull"

if __name__ == '__main__':
    import os
    os.makedirs('./runs/hull_test', exist_ok=True)
    create_concept_diagram()
    print("\nConcept diagram created successfully!")
    print("This demonstrates the 5 steps of hull comparison:")
    print("  1. Original points with convex hull")
    print("  2. Remove hull points")
    print("  3. Remaining points")
    print("  4. Calculate new hull")
    print("  5. Compare original vs new hull")
