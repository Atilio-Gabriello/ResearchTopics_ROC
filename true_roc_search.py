"""
True ROC Search Implementation

This implements the genuine ROC search algorithm with adaptive width calculation
based on ROC quality criteria, as described in research papers. Unlike beam search
with fixed width, this dynamically determines the optimal number of subgroups
to keep based on their contribution to the ROC convex hull.

Key differences from enhanced_roc_search.py:
- Adaptive width calculation (not fixed beam width)
- ROC convex hull-based pruning
- Quality-driven subgroup selection
"""

import pandas as pd
import numpy as np
from itertools import combinations
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, KDTree
import time

def load_data(filepath):
    """Load data from file, handling different formats."""
    try:
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        else:
            # Assume tab-separated or space-separated
            data = pd.read_csv(filepath, sep=None, engine='python')
        
        print(f"Loaded data with shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_subgroup_stats(data, conditions, target_col):
    """Calculate statistics for a subgroup defined by conditions."""
    if not conditions:
        mask = pd.Series([True] * len(data))
    else:
        mask = pd.Series([True] * len(data))
        for col, op, val in conditions:
            if op == '>=':
                mask = mask & (data[col] >= val)
            elif op == '<=':
                mask = mask & (data[col] <= val)
            elif op == '==':
                mask = mask & (data[col] == val)
            elif op == '!=':
                mask = mask & (data[col] != val)
    
    subgroup_data = data[mask]
    if len(subgroup_data) == 0:
        return None
    
    # Calculate coverage and target statistics
    coverage = len(subgroup_data)
    coverage_ratio = coverage / len(data)
    
    if target_col not in subgroup_data.columns:
        return None
    
    # Convert target to binary if needed
    target_values = data[target_col].unique()
    if len(target_values) == 2:
        # Convert to binary (1 for positive class, 0 for negative)
        # For income data, positive class should be high income (gr50K, >50K, etc.)
        positive_class = None
        for val in target_values:
            if 'gr' in str(val).lower() or '>50' in str(val) or '1' in str(val):
                positive_class = val
                break
        
        # If no obvious positive class, use the less frequent class (minority class)
        if positive_class is None:
            value_counts = data[target_col].value_counts()
            positive_class = value_counts.idxmin()  # Less frequent class
        
        print(f"Using '{positive_class}' as positive class (high income)")
        data_binary = (data[target_col] == positive_class).astype(int)
        subgroup_binary = (subgroup_data[target_col] == positive_class).astype(int)
        
        target_mean = subgroup_binary.mean()
        population_mean = data_binary.mean()
        
        # Calculate confusion matrix metrics
        tp = subgroup_binary.sum()
        fp = (subgroup_binary == 0).sum()
        
        # Population totals
        total_positives = data_binary.sum()
        total_negatives = (data_binary == 0).sum()
        
        tpr = tp / total_positives if total_positives > 0 else 0
        fpr = fp / total_negatives if total_negatives > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return {
            'conditions': conditions,
            'coverage': coverage,
            'coverage_ratio': coverage_ratio,
            'target_mean': target_mean,
            'population_mean': population_mean,
            'lift': target_mean / population_mean if population_mean > 0 else 0,
            'tp': tp,
            'fp': fp,
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision
        }
    else:
        # Numeric target
        target_mean = subgroup_data[target_col].mean()
        population_mean = data[target_col].mean()
        
        return {
            'conditions': conditions,
            'coverage': coverage,
            'coverage_ratio': coverage_ratio,
            'target_mean': target_mean,
            'population_mean': population_mean,
            'lift': target_mean / population_mean if population_mean > 0 else 0
        }

def roc_quality_measure(tpr, fpr, alpha=None):
    """
    Calculate ROC quality measure. If alpha is provided, uses weighted measure.
    If alpha is None, uses distance from diagonal (ROC improvement).
    
    Args:
        tpr: True Positive Rate
        fpr: False Positive Rate  
        alpha: Weight parameter [0,1] (optional)
    
    Returns:
        ROC quality score
    """
    if alpha is not None:
        return alpha * tpr + (1 - alpha) * (1 - fpr)
    else:
        # Pure ROC quality: distance above diagonal (TPR - FPR)
        # This measures how much better than random the classifier is
        return tpr - fpr

def is_on_roc_hull(points):
    """
    Determine which points are on the ROC convex hull.
    
    Args:
        points: List of (fpr, tpr) tuples
    
    Returns:
        Boolean array indicating which points are on the hull
    """
    if len(points) < 3:
        return [True] * len(points)
    
    # Convert to numpy array
    points_array = np.array(points)
    
    # Add corner points for ROC space: (0,0) and (1,1)
    extended_points = np.vstack([
        [0, 0],  # Perfect specificity, no sensitivity
        points_array,
        [1, 1]   # No specificity, perfect sensitivity
    ])
    
    try:
        hull = ConvexHull(extended_points)
        hull_indices = set(hull.vertices)
        
        # Remove the added corner points (indices 0 and len(points)+1)
        hull_indices.discard(0)
        hull_indices.discard(len(points) + 1)
        
        # Adjust indices back to original points
        original_hull_indices = [i - 1 for i in hull_indices if 1 <= i <= len(points)]
        
        on_hull = [False] * len(points)
        for i in original_hull_indices:
            on_hull[i] = True
            
        return on_hull
    
    except Exception:
        # If hull calculation fails, keep all points
        return [True] * len(points)

def adaptive_roc_pruning(subgroups, alpha=None, quality_threshold=None):
    """
    Implement true ROC search pruning with adaptive width.
    
    Keep only subgroups that are on the ROC convex hull or significantly 
    contribute to ROC performance. If alpha is None, uses pure ROC approach.
    
    Args:
        subgroups: List of subgroup statistics
        alpha: ROC quality weight parameter (optional, for alpha-ROC search)
        quality_threshold: Minimum quality to keep (auto-calculated if None)
    
    Returns:
        Pruned list of subgroups with adaptive width
    """
    if not subgroups:
        return []
    
    # Calculate ROC quality for each subgroup
    for sg in subgroups:
        sg['roc_quality'] = roc_quality_measure(sg['tpr'], sg['fpr'], alpha)
    
    # Sort by ROC quality (descending)
    subgroups.sort(key=lambda x: x['roc_quality'], reverse=True)
    
    # Filter out subgroups below diagonal (worse than random) for pure ROC
    if alpha is None:
        subgroups = [sg for sg in subgroups if sg['roc_quality'] > 0]
        if not subgroups:
            print("No subgroups better than random found!")
            return []
    
    # Auto-calculate quality threshold if not provided
    if quality_threshold is None:
        qualities = [sg['roc_quality'] for sg in subgroups]
        if alpha is None:
            # For pure ROC, be more selective
            quality_threshold = np.percentile(qualities, 80)  # Keep top 20%
        else:
            quality_threshold = np.percentile(qualities, 90)  # Keep top 10%
    
    # Extract ROC points
    roc_points = [(sg['fpr'], sg['tpr']) for sg in subgroups]
    
    # Determine which points are on convex hull
    on_hull = is_on_roc_hull(roc_points)
    
    # Keep subgroups that meet ROC criteria
    kept_subgroups = []
    
    for i, sg in enumerate(subgroups):
        keep = False
        
        # Always keep if on ROC hull
        if on_hull[i]:
            keep = True
            sg['keep_reason'] = 'ROC_HULL'
        
        # Keep high-quality subgroups
        elif sg['roc_quality'] >= quality_threshold:
            keep = True
            sg['keep_reason'] = 'HIGH_QUALITY'
        
        # Keep top few performers to ensure minimum diversity
        elif i < 3:
            keep = True
            sg['keep_reason'] = 'TOP_PERFORMER'
        
        if keep:
            kept_subgroups.append(sg)
    
    # Additional filtering: remove redundant subgroups with very similar ROC points
    if len(kept_subgroups) > 30:  # More aggressive for pure ROC
        filtered_subgroups = []
        for sg in kept_subgroups:
            # Check if this subgroup is too similar to already kept subgroups
            is_redundant = False
            for existing in filtered_subgroups:
                fpr_dist = abs(sg['fpr'] - existing['fpr'])
                tpr_dist = abs(sg['tpr'] - existing['tpr'])
                if fpr_dist < 0.05 and tpr_dist < 0.05:  # Very similar ROC points
                    if sg['roc_quality'] <= existing['roc_quality']:
                        is_redundant = True
                        break
            
            if not is_redundant:
                filtered_subgroups.append(sg)
            
            # Stop if we have enough diverse subgroups
            if len(filtered_subgroups) >= 20:  # Smaller limit for pure ROC
                break
        
        kept_subgroups = filtered_subgroups
    
    search_type = "Pure ROC" if alpha is None else f"Alpha-ROC (α={alpha})"
    print(f"{search_type} pruning: {len(subgroups)} → {len(kept_subgroups)} subgroups (width: {len(kept_subgroups)})")
    
    return kept_subgroups

def generate_candidates(data, target_col, current_subgroups, depth, min_coverage=10):
    """
    Generate candidate subgroups by extending current subgroups.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        current_subgroups: List of current subgroup statistics
        depth: Current search depth
        min_coverage: Minimum coverage for candidates
    
    Returns:
        List of candidate subgroup statistics
    """
    candidates = []
    subgroups = []
    
    # Get numeric and categorical columns for conditions
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target column from both lists
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Limit categorical columns to those with reasonable number of unique values
    categorical_cols = [col for col in categorical_cols 
                       if data[col].nunique() <= 20]  # Avoid explosion
    
    all_cols = numeric_cols + categorical_cols
    
    # Generate candidates from current subgroups
    for sg in current_subgroups:
        conditions = sg['conditions']
        
        if len(conditions) >= depth:
            continue
        
        # Try adding new conditions
        for col in all_cols:
            # Skip if column already has a condition
            existing_cols = [c[0] for c in conditions]
            if col in existing_cols:
                continue
            
            # Handle numeric columns
            if col in numeric_cols:
                col_values = data[col].dropna()
                if len(col_values) == 0:
                    continue
                
                # Generate threshold candidates
                percentiles = [25, 50, 75]
                thresholds = []
                
                for p in percentiles:
                    thresh = np.percentile(col_values, p)
                    thresholds.append(thresh)
                
                # Add min/max for diversity
                thresholds.extend([col_values.min(), col_values.max()])
                thresholds = sorted(list(set(thresholds)))
                
                # Create candidates with different operators
                for thresh in thresholds[:3]:  # Limit to avoid explosion
                    for op in ['>=', '<=', '==', '!=']:
                        new_conditions = conditions + [(col, op, thresh)]
                        
                        # Calculate statistics for candidate
                        candidate_stats = calculate_subgroup_stats(data, new_conditions, target_col)

                        if (candidate_stats and 
                            candidate_stats['coverage'] >= min_coverage and
                            'tpr' in candidate_stats and 'fpr' in candidate_stats):
                            subgroups.append([candidate_stats.get('fpr').tolist(), candidate_stats.get('tpr').tolist()])
                            candidates.append(candidate_stats)
            
            # Handle categorical columns
            elif col in categorical_cols:
                unique_values = data[col].dropna().unique()
                if len(unique_values) == 0:
                    continue
                
                # Limit number of values to avoid explosion
                if len(unique_values) > 10:
                    # Use top 10 most frequent values
                    top_values = data[col].value_counts().head(10).index.tolist()
                    unique_values = top_values
                
                # Create candidates with == and != operators
                for val in unique_values:
                    for op in ['==', '!=']:
                        new_conditions = conditions + [(col, op, val)]
                        
                        # Calculate statistics for candidate
                        candidate_stats = calculate_subgroup_stats(data, new_conditions, target_col)

                        if (candidate_stats and 
                            candidate_stats['coverage'] >= min_coverage and
                            'tpr' in candidate_stats and 'fpr' in candidate_stats):
                            subgroups.append([candidate_stats.get('fpr').tolist(), candidate_stats.get('tpr').tolist()])
                            candidates.append(candidate_stats)
                    # if (candidate_stats['tpr'] >= candidate_stats['fpr']):

    subgroups = np.array(subgroups)
    print(subgroups)
    # In order to calculate the convex hull above the diagonal we need to filter to only points above the diagonal (TPR > FPR)
    ch_eligible = subgroups[subgroups[:, 1] > subgroups[:, 0]]
    print(ch_eligible)

    # Add anchor points (0, 0) and (1, 1) to the set
    hull_points = np.vstack([[0, 0], ch_eligible, [1, 1]])
    print(hull_points)

    # Compute convex hull
    hull = ConvexHull(hull_points)
    hull_points_indices = hull.vertices
    print(hull_points_indices)
    points_on_hull = hull_points[hull_points_indices]
    print(points_on_hull)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(subgroups[:, 0], subgroups[:, 1], 'bo', label='All points')
    plt.plot(ch_eligible[:, 0], ch_eligible[:, 1], 'go', label='Points above diagonal')
    plt.plot([0, 1], [0, 1], 'k--', label='Diagonal (y = x)')
    plt.fill(hull_points[hull.vertices, 0], hull_points[hull.vertices, 1], 'r', alpha=0.2, label='Convex hull')
    plt.plot(hull_points[hull.vertices, 0], hull_points[hull.vertices, 1], 'r-', lw=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Convex Hull of Points Above Diagonal')
    plt.legend()
    plt.grid(True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    plt.close()                    
    print(candidates)
    return candidates

def true_roc_search(data, target_col, alphas=None, max_depth=3, min_coverage=50):
    """
    Implement true ROC search with adaptive width calculation.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        alphas: List of alpha values to test (None for pure ROC search)
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with results for each alpha (or single result for pure ROC)
    """
    results = {}
    
    # If no alphas provided, run pure ROC search
    if alphas is None:
        alphas = [None]
        search_modes = ["Pure ROC"]
    else:
        search_modes = [f"α = {alpha}" for alpha in alphas]
    
    for alpha, mode_name in zip(alphas, search_modes):
        print(f"\n=== True ROC Search: {mode_name} ===")
        start_time = time.time()
        
        # Initialize with population (empty conditions)
        population_stats = calculate_subgroup_stats(data, [], target_col)
        if not population_stats or 'tpr' not in population_stats:
            print(f"Could not calculate population statistics")
            continue
        
        current_subgroups = [population_stats]
        all_subgroups = [population_stats]
        
        candidates_explored = 0
        
        for depth in range(1, max_depth + 1):
            print(f"Depth {depth}: Starting with {len(current_subgroups)} subgroups")
            
            # Generate candidates
            candidates = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
            candidates_explored += len(candidates)
            
            if not candidates:
                print(f"No valid candidates at depth {depth}")
                break
            
            print(f"Generated {len(candidates)} candidates")
            
            # Combine current subgroups with new candidates
            all_candidates = current_subgroups + candidates
            
            # Apply adaptive ROC pruning
            current_subgroups = adaptive_roc_pruning(all_candidates, alpha)
            
            # Keep track of all subgroups found
            all_subgroups.extend(candidates)
            
            if not current_subgroups:
                print(f"No subgroups survived pruning at depth {depth}")
                break
        
        # Final adaptive pruning on all subgroups
        final_subgroups = adaptive_roc_pruning(all_subgroups, alpha)
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        adaptive_width = len(final_subgroups)
        
        # Calculate AUC approximation
        if len(final_subgroups) > 1:
            # Sort by FPR for AUC calculation
            sorted_subgroups = sorted(final_subgroups, key=lambda x: x['fpr'])
            
            auc = 0.0
            for i in range(len(sorted_subgroups) - 1):
                x1, y1 = sorted_subgroups[i]['fpr'], sorted_subgroups[i]['tpr']
                x2, y2 = sorted_subgroups[i+1]['fpr'], sorted_subgroups[i+1]['tpr']
                auc += (x2 - x1) * (y1 + y2) / 2
        else:
            auc = 0.0
        
        # Best quality subgroup
        best_sg = max(final_subgroups, key=lambda x: x['roc_quality'])
        
        # Use 'pure_roc' as key for alpha-free search
        result_key = 'pure_roc' if alpha is None else alpha
        
        results[result_key] = {
            'alpha': alpha,
            'mode': mode_name,
            'adaptive_width': adaptive_width,
            'total_candidates': candidates_explored,
            'final_subgroups': len(final_subgroups),
            'search_time': elapsed_time,
            'auc_approx': auc,
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': final_subgroups
        }
        
        print(f"Completed {mode_name}:")
        print(f"  Adaptive width: {adaptive_width}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC approximation: {auc:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
    
    return results

def save_results(results, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary comparison
    summary_data = []
    for alpha, result in results.items():
        summary_data.append({
            'alpha': alpha,
            'adaptive_width': result['adaptive_width'],
            'total_candidates': result['total_candidates'],
            'auc_approx': result['auc_approx'],
            'best_quality': result['best_quality'],
            'best_tpr': result['best_tpr'],
            'best_fpr': result['best_fpr'],
            'best_precision': result['best_precision'],
            'search_time': result['search_time']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / 'true_roc_comparison.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")
    
    # Save detailed results for each alpha
    for alpha, result in results.items():
        alpha_dir = output_dir / f"alpha_{alpha}"
        alpha_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config = {
            'alpha': alpha,
            'adaptive_width': result['adaptive_width'],
            'algorithm': 'true_roc_search',
            'total_candidates': result['total_candidates'],
            'search_time': result['search_time']
        }
        
        with open(alpha_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save subgroups
        subgroups_data = []
        roc_points_data = []
        
        for i, sg in enumerate(result['subgroups']):
            # Subgroup details
            conditions_str = ' AND '.join([f"{col} {op} {val}" for col, op, val in sg['conditions']])
            if not conditions_str:
                conditions_str = "Population (no conditions)"
            
            subgroups_data.append({
                'rank': i + 1,
                'conditions': conditions_str,
                'coverage': sg['coverage'],
                'coverage_ratio': sg['coverage_ratio'],
                'tpr': sg['tpr'],
                'fpr': sg['fpr'],
                'precision': sg['precision'],
                'roc_quality': sg['roc_quality'],
                'keep_reason': sg.get('keep_reason', 'FINAL_SELECTION')
            })
            
            # ROC points
            roc_points_data.append({
                'fpr': sg['fpr'],
                'tpr': sg['tpr'],
                'quality': sg['roc_quality'],
                'coverage': sg['coverage']
            })
        
        # Save to CSV
        pd.DataFrame(subgroups_data).to_csv(alpha_dir / 'subgroups.csv', index=False)
        pd.DataFrame(roc_points_data).to_csv(alpha_dir / 'roc_points.csv', index=False)
        
        # Create ROC plot
        create_roc_plot(result['subgroups'], alpha, alpha_dir / 'roc_curve.png')

def create_roc_plot(subgroups, alpha, output_path):
    """Create ROC curve visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract points
    fpr_vals = [sg['fpr'] for sg in subgroups]
    tpr_vals = [sg['tpr'] for sg in subgroups]
    qualities = [sg['roc_quality'] for sg in subgroups]
    
    # Plot points colored by quality
    scatter = ax.scatter(fpr_vals, tpr_vals, c=qualities, cmap='viridis', 
                        s=100, alpha=0.7, edgecolors='black')
    
    # Add diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    # Formatting
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title(f'True ROC Search Results (α = {alpha})\nAdaptive Width: {len(subgroups)} subgroups')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('ROC Quality')
    
    # Add legend
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plot(results, output_path):
    """Create comparison plot across different alpha values."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    alphas = list(results.keys())
    
    # Plot 1: Adaptive width vs alpha
    widths = [results[alpha]['adaptive_width'] for alpha in alphas]
    ax1.plot(alphas, widths, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Adaptive Width')
    ax1.set_title('Adaptive Width vs Alpha')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC vs alpha
    aucs = [results[alpha]['auc_approx'] for alpha in alphas]
    ax2.plot(alphas, aucs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('AUC Approximation')
    ax2.set_title('AUC vs Alpha')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Best quality vs alpha
    qualities = [results[alpha]['best_quality'] for alpha in alphas]
    ax3.plot(alphas, qualities, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Alpha')
    ax3.set_ylabel('Best ROC Quality')
    ax3.set_title('Best Quality vs Alpha')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ROC space visualization
    colors = plt.cm.viridis(np.linspace(0, 1, len(alphas)))
    for i, (alpha, color) in enumerate(zip(alphas, colors)):
        subgroups = results[alpha]['subgroups']
        fpr_vals = [sg['fpr'] for sg in subgroups]
        tpr_vals = [sg['tpr'] for sg in subgroups]
        ax4.scatter(fpr_vals, tpr_vals, c=[color], alpha=0.7, 
                   label=f'α = {alpha} (w={len(subgroups)})', s=60)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Space Comparison')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='True ROC Search with Adaptive Width')
    parser.add_argument('--data', default='./tests/adult.txt', help='Path to data file')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--alphas', nargs='+', type=float, default=None,
                       help='Alpha values to test (omit for pure ROC search)')
    parser.add_argument('--pure-roc', action='store_true', 
                       help='Run pure ROC search without alpha parameter')
    parser.add_argument('--depth', type=int, default=3, help='Maximum search depth')
    parser.add_argument('--min-coverage', type=int, default=50, help='Minimum subgroup coverage')
    parser.add_argument('--output', default='./runs/true_roc', help='Output directory')
    
    args = parser.parse_args()
    
    # Determine search mode
    if args.pure_roc or args.alphas is None:
        alphas = None
        search_mode = "Pure ROC Search (no alpha)"
    else:
        alphas = args.alphas
        search_mode = f"Alpha-ROC Search (alphas: {alphas})"
    
    print("=== True ROC Search Implementation ===")
    print(f"Data: {args.data}")
    print(f"Target: {args.target}")
    print(f"Search mode: {search_mode}")
    print(f"Max depth: {args.depth}")
    print(f"Min coverage: {args.min_coverage}")
    print(f"Output: {args.output}")
    
    # Load data
    data = load_data(args.data)
    if data is None:
        return
    
    # Run true ROC search
    results = true_roc_search(data, args.target, alphas, args.depth, args.min_coverage)
    
    if results:
        # Save results
        save_results(results, args.output)
        
        # Create comparison plot (only if multiple results)
        if len(results) > 1:
            comparison_path = Path(args.output) / 'true_roc_comparison.png'
            create_comparison_plot(results, comparison_path)
            print(f"Saved comparison plot: {comparison_path}")
        
        # Print summary
        print("\n=== True ROC Search Summary ===")
        for key in results.keys():
            r = results[key]
            mode_str = r['mode']
            print(f"{mode_str}: width = {r['adaptive_width']}, "
                  f"AUC = {r['auc_approx']:.3f}, "
                  f"quality = {r['best_quality']:.3f}")

if __name__ == '__main__':
    main()