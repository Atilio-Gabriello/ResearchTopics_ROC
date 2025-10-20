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
## TODO
## Implement the 5 changes
## get the pseudocode for the ROC search

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
import glob

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
    # Note: Missing values should already be filtered by preprocess_categorical_data
    # But we keep a safety check here just in case
    if data[target_col].dtype == 'object':
        # If data is not preprocessed, filter missing values
        clean_data = data.copy()
        clean_data = clean_data[clean_data[target_col].notna()]
        clean_data = clean_data[clean_data[target_col] != '?']
        clean_data = clean_data[clean_data[target_col] != '']
        clean_data = clean_data.reset_index(drop=True)
    else:
        # Data is already preprocessed (numerical)
        clean_data = data
    
    if len(clean_data) == 0:
        return None
    
    if not conditions:
        mask = pd.Series([True] * len(clean_data))
    else:
        mask = pd.Series([True] * len(clean_data))
        for col, op, val in conditions:
            if op == '>=':
                mask = mask & (clean_data[col] >= val)
            elif op == '<=':
                mask = mask & (clean_data[col] <= val)
            elif op == '==':
                mask = mask & (clean_data[col] == val)
            elif op == '!=':
                mask = mask & (clean_data[col] != val)
    
    subgroup_data = clean_data[mask]
    if len(subgroup_data) == 0:
        return None
    
    # Calculate coverage and target statistics
    coverage = len(subgroup_data)
    coverage_ratio = coverage / len(clean_data)
    
    if target_col not in subgroup_data.columns:
        return None
    
    # Convert target to binary if needed
    target_values = clean_data[target_col].unique()
    if len(target_values) == 2:
        # Convert to binary (1 for positive class, 0 for negative)
        # For income data, positive class should be high income (gr50K, >50K, etc.)
        # For credit data, positive class might be '+' or 'good' or '1'
        positive_class = None
        for val in target_values:
            val_str = str(val).lower()
            if any(pattern in val_str for pattern in ['gr', '>50', '+', 'good', 'yes', 'true']):
                positive_class = val
                break
        
        # If no obvious positive class, use the less frequent class (minority class)
        if positive_class is None:
            value_counts = clean_data[target_col].value_counts()
            positive_class = value_counts.idxmin()  # Less frequent class
        
        print(f"Using '{positive_class}' as positive class")
        clean_data_binary = (clean_data[target_col] == positive_class).astype(int)
        subgroup_binary = (subgroup_data[target_col] == positive_class).astype(int)
        
        target_mean = subgroup_binary.mean()
        population_mean = clean_data_binary.mean()
        
        # Calculate confusion matrix metrics
        tp = subgroup_binary.sum()
        fp = (subgroup_binary == 0).sum()
        
        # Population totals
        total_positives = clean_data_binary.sum()
        total_negatives = (clean_data_binary == 0).sum()
        
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

def calculate_roc_metrics(points):
    """
    Calculate ROC metrics for a set of points.
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
    
    Returns:
        Dictionary with AUC, number of points, best TPR, best FPR, etc.
    """
    if len(points) == 0:
        return {
            'auc': 0.0,
            'num_points': 0,
            'best_tpr': 0.0,
            'best_fpr': 0.0,
            'avg_tpr': 0.0,
            'avg_fpr': 0.0,
            'max_quality': 0.0
        }
    
    points_array = np.array(points)
    
    # Calculate AUC using trapezoidal rule
    # Sort by FPR
    sorted_indices = np.argsort(points_array[:, 0])
    sorted_points = points_array[sorted_indices]
    
    # Add (0, 0) and (1, 1) for complete ROC curve
    roc_curve = np.vstack([[0, 0], sorted_points, [1, 1]])
    
    # Calculate AUC using trapezoidal rule
    auc = 0.0
    for i in range(len(roc_curve) - 1):
        x1, y1 = roc_curve[i]
        x2, y2 = roc_curve[i + 1]
        # Trapezoidal area
        auc += (x2 - x1) * (y1 + y2) / 2
    
    # Calculate best point (highest TPR - FPR, or ROC quality)
    qualities = points_array[:, 1] - points_array[:, 0]  # TPR - FPR
    best_idx = np.argmax(qualities)
    
    return {
        'auc': auc,
        'num_points': len(points),
        'best_tpr': points_array[best_idx, 1],
        'best_fpr': points_array[best_idx, 0],
        'avg_tpr': np.mean(points_array[:, 1]),
        'avg_fpr': np.mean(points_array[:, 0]),
        'max_quality': qualities[best_idx],
        'avg_quality': np.mean(qualities)
    }

def remove_hull_points_and_recalculate(points, return_details=False):
    """
    Remove points on the original convex hull and recalculate the hull with remaining points.
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
        return_details: If True, return detailed comparison information
    
    Returns:
        If return_details=False: Array of new hull points
        If return_details=True: Dictionary with original hull, new hull, removed points, and ROC metrics
    """
    if len(points) < 3:
        if return_details:
            return {
                'original_hull': points,
                'new_hull': np.array([]),
                'removed_points': points,
                'remaining_points': np.array([]),
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    points_array = np.array(points)
    
    # Filter to only points above diagonal (TPR > FPR)
    above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
    
    if len(above_diagonal) < 3:
        if return_details:
            return {
                'original_hull': above_diagonal,
                'new_hull': np.array([]),
                'removed_points': above_diagonal,
                'remaining_points': np.array([]),
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    # Add anchor points (0, 0) and (1, 1)
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    
    try:
        # Compute original convex hull
        original_hull = ConvexHull(extended_points)
        original_hull_indices = set(original_hull.vertices)
        
        # Get original hull points (excluding anchors)
        original_hull_points_indices = [i - 1 for i in original_hull_indices 
                                       if 1 <= i <= len(above_diagonal)]
        original_hull_points = above_diagonal[original_hull_points_indices]
        
        # Calculate original hull area
        original_hull_area = original_hull.volume
        
        # Remove hull points from the set
        remaining_indices = [i for i in range(len(above_diagonal)) 
                           if i not in original_hull_points_indices]
        remaining_points = above_diagonal[remaining_indices]
        
        if len(remaining_points) < 3:
            # Not enough points to form a new hull
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'removed_points': original_hull_points,
                    'remaining_points': remaining_points,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0,
                    'hull_area_reduction': original_hull_area
                }
            return np.array([])
        
        # Recalculate hull with remaining points
        new_extended_points = np.vstack([[0, 0], remaining_points, [1, 1]])
        new_hull = ConvexHull(new_extended_points)
        new_hull_indices = set(new_hull.vertices)
        
        # Get new hull points (excluding anchors)
        new_hull_points_indices = [i - 1 for i in new_hull_indices 
                                  if 1 <= i <= len(remaining_points)]
        new_hull_points = remaining_points[new_hull_points_indices]
        
        # Calculate new hull area
        new_hull_area = new_hull.volume
        
        if return_details:
            # Calculate ROC metrics for original hull
            original_metrics = calculate_roc_metrics(original_hull_points)
            
            # Calculate ROC metrics for new hull
            new_metrics = calculate_roc_metrics(new_hull_points)
            
            # Calculate metrics for all points (for reference)
            all_metrics = calculate_roc_metrics(above_diagonal)
            
            return {
                'original_hull': original_hull_points,
                'new_hull': new_hull_points,
                'removed_points': original_hull_points,
                'remaining_points': remaining_points,
                'all_points': above_diagonal,
                'original_hull_area': original_hull_area,
                'new_hull_area': new_hull_area,
                'hull_area_reduction': original_hull_area - new_hull_area,
                'reduction_percentage': ((original_hull_area - new_hull_area) / original_hull_area * 100) 
                                       if original_hull_area > 0 else 0,
                # Original hull metrics
                'original_auc': original_metrics['auc'],
                'original_num_subgroups': original_metrics['num_points'],
                'original_best_tpr': original_metrics['best_tpr'],
                'original_best_fpr': original_metrics['best_fpr'],
                'original_avg_quality': original_metrics['avg_quality'],
                'original_max_quality': original_metrics['max_quality'],
                # New hull metrics
                'new_auc': new_metrics['auc'],
                'new_num_subgroups': new_metrics['num_points'],
                'new_best_tpr': new_metrics['best_tpr'],
                'new_best_fpr': new_metrics['best_fpr'],
                'new_avg_quality': new_metrics['avg_quality'],
                'new_max_quality': new_metrics['max_quality'],
                # All points metrics (for reference)
                'all_points_auc': all_metrics['auc'],
                'all_points_num': all_metrics['num_points'],
                'all_points_max_quality': all_metrics['max_quality'],
                # Comparison metrics
                'auc_reduction': original_metrics['auc'] - new_metrics['auc'],
                'auc_reduction_percentage': ((original_metrics['auc'] - new_metrics['auc']) / original_metrics['auc'] * 100)
                                           if original_metrics['auc'] > 0 else 0,
                'subgroups_removed': original_metrics['num_points'],
                'subgroups_remaining': len(remaining_points),
                'quality_reduction': original_metrics['max_quality'] - new_metrics['max_quality']
            }
        
        return new_hull_points
        
    except Exception as e:
        print(f"Error in hull recalculation: {e}")
        if return_details:
            return {
                'original_hull': np.array([]),
                'new_hull': np.array([]),
                'removed_points': np.array([]),
                'remaining_points': points_array,
                'original_hull_area': 0,
                'new_hull_area': 0,
                'error': str(e)
            }
        return np.array([])

def select_closest_points_to_hull(points, n_points, return_details=False, exclude_hull_points=False):
    """
    Select the n closest points to the convex hull and create a new ROC curve.
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
        n_points: Number of closest points to select
        return_details: If True, return detailed comparison information
        exclude_hull_points: If True, only select from non-hull points (guarantees curve change)
                           If False (default), may include original hull points (curve may stay same)
                           NOTE: Set to True to force a different curve. Original behavior (False) 
                           is useful to see if hull is stable with nearby points included.
    
    Returns:
        If return_details=False: Array of new hull points from selected closest points
        If return_details=True: Dictionary with original hull, new hull, and ROC metrics
    """
    if len(points) < 3:
        if return_details:
            return {
                'original_hull': points,
                'new_hull': np.array([]),
                'selected_points': points,
                'all_points': points,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    points_array = np.array(points)
    
    # Filter to only points above diagonal (TPR > FPR)
    above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
    
    if len(above_diagonal) < 3:
        if return_details:
            return {
                'original_hull': above_diagonal,
                'new_hull': np.array([]),
                'selected_points': above_diagonal,
                'all_points': above_diagonal,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    # Calculate original convex hull
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    
    try:
        original_hull = ConvexHull(extended_points)
        original_hull_area = original_hull.volume
        
        # Get original hull points (excluding anchors)
        original_hull_indices = [i - 1 for i in original_hull.vertices 
                                if 1 <= i <= len(above_diagonal)]
        original_hull_points = above_diagonal[original_hull_indices]
        
        # MODIFICATION: Option to exclude hull points from selection
        # This ensures the new curve will be different from the original
        if exclude_hull_points:
            # Select only from non-hull points
            non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
            non_hull_mask[original_hull_indices] = False
            candidate_points = above_diagonal[non_hull_mask]
            candidate_indices = np.where(non_hull_mask)[0]
            
            if len(candidate_points) < 3:
                # Not enough non-hull points
                if return_details:
                    return {
                        'original_hull': original_hull_points,
                        'new_hull': np.array([]),
                        'selected_points': candidate_points,
                        'all_points': above_diagonal,
                        'original_hull_area': original_hull_area,
                        'new_hull_area': 0,
                        'note': 'Not enough non-hull points to form new hull'
                    }
                return np.array([])
        else:
            # Include all points (original behavior - may include hull points)
            candidate_points = above_diagonal
            candidate_indices = np.arange(len(above_diagonal))
        
        # Calculate distance from each candidate point to the hull
        # Use KDTree for efficient nearest neighbor search
        if len(original_hull_points) > 0:
            hull_tree = KDTree(original_hull_points)
            distances, _ = hull_tree.query(candidate_points)
        else:
            distances = np.zeros(len(candidate_points))
        
        # Select n_points closest to the hull from candidates
        n_select = min(n_points, len(candidate_points))
        closest_indices_in_candidates = np.argsort(distances)[:n_select]
        selected_points = candidate_points[closest_indices_in_candidates]
        
        if len(selected_points) < 3:
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': selected_points,
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0
                }
            return np.array([])
        
        # Calculate new hull with selected points
        new_extended_points = np.vstack([[0, 0], selected_points, [1, 1]])
        new_hull = ConvexHull(new_extended_points)
        new_hull_area = new_hull.volume
        
        # Get new hull points (excluding anchors)
        new_hull_indices = [i - 1 for i in new_hull.vertices 
                           if 1 <= i <= len(selected_points)]
        new_hull_points = selected_points[new_hull_indices]
        
        if return_details:
            # Calculate ROC metrics
            original_metrics = calculate_roc_metrics(original_hull_points)
            new_metrics = calculate_roc_metrics(new_hull_points)
            all_metrics = calculate_roc_metrics(above_diagonal)
            
            return {
                'original_hull': original_hull_points,
                'new_hull': new_hull_points,
                'selected_points': selected_points,
                'all_points': above_diagonal,
                'selection_criterion': f'closest_{n_points}_to_hull',
                'n_selected': n_select,
                'original_hull_area': original_hull_area,
                'new_hull_area': new_hull_area,
                'hull_area_reduction': original_hull_area - new_hull_area,
                'reduction_percentage': ((original_hull_area - new_hull_area) / original_hull_area * 100) 
                                       if original_hull_area > 0 else 0,
                # Original hull metrics
                'original_auc': original_metrics['auc'],
                'original_num_subgroups': original_metrics['num_points'],
                'original_best_tpr': original_metrics['best_tpr'],
                'original_best_fpr': original_metrics['best_fpr'],
                'original_avg_quality': original_metrics['avg_quality'],
                'original_max_quality': original_metrics['max_quality'],
                # New hull metrics
                'new_auc': new_metrics['auc'],
                'new_num_subgroups': new_metrics['num_points'],
                'new_best_tpr': new_metrics['best_tpr'],
                'new_best_fpr': new_metrics['best_fpr'],
                'new_avg_quality': new_metrics['avg_quality'],
                'new_max_quality': new_metrics['max_quality'],
                # All points metrics
                'all_points_auc': all_metrics['auc'],
                'all_points_num': all_metrics['num_points'],
                'all_points_max_quality': all_metrics['max_quality'],
                # Comparison metrics
                'auc_reduction': original_metrics['auc'] - new_metrics['auc'],
                'auc_reduction_percentage': ((original_metrics['auc'] - new_metrics['auc']) / original_metrics['auc'] * 100)
                                           if original_metrics['auc'] > 0 else 0,
                'quality_reduction': original_metrics['max_quality'] - new_metrics['max_quality']
            }
        
        return new_hull_points
        
    except Exception as e:
        print(f"Error in closest points selection: {e}")
        if return_details:
            return {
                'original_hull': np.array([]),
                'new_hull': np.array([]),
                'selected_points': np.array([]),
                'all_points': points_array,
                'original_hull_area': 0,
                'new_hull_area': 0,
                'error': str(e)
            }
        return np.array([])


def select_furthest_points_from_diagonal(points, n_points, return_details=False, exclude_hull_points=False):
    """
    Select the n furthest points from the diagonal and create a new ROC curve.
    Distance from diagonal is measured as TPR - FPR (ROC quality).
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
        n_points: Number of furthest points to select
        return_details: If True, return detailed comparison information
        exclude_hull_points: If True, only select from non-hull points (guarantees curve change)
                           If False (default), may include original hull points
                           NOTE: Set to True to force a different curve. Original behavior (False)
                           allows selecting best performers even if they're on the hull.
    
    Returns:
        If return_details=False: Array of new hull points from selected furthest points
        If return_details=True: Dictionary with original hull, new hull, and ROC metrics
    """
    if len(points) < 3:
        if return_details:
            return {
                'original_hull': points,
                'new_hull': np.array([]),
                'selected_points': points,
                'all_points': points,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    points_array = np.array(points)
    
    # Filter to only points above diagonal (TPR > FPR)
    above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
    
    if len(above_diagonal) < 3:
        if return_details:
            return {
                'original_hull': above_diagonal,
                'new_hull': np.array([]),
                'selected_points': above_diagonal,
                'all_points': above_diagonal,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    # Calculate original convex hull
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    
    try:
        original_hull = ConvexHull(extended_points)
        original_hull_area = original_hull.volume
        
        # Get original hull points (excluding anchors)
        original_hull_indices = [i - 1 for i in original_hull.vertices 
                                if 1 <= i <= len(above_diagonal)]
        original_hull_points = above_diagonal[original_hull_indices]
        
        # MODIFICATION: Option to exclude hull points from selection
        # This ensures the new curve will be different from the original
        if exclude_hull_points:
            # Select only from non-hull points
            non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
            non_hull_mask[original_hull_indices] = False
            candidate_points = above_diagonal[non_hull_mask]
            
            if len(candidate_points) < 3:
                # Not enough non-hull points
                if return_details:
                    return {
                        'original_hull': original_hull_points,
                        'new_hull': np.array([]),
                        'selected_points': candidate_points,
                        'all_points': above_diagonal,
                        'original_hull_area': original_hull_area,
                        'new_hull_area': 0,
                        'note': 'Not enough non-hull points to form new hull'
                    }
                return np.array([])
        else:
            # Include all points (original behavior - may include hull points)
            candidate_points = above_diagonal
        
        # Calculate distance from diagonal for each candidate point
        # Distance = TPR - FPR (perpendicular distance to y=x line)
        distances_from_diagonal = candidate_points[:, 1] - candidate_points[:, 0]
        
        # Select n_points furthest from the diagonal from candidates
        n_select = min(n_points, len(candidate_points))
        furthest_indices = np.argsort(distances_from_diagonal)[-n_select:]
        selected_points = candidate_points[furthest_indices]
        
        if len(selected_points) < 3:
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': selected_points,
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0
                }
            return np.array([])
        
        # Calculate new hull with selected points
        new_extended_points = np.vstack([[0, 0], selected_points, [1, 1]])
        new_hull = ConvexHull(new_extended_points)
        new_hull_area = new_hull.volume
        
        # Get new hull points (excluding anchors)
        new_hull_indices = [i - 1 for i in new_hull.vertices 
                           if 1 <= i <= len(selected_points)]
        new_hull_points = selected_points[new_hull_indices]
        
        if return_details:
            # Calculate ROC metrics
            original_metrics = calculate_roc_metrics(original_hull_points)
            new_metrics = calculate_roc_metrics(new_hull_points)
            all_metrics = calculate_roc_metrics(above_diagonal)
            
            return {
                'original_hull': original_hull_points,
                'new_hull': new_hull_points,
                'selected_points': selected_points,
                'all_points': above_diagonal,
                'selection_criterion': f'furthest_{n_points}_from_diagonal',
                'n_selected': n_select,
                'original_hull_area': original_hull_area,
                'new_hull_area': new_hull_area,
                'hull_area_reduction': original_hull_area - new_hull_area,
                'reduction_percentage': ((original_hull_area - new_hull_area) / original_hull_area * 100) 
                                       if original_hull_area > 0 else 0,
                # Original hull metrics
                'original_auc': original_metrics['auc'],
                'original_num_subgroups': original_metrics['num_points'],
                'original_best_tpr': original_metrics['best_tpr'],
                'original_best_fpr': original_metrics['best_fpr'],
                'original_avg_quality': original_metrics['avg_quality'],
                'original_max_quality': original_metrics['max_quality'],
                # New hull metrics
                'new_auc': new_metrics['auc'],
                'new_num_subgroups': new_metrics['num_points'],
                'new_best_tpr': new_metrics['best_tpr'],
                'new_best_fpr': new_metrics['best_fpr'],
                'new_avg_quality': new_metrics['avg_quality'],
                'new_max_quality': new_metrics['max_quality'],
                # All points metrics
                'all_points_auc': all_metrics['auc'],
                'all_points_num': all_metrics['num_points'],
                'all_points_max_quality': all_metrics['max_quality'],
                # Comparison metrics
                'auc_reduction': original_metrics['auc'] - new_metrics['auc'],
                'auc_reduction_percentage': ((original_metrics['auc'] - new_metrics['auc']) / original_metrics['auc'] * 100)
                                           if original_metrics['auc'] > 0 else 0,
                'quality_reduction': original_metrics['max_quality'] - new_metrics['max_quality'],
                # Distance metrics
                'avg_distance_from_diagonal': np.mean(distances_from_diagonal[furthest_indices]),
                'min_distance_from_diagonal': np.min(distances_from_diagonal[furthest_indices]),
                'max_distance_from_diagonal': np.max(distances_from_diagonal[furthest_indices])
            }
        
        return new_hull_points
        
    except Exception as e:
        print(f"Error in furthest points selection: {e}")
        if return_details:
            return {
                'original_hull': np.array([]),
                'new_hull': np.array([]),
                'selected_points': np.array([]),
                'all_points': points_array,
                'original_hull_area': 0,
                'new_hull_area': 0,
                'error': str(e)
            }
        return np.array([])


def select_points_below_hull(points, distance_percentage=1.0, return_details=False, exclude_hull_points=True):
    """
    Select points within a percentage threshold of the maximum diagonal distance, 
    measured by their vertical distance below the convex hull.
    
    NEW APPROACH:
    1. Find the point furthest from the diagonal (max TPR - FPR)
    2. Use that distance as the reference
    3. Calculate threshold = reference_distance × (distance_percentage / 100)
    4. Select points where vertical_distance_below_hull ≤ threshold
    
    Example: If max diagonal distance is 0.8 and distance_percentage=1.0:
        - Reference distance: 0.8
        - Threshold: 0.8 × 0.01 = 0.008
        - Select points with hull_distance ≤ 0.008
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
        distance_percentage: Percentage of max diagonal distance to use as threshold (default 1.0%)
        return_details: If True, return detailed comparison information
        exclude_hull_points: If True, only select from non-hull points (default True)
    
    Returns:
        If return_details=False: Array of new hull points from selected points
        If return_details=True: Dictionary with original hull, new hull, and ROC metrics
    """
    if len(points) < 3:
        if return_details:
            return {
                'original_hull': points,
                'new_hull': np.array([]),
                'selected_points': points,
                'all_points': points,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    points_array = np.array(points)
    
    # Filter to only points above diagonal (TPR > FPR)
    above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
    
    if len(above_diagonal) < 3:
        if return_details:
            return {
                'original_hull': above_diagonal,
                'new_hull': np.array([]),
                'selected_points': above_diagonal,
                'all_points': above_diagonal,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    # Calculate original convex hull
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    
    try:
        original_hull = ConvexHull(extended_points)
        original_hull_area = original_hull.volume
        
        # Get original hull points (excluding anchors)
        original_hull_indices = [i - 1 for i in original_hull.vertices 
                                if 1 <= i <= len(above_diagonal)]
        original_hull_points = above_diagonal[original_hull_indices]
        
        # NEW APPROACH: Use furthest point from diagonal as reference
        # MODIFICATION: Exclude hull points FIRST, then calculate reference from candidates
        # This ensures the threshold is based on achievable distances
        if exclude_hull_points:
            # Select only from non-hull points
            non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
            non_hull_mask[original_hull_indices] = False
            candidate_points = above_diagonal[non_hull_mask]
            
            if len(candidate_points) < 3:
                # Not enough non-hull points
                if return_details:
                    return {
                        'original_hull': original_hull_points,
                        'new_hull': np.array([]),
                        'selected_points': candidate_points,
                        'all_points': above_diagonal,
                        'original_hull_area': original_hull_area,
                        'new_hull_area': 0,
                        'reference_distance': 0,
                        'threshold_distance': 0,
                        'note': 'Not enough non-hull points to form new hull'
                    }
                return np.array([])
        else:
            # Include all points (may include original hull points)
            candidate_points = above_diagonal
        
        # Calculate reference from CANDIDATE points only (after hull exclusion)
        diagonal_distances_candidates = candidate_points[:, 1] - candidate_points[:, 0]
        max_diagonal_distance = np.max(diagonal_distances_candidates)
        
        if max_diagonal_distance <= 0:
            # No points above diagonal (should not happen due to filter above)
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': np.array([]),
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0,
                    'reference_distance': max_diagonal_distance,
                    'threshold_distance': 0,
                    'note': 'No valid points above diagonal'
                }
            return np.array([])
        
        # Step 2: Calculate threshold based on percentage of max diagonal distance
        threshold = max_diagonal_distance * (distance_percentage / 100.0)
        
        # Step 3: Calculate vertical distance below hull for each candidate point
        vertical_distances = np.zeros(len(candidate_points))
        
        # Sort hull points by FPR for interpolation
        hull_sorted_indices = np.argsort(original_hull_points[:, 0])
        hull_sorted = original_hull_points[hull_sorted_indices]
        
        # Add anchors for complete hull
        hull_with_anchors = np.vstack([[0, 0], hull_sorted, [1, 1]])
        
        for i, point in enumerate(candidate_points):
            fpr, tpr = point
            
            # Find hull TPR at this FPR (linear interpolation)
            hull_tpr = np.interp(fpr, hull_with_anchors[:, 0], hull_with_anchors[:, 1])
            
            # Vertical distance (positive if below hull, negative if above/on hull)
            vertical_distances[i] = hull_tpr - tpr
        
        # Step 4: Select points where vertical_distance <= threshold
        # Only consider points with positive distance (actually below hull)
        below_hull_mask = vertical_distances > 0
        within_threshold_mask = vertical_distances <= threshold
        selection_mask = below_hull_mask & within_threshold_mask
        
        selected_indices = np.where(selection_mask)[0]
        
        if len(selected_indices) == 0:
            # No points meet criteria, return empty
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': np.array([]),
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0,
                    'reference_distance': max_diagonal_distance,
                    'threshold_distance': threshold,
                    'n_selected': 0,
                    'note': f'No points within threshold {threshold:.4f} (max_diagonal_dist={max_diagonal_distance:.4f}, {distance_percentage}%)'
                }
            return np.array([])
        
        selected_points = candidate_points[selected_indices]
        n_select = len(selected_points)
        
        if len(selected_points) < 3:
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': selected_points,
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0
                }
            return np.array([])
        
        # Calculate new hull with selected points
        new_extended_points = np.vstack([[0, 0], selected_points, [1, 1]])
        new_hull = ConvexHull(new_extended_points)
        new_hull_area = new_hull.volume
        
        # Get new hull points (excluding anchors)
        new_hull_indices = [i - 1 for i in new_hull.vertices 
                           if 1 <= i <= len(selected_points)]
        new_hull_points = selected_points[new_hull_indices]
        
        if return_details:
            # Calculate ROC metrics
            original_metrics = calculate_roc_metrics(original_hull_points)
            new_metrics = calculate_roc_metrics(new_hull_points)
            all_metrics = calculate_roc_metrics(above_diagonal)
            
            return {
                'original_hull': original_hull_points,
                'new_hull': new_hull_points,
                'selected_points': selected_points,
                'all_points': above_diagonal,
                'selection_criterion': f'below_hull_{distance_percentage}pct_of_max_diagonal',
                'n_selected': n_select,
                'reference_distance': max_diagonal_distance,
                'threshold_distance': threshold,
                'distance_percentage': distance_percentage,
                'original_hull_area': original_hull_area,
                'new_hull_area': new_hull_area,
                'hull_area_reduction': original_hull_area - new_hull_area,
                'reduction_percentage': ((original_hull_area - new_hull_area) / original_hull_area * 100) 
                                       if original_hull_area > 0 else 0,
                # Original hull metrics
                'original_auc': original_metrics['auc'],
                'original_num_subgroups': original_metrics['num_points'],
                'original_best_tpr': original_metrics['best_tpr'],
                'original_best_fpr': original_metrics['best_fpr'],
                'original_avg_quality': original_metrics['avg_quality'],
                'original_max_quality': original_metrics['max_quality'],
                # New hull metrics
                'new_auc': new_metrics['auc'],
                'new_num_subgroups': new_metrics['num_points'],
                'new_best_tpr': new_metrics['best_tpr'],
                'new_best_fpr': new_metrics['best_fpr'],
                'new_avg_quality': new_metrics['avg_quality'],
                'new_max_quality': new_metrics['max_quality'],
                # All points metrics
                'all_points_auc': all_metrics['auc'],
                'all_points_num': all_metrics['num_points'],
                'all_points_max_quality': all_metrics['max_quality'],
                # Comparison metrics
                'auc_reduction': original_metrics['auc'] - new_metrics['auc'],
                'auc_reduction_percentage': ((original_metrics['auc'] - new_metrics['auc']) / original_metrics['auc'] * 100)
                                           if original_metrics['auc'] > 0 else 0,
                'quality_reduction': original_metrics['max_quality'] - new_metrics['max_quality'],
                # Vertical distance metrics
                'avg_vertical_distance': np.mean(vertical_distances[selected_indices]),
                'max_vertical_distance': np.max(vertical_distances[selected_indices]),
                'min_vertical_distance': np.min(vertical_distances[selected_indices])
            }
        
        return new_hull_points
        
    except Exception as e:
        print(f"Error in below hull selection: {e}")
        if return_details:
            return {
                'original_hull': np.array([]),
                'new_hull': np.array([]),
                'selected_points': np.array([]),
                'all_points': points_array,
                'original_hull_area': 0,
                'new_hull_area': 0,
                'error': str(e)
            }
        return np.array([])


def select_points_above_diagonal(points, distance_percentage=1.0, return_details=False, exclude_hull_points=True):
    """
    Select points at or above a percentage threshold of the maximum diagonal distance,
    measured by their distance from the diagonal (TPR - FPR).
    
    NEW APPROACH:
    1. Find the point furthest from the diagonal (max TPR - FPR)
    2. Use that distance as the reference
    3. Calculate threshold = reference_distance × ((100 - distance_percentage) / 100)
    4. Select points where diagonal_distance ≥ threshold
    
    Example: If max diagonal distance is 0.8 and distance_percentage=1.0:
        - Reference distance: 0.8
        - Threshold: 0.8 × (100-1)/100 = 0.8 × 0.99 = 0.792
        - Select points with diagonal_distance ≥ 0.792
    
    Args:
        points: Array of (fpr, tpr) points, shape (n, 2)
        distance_percentage: Percentage threshold (default 1.0% means select points at least 99% as good)
        return_details: If True, return detailed comparison information
        exclude_hull_points: If True, only select from non-hull points (default True)
    
    Returns:
        If return_details=False: Array of new hull points from selected points
        If return_details=True: Dictionary with original hull, new hull, and ROC metrics
    """
    if len(points) < 3:
        if return_details:
            return {
                'original_hull': points,
                'new_hull': np.array([]),
                'selected_points': points,
                'all_points': points,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    points_array = np.array(points)
    
    # Filter to only points above diagonal (TPR > FPR)
    above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
    
    if len(above_diagonal) < 3:
        if return_details:
            return {
                'original_hull': above_diagonal,
                'new_hull': np.array([]),
                'selected_points': above_diagonal,
                'all_points': above_diagonal,
                'original_hull_area': 0,
                'new_hull_area': 0
            }
        return np.array([])
    
    # Calculate original convex hull
    extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
    
    try:
        original_hull = ConvexHull(extended_points)
        original_hull_area = original_hull.volume
        
        # Get original hull points (excluding anchors)
        original_hull_indices = [i - 1 for i in original_hull.vertices 
                                if 1 <= i <= len(above_diagonal)]
        original_hull_points = above_diagonal[original_hull_indices]
        
        # MODIFICATION: Option to exclude hull points from selection
        # This ensures the new curve will be different from the original
        if exclude_hull_points:
            # Select only from non-hull points
            non_hull_mask = np.ones(len(above_diagonal), dtype=bool)
            non_hull_mask[original_hull_indices] = False
            candidate_points = above_diagonal[non_hull_mask]
            
            if len(candidate_points) < 3:
                # Not enough non-hull points
                if return_details:
                    return {
                        'original_hull': original_hull_points,
                        'new_hull': np.array([]),
                        'selected_points': candidate_points,
                        'all_points': above_diagonal,
                        'original_hull_area': original_hull_area,
                        'new_hull_area': 0,
                        'reference_distance': 0,
                        'threshold_distance': 0,
                        'note': 'Not enough non-hull points to form new hull'
                    }
                return np.array([])
        else:
            # Include all points (original behavior - may include hull points)
            candidate_points = above_diagonal
        
        # NEW APPROACH: Use furthest point from diagonal as reference (same as function 4)
        # Calculate reference from CANDIDATE points only (after hull exclusion)
        # This ensures the threshold is based on achievable distances
        diagonal_distances_candidates = candidate_points[:, 1] - candidate_points[:, 0]
        max_diagonal_distance = np.max(diagonal_distances_candidates)
        
        if max_diagonal_distance <= 0:
            # No points above diagonal (should not happen due to filter above)
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': np.array([]),
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0,
                    'reference_distance': max_diagonal_distance,
                    'threshold_distance': 0,
                    'note': 'No valid points above diagonal'
                }
            return np.array([])
        
        # Step 2: Calculate threshold based on (100 - percentage) of max diagonal distance
        # Example: 1% parameter means 99% threshold
        threshold = max_diagonal_distance * ((100.0 - distance_percentage) / 100.0)
        
        # Step 3: Calculate diagonal distance for each candidate point
        diagonal_distances = candidate_points[:, 1] - candidate_points[:, 0]  # TPR - FPR
        
        # Step 4: Select points where diagonal_distance >= threshold
        selection_mask = diagonal_distances >= threshold
        selected_indices = np.where(selection_mask)[0]
        
        if len(selected_indices) == 0:
            # No points meet criteria, return empty
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': np.array([]),
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0,
                    'reference_distance': max_diagonal_distance,
                    'threshold_distance': threshold,
                    'n_selected': 0,
                    'note': f'No points above threshold {threshold:.4f} (max_diagonal_dist={max_diagonal_distance:.4f}, {distance_percentage}%)'
                }
            return np.array([])
        
        selected_points = candidate_points[selected_indices]
        n_select = len(selected_points)
        
        if len(selected_points) < 3:
            if return_details:
                return {
                    'original_hull': original_hull_points,
                    'new_hull': np.array([]),
                    'selected_points': selected_points,
                    'all_points': above_diagonal,
                    'original_hull_area': original_hull_area,
                    'new_hull_area': 0
                }
            return np.array([])
        
        # Calculate new hull with selected points
        new_extended_points = np.vstack([[0, 0], selected_points, [1, 1]])
        new_hull = ConvexHull(new_extended_points)
        new_hull_area = new_hull.volume
        
        # Get new hull points (excluding anchors)
        new_hull_indices = [i - 1 for i in new_hull.vertices 
                           if 1 <= i <= len(selected_points)]
        new_hull_points = selected_points[new_hull_indices]
        
        if return_details:
            # Calculate ROC metrics
            original_metrics = calculate_roc_metrics(original_hull_points)
            new_metrics = calculate_roc_metrics(new_hull_points)
            all_metrics = calculate_roc_metrics(above_diagonal)
            
            return {
                'original_hull': original_hull_points,
                'new_hull': new_hull_points,
                'selected_points': selected_points,
                'all_points': above_diagonal,
                'selection_criterion': f'above_diagonal_{distance_percentage}pct_threshold',
                'n_selected': n_select,
                'reference_distance': max_diagonal_distance,
                'threshold_distance': threshold,
                'distance_percentage': distance_percentage,
                'original_hull_area': original_hull_area,
                'new_hull_area': new_hull_area,
                'hull_area_reduction': original_hull_area - new_hull_area,
                'reduction_percentage': ((original_hull_area - new_hull_area) / original_hull_area * 100) 
                                       if original_hull_area > 0 else 0,
                # Original hull metrics
                'original_auc': original_metrics['auc'],
                'original_num_subgroups': original_metrics['num_points'],
                'original_best_tpr': original_metrics['best_tpr'],
                'original_best_fpr': original_metrics['best_fpr'],
                'original_avg_quality': original_metrics['avg_quality'],
                'original_max_quality': original_metrics['max_quality'],
                # New hull metrics
                'new_auc': new_metrics['auc'],
                'new_num_subgroups': new_metrics['num_points'],
                'new_best_tpr': new_metrics['best_tpr'],
                'new_best_fpr': new_metrics['best_fpr'],
                'new_avg_quality': new_metrics['avg_quality'],
                'new_max_quality': new_metrics['max_quality'],
                # All points metrics
                'all_points_auc': all_metrics['auc'],
                'all_points_num': all_metrics['num_points'],
                'all_points_max_quality': all_metrics['max_quality'],
                # Comparison metrics
                'auc_reduction': original_metrics['auc'] - new_metrics['auc'],
                'auc_reduction_percentage': ((original_metrics['auc'] - new_metrics['auc']) / original_metrics['auc'] * 100)
                                           if original_metrics['auc'] > 0 else 0,
                'quality_reduction': original_metrics['max_quality'] - new_metrics['max_quality'],
                # Diagonal distance metrics
                'avg_diagonal_distance': np.mean(diagonal_distances[selected_indices]),
                'max_diagonal_distance_selected': np.max(diagonal_distances[selected_indices]),
                'min_diagonal_distance_selected': np.min(diagonal_distances[selected_indices])
            }
        
        return new_hull_points
        
    except Exception as e:
        print(f"Error in above diagonal selection: {e}")
        if return_details:
            return {
                'original_hull': np.array([]),
                'new_hull': np.array([]),
                'selected_points': np.array([]),
                'all_points': points_array,
                'original_hull_area': 0,
                'new_hull_area': 0,
                'error': str(e)
            }
        return np.array([])


def plot_hull_comparison(hull_data, depth, output_path=None, title_suffix=""):
    """
    Create a visualization comparing original hull with recalculated hull after removing original hull points.
    
    Args:
        hull_data: Dictionary from remove_hull_points_and_recalculate with return_details=True
        depth: Search depth for labeling
        output_path: Path to save the plot (optional)
        title_suffix: Additional text for the title
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract data
    all_points = hull_data['all_points']
    original_hull = hull_data['original_hull']
    new_hull = hull_data.get('new_hull', np.array([]))
    # Handle both 'remaining_points' (hull removal) and 'selected_points' (other methods)
    remaining_points = hull_data.get('remaining_points', hull_data.get('selected_points', np.array([])))
    
    # Plot 1: Original hull
    ax1.plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.5, label='All points', markersize=6)
    if len(original_hull) > 0:
        # Sort hull points for proper polygon drawing
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        ax1.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2, label='Original hull')
        ax1.scatter(original_hull[:, 0], original_hull[:, 1], c='red', s=100, 
                   marker='*', zorder=5, label='Hull points', edgecolors='black')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.set_title(f'Original Hull (Depth {depth})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Remaining points and new hull
    if len(remaining_points) > 0 and remaining_points.ndim == 2:
        ax2.plot(remaining_points[:, 0], remaining_points[:, 1], 'go', alpha=0.5, 
                label='Remaining points', markersize=6)
    if len(new_hull) > 0:
        # Sort hull points for proper polygon drawing
        new_hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
        ax2.plot(new_hull_sorted[:, 0], new_hull_sorted[:, 1], 'purple', linewidth=2, 
                label='New hull')
        ax2.scatter(new_hull[:, 0], new_hull[:, 1], c='purple', s=100, 
                   marker='*', zorder=5, label='New hull points', edgecolors='black')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax2.set_xlabel('FPR')
    ax2.set_ylabel('TPR')
    ax2.set_title(f'New Hull After Removal (Depth {depth})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Comparison overlay
    ax3.plot(all_points[:, 0], all_points[:, 1], 'bo', alpha=0.3, label='All points', markersize=4)
    if len(original_hull) > 0:
        hull_sorted = original_hull[np.argsort(original_hull[:, 0])]
        ax3.plot(hull_sorted[:, 0], hull_sorted[:, 1], 'r-', linewidth=2, 
                alpha=0.7, label='Original hull')
    if len(new_hull) > 0:
        new_hull_sorted = new_hull[np.argsort(new_hull[:, 0])]
        ax3.plot(new_hull_sorted[:, 0], new_hull_sorted[:, 1], 'purple', 
                linewidth=2, linestyle='--', alpha=0.7, label='New hull')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Diagonal')
    ax3.set_xlabel('FPR')
    ax3.set_ylabel('TPR')
    
    # Add statistics to title
    area_reduction = hull_data.get('hull_area_reduction', 0)
    reduction_pct = hull_data.get('reduction_percentage', 0)
    ax3.set_title(f'Hull Comparison (Depth {depth})\n'
                 f'Area reduction: {area_reduction:.3f} ({reduction_pct:.1f}%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    plt.suptitle(f'ROC Convex Hull Comparison{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        # Create directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved hull comparison plot to: {output_path}")
    
    plt.show()
    plt.close()
    
    # Print detailed statistics
    print(f"\n=== Hull Comparison Statistics (Depth {depth}) ===")
    print(f"\n--- Point Counts ---")
    print(f"Total points: {len(all_points)}")
    print(f"Original hull points: {len(original_hull)}")
    print(f"Removed points: {hull_data.get('subgroups_removed', len(original_hull))}")
    print(f"Remaining points: {len(remaining_points)}")
    print(f"New hull points: {len(new_hull)}")
    
    print(f"\n--- Hull Area Metrics ---")
    print(f"Original hull area: {hull_data.get('original_hull_area', 0):.4f}")
    print(f"New hull area: {hull_data.get('new_hull_area', 0):.4f}")
    print(f"Area reduction: {area_reduction:.4f} ({reduction_pct:.1f}%)")
    
    print(f"\n--- AUC Metrics ---")
    print(f"Original hull AUC: {hull_data.get('original_auc', 0):.4f}")
    print(f"New hull AUC: {hull_data.get('new_auc', 0):.4f}")
    print(f"AUC reduction: {hull_data.get('auc_reduction', 0):.4f} ({hull_data.get('auc_reduction_percentage', 0):.1f}%)")
    print(f"All points AUC: {hull_data.get('all_points_auc', 0):.4f}")
    
    print(f"\n--- Quality Metrics ---")
    print(f"Original best quality (TPR-FPR): {hull_data.get('original_max_quality', 0):.4f}")
    print(f"Original avg quality: {hull_data.get('original_avg_quality', 0):.4f}")
    print(f"New best quality (TPR-FPR): {hull_data.get('new_max_quality', 0):.4f}")
    print(f"New avg quality: {hull_data.get('new_avg_quality', 0):.4f}")
    print(f"Quality reduction: {hull_data.get('quality_reduction', 0):.4f}")
    
    print(f"\n--- Best Subgroup (Original Hull) ---")
    print(f"Best TPR: {hull_data.get('original_best_tpr', 0):.4f}")
    print(f"Best FPR: {hull_data.get('original_best_fpr', 0):.4f}")
    
    print(f"\n--- Best Subgroup (New Hull) ---")
    print(f"Best TPR: {hull_data.get('new_best_tpr', 0):.4f}")
    print(f"Best FPR: {hull_data.get('new_best_fpr', 0):.4f}")


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
    print(f"{search_type} pruning: {len(subgroups)} -> {len(kept_subgroups)} subgroups (width: {len(kept_subgroups)})")
    
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
    
    # Store hull information for comparison
    hull_comparison = {
        'all_points': subgroups,
        'original_hull_points': points_on_hull,
        'original_hull_indices': hull_points_indices,
        'ch_eligible': ch_eligible
    }
    
    return candidates, hull_comparison

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
        
        # Calculate ROC quality for population
        population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], alpha)
        
        current_subgroups = [population_stats]
        all_subgroups = [population_stats]
        
        candidates_explored = 0
        depth_analysis = []  # Track statistics for each depth
        hull_comparisons = []  # Track hull comparisons for each depth
        
        # Add depth 0 (population) to analysis
        depth_analysis.append({
            'depth': 0,
            'subgroups_start': 1,
            'candidates_generated': 0,
            'subgroups_after_pruning': 1,
            'width': 1,
            'best_quality': population_stats['roc_quality'],
            'avg_coverage': population_stats['coverage'],
            'cumulative_candidates': 0
        })
        
        for depth in range(1, max_depth + 1):
            print(f"Depth {depth}: Starting with {len(current_subgroups)} subgroups")
            depth_start_subgroups = len(current_subgroups)
            
            # Generate candidates and get hull comparison data
            candidates, hull_comparison = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
            candidates_explored += len(candidates)
            depth_candidates = len(candidates)
            
            # Perform hull comparison analysis
            if len(hull_comparison.get('ch_eligible', [])) > 0:
                hull_data = remove_hull_points_and_recalculate(
                    hull_comparison['ch_eligible'], 
                    return_details=True
                )
                hull_data['depth'] = depth
                hull_comparisons.append(hull_data)
                
                print(f"Hull comparison at depth {depth}:")
                print(f"  Original hull points: {len(hull_data['original_hull'])}")
                print(f"  New hull points: {len(hull_data.get('new_hull', []))}")
                print(f"  Area reduction: {hull_data.get('hull_area_reduction', 0):.4f}")
            else:
                print(f"No hull comparison possible at depth {depth} (insufficient points)")
                hull_comparisons.append({
                    'depth': depth,
                    'original_hull': np.array([]),
                    'new_hull': np.array([]),
                    'error': 'Insufficient points above diagonal'
                })
            
            if not candidates:
                print(f"No valid candidates at depth {depth}")
                # Still record this depth with no progress
                depth_analysis.append({
                    'depth': depth,
                    'subgroups_start': depth_start_subgroups,
                    'candidates_generated': 0,
                    'subgroups_after_pruning': depth_start_subgroups,
                    'best_quality': max([sg.get('roc_quality', 0) for sg in current_subgroups]) if current_subgroups else 0,
                    'avg_coverage': np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0,
                    'cumulative_candidates': candidates_explored
                })
                break
            
            print(f"Generated {len(candidates)} candidates")
            
            # Combine current subgroups with new candidates
            all_candidates = current_subgroups + candidates
            
            # Apply adaptive ROC pruning
            current_subgroups = adaptive_roc_pruning(all_candidates, alpha)
            
            # Keep track of all subgroups found
            all_subgroups.extend(candidates)
            
            # Record depth statistics
            if current_subgroups:
                best_quality = max([sg.get('roc_quality', 0) for sg in current_subgroups])
                avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups])
            else:
                best_quality = 0
                avg_coverage = 0
            
            depth_analysis.append({
                'depth': depth,
                'subgroups_start': depth_start_subgroups,
                'candidates_generated': depth_candidates,
                'subgroups_after_pruning': len(current_subgroups),
                'width': len(current_subgroups),
                'best_quality': best_quality,
                'avg_coverage': avg_coverage,
                'cumulative_candidates': candidates_explored
            })
            
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
            'subgroups': final_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"Completed {mode_name}:")
        print(f"  Adaptive width: {adaptive_width}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC approximation: {auc:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
    
    return results


def hull_removal_search(data, target_col, max_depth=3, min_coverage=50):
    """
    ROC search using hull removal strategy: at each depth, remove hull points
    and recalculate with remaining points.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with search results
    """
    print("\n=== Hull Removal Search ===")
    start_time = time.time()
    
    # Initialize with population
    population_stats = calculate_subgroup_stats(data, [], target_col)
    if not population_stats or 'tpr' not in population_stats:
        print("Error: Could not calculate population statistics")
        return None
    
    population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], None)
    
    current_subgroups = [population_stats]
    all_subgroups = [population_stats]
    
    candidates_explored = 0
    depth_analysis = []
    hull_comparisons = []
    
    # Add depth 0
    depth_analysis.append({
        'depth': 0,
        'subgroups_start': 1,
        'candidates_generated': 0,
        'subgroups_after_pruning': 1,
        'width': 1,
        'best_quality': population_stats['roc_quality'],
        'avg_coverage': population_stats['coverage'],
        'cumulative_candidates': 0
    })
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Depth {depth} ---")
        
        # Generate candidates
        candidates, hull_comp = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
        candidates_explored += len(candidates)
        
        if not candidates:
            print(f"No candidates at depth {depth}")
            break
        
        print(f"Generated {len(candidates)} candidates")
        
        # Candidates already have stats calculated, just add roc_quality
        candidate_subgroups = []
        for stats in candidates:
            if stats and 'tpr' in stats:
                if 'roc_quality' not in stats:
                    stats['roc_quality'] = roc_quality_measure(stats['tpr'], stats['fpr'], None)
                candidate_subgroups.append(stats)
        
        all_subgroups.extend(candidate_subgroups)
        
        # Apply hull removal pruning
        roc_points = np.array([(sg['fpr'], sg['tpr']) for sg in candidate_subgroups])
        
        if len(roc_points) >= 3:
            hull_data = remove_hull_points_and_recalculate(roc_points, return_details=True)
            hull_data['depth'] = depth
            hull_comparisons.append(hull_data)
            
            # Keep only remaining points (non-hull points) for next iteration
            remaining_points = hull_data.get('remaining_points', np.array([]))
            
            if len(remaining_points) > 0:
                # Find subgroups corresponding to remaining points (optimized)
                kept_subgroups = []
                remaining_set = set(map(tuple, remaining_points))  # Convert to set of tuples for O(1) lookup
                
                for sg in candidate_subgroups:
                    sg_point = (round(sg['fpr'], 6), round(sg['tpr'], 6))  # Round to 6 decimals
                    if sg_point in remaining_set:
                        kept_subgroups.append(sg)
                
                current_subgroups = kept_subgroups if kept_subgroups else candidate_subgroups[:10]
            else:
                current_subgroups = candidate_subgroups[:10]
        else:
            current_subgroups = candidate_subgroups
        
        width = len(current_subgroups)
        best_quality = max([sg['roc_quality'] for sg in current_subgroups]) if current_subgroups else 0
        avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0
        
        depth_analysis.append({
            'depth': depth,
            'subgroups_start': len(candidate_subgroups),
            'candidates_generated': len(candidates),
            'subgroups_after_pruning': len(current_subgroups),
            'width': width,
            'best_quality': best_quality,
            'avg_coverage': avg_coverage,
            'cumulative_candidates': candidates_explored
        })
        
        print(f"After hull removal: {len(candidate_subgroups)} -> {width} subgroups")
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    if all_subgroups:
        roc_points = [(sg['fpr'], sg['tpr']) for sg in all_subgroups]
        roc_metrics = calculate_roc_metrics(roc_points)
        best_sg = max(all_subgroups, key=lambda x: x['roc_quality'])
        
        result = {
            'algorithm': 'Hull Removal',
            'adaptive_width': len(current_subgroups),
            'total_candidates': candidates_explored,
            'final_subgroups': len(all_subgroups),
            'search_time': elapsed_time,
            'auc_approx': roc_metrics['auc'],
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': all_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"\nCompleted Hull Removal Search:")
        print(f"  Final width: {result['adaptive_width']}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC: {roc_metrics['auc']:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
        
        return result
    
    return None


def closest_to_hull_search(data, target_col, n_points=10, max_depth=3, min_coverage=50):
    """
    ROC search using closest-to-hull strategy: at each depth, keep n points
    closest to the convex hull.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        n_points: Number of closest points to keep
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with search results
    """
    print(f"\n=== Closest to Hull Search (n={n_points}) ===")
    start_time = time.time()
    
    # Initialize with population
    population_stats = calculate_subgroup_stats(data, [], target_col)
    if not population_stats or 'tpr' not in population_stats:
        print("Error: Could not calculate population statistics")
        return None
    
    population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], None)
    
    current_subgroups = [population_stats]
    all_subgroups = [population_stats]
    
    candidates_explored = 0
    depth_analysis = []
    hull_comparisons = []
    
    # Add depth 0
    depth_analysis.append({
        'depth': 0,
        'subgroups_start': 1,
        'candidates_generated': 0,
        'subgroups_after_pruning': 1,
        'width': 1,
        'best_quality': population_stats['roc_quality'],
        'avg_coverage': population_stats['coverage'],
        'cumulative_candidates': 0
    })
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Depth {depth} ---")
        
        # Generate candidates
        candidates, hull_comp = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
        candidates_explored += len(candidates)
        
        if not candidates:
            print(f"No candidates at depth {depth}")
            break
        
        print(f"Generated {len(candidates)} candidates")
        
        # Candidates already have stats calculated, just add roc_quality
        candidate_subgroups = []
        for stats in candidates:
            if stats and 'tpr' in stats:
                if 'roc_quality' not in stats:
                    stats['roc_quality'] = roc_quality_measure(stats['tpr'], stats['fpr'], None)
                candidate_subgroups.append(stats)
        
        all_subgroups.extend(candidate_subgroups)
        
        # Apply closest-to-hull pruning
        roc_points = np.array([(sg['fpr'], sg['tpr']) for sg in candidate_subgroups])
        
        if len(roc_points) >= 3:
            hull_data = select_closest_points_to_hull(roc_points, n_points, return_details=True, exclude_hull_points=True)
            hull_data['depth'] = depth
            hull_comparisons.append(hull_data)
            
            new_hull_points = hull_data.get('new_hull', np.array([]))
            
            if len(new_hull_points) > 0:
                # Find subgroups corresponding to selected points (optimized)
                kept_subgroups = []
                selected_points = hull_data.get('selected_points', new_hull_points)
                selected_set = set(map(lambda pt: (round(pt[0], 6), round(pt[1], 6)), selected_points))
                
                for sg in candidate_subgroups:
                    sg_point = (round(sg['fpr'], 6), round(sg['tpr'], 6))
                    if sg_point in selected_set:
                        kept_subgroups.append(sg)
                
                current_subgroups = kept_subgroups if kept_subgroups else candidate_subgroups[:n_points]
            else:
                current_subgroups = candidate_subgroups[:n_points]
        else:
            current_subgroups = candidate_subgroups[:n_points]
        
        width = len(current_subgroups)
        best_quality = max([sg['roc_quality'] for sg in current_subgroups]) if current_subgroups else 0
        avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0
        
        depth_analysis.append({
            'depth': depth,
            'subgroups_start': len(candidate_subgroups),
            'candidates_generated': len(candidates),
            'subgroups_after_pruning': len(current_subgroups),
            'width': width,
            'best_quality': best_quality,
            'avg_coverage': avg_coverage,
            'cumulative_candidates': candidates_explored
        })
        
        print(f"After closest-to-hull selection: {len(candidate_subgroups)} -> {width} subgroups")
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    if all_subgroups:
        roc_points = [(sg['fpr'], sg['tpr']) for sg in all_subgroups]
        roc_metrics = calculate_roc_metrics(roc_points)
        best_sg = max(all_subgroups, key=lambda x: x['roc_quality'])
        
        result = {
            'algorithm': f'Closest to Hull (n={n_points})',
            'adaptive_width': len(current_subgroups),
            'total_candidates': candidates_explored,
            'final_subgroups': len(all_subgroups),
            'search_time': elapsed_time,
            'auc_approx': roc_metrics['auc'],
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': all_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"\nCompleted Closest to Hull Search:")
        print(f"  Final width: {result['adaptive_width']}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC: {roc_metrics['auc']:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
        
        return result
    
    return None


def furthest_from_diagonal_search(data, target_col, n_points=10, max_depth=3, min_coverage=50):
    """
    ROC search using furthest-from-diagonal strategy: at each depth, keep n points
    furthest from the diagonal (highest TPR - FPR).
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        n_points: Number of furthest points to keep
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with search results
    """
    print(f"\n=== Furthest from Diagonal Search (n={n_points}) ===")
    start_time = time.time()
    
    # Initialize with population
    population_stats = calculate_subgroup_stats(data, [], target_col)
    if not population_stats or 'tpr' not in population_stats:
        print("Error: Could not calculate population statistics")
        return None
    
    population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], None)
    
    current_subgroups = [population_stats]
    all_subgroups = [population_stats]
    
    candidates_explored = 0
    depth_analysis = []
    hull_comparisons = []
    
    # Add depth 0
    depth_analysis.append({
        'depth': 0,
        'subgroups_start': 1,
        'candidates_generated': 0,
        'subgroups_after_pruning': 1,
        'width': 1,
        'best_quality': population_stats['roc_quality'],
        'avg_coverage': population_stats['coverage'],
        'cumulative_candidates': 0
    })
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Depth {depth} ---")
        
        # Generate candidates
        candidates, hull_comp = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
        candidates_explored += len(candidates)
        
        if not candidates:
            print(f"No candidates at depth {depth}")
            break
        
        print(f"Generated {len(candidates)} candidates")
        
        # Candidates already have stats calculated, just add roc_quality
        candidate_subgroups = []
        for stats in candidates:
            if stats and 'tpr' in stats:
                if 'roc_quality' not in stats:
                    stats['roc_quality'] = roc_quality_measure(stats['tpr'], stats['fpr'], None)
                candidate_subgroups.append(stats)
        
        all_subgroups.extend(candidate_subgroups)
        
        # Apply furthest-from-diagonal pruning
        roc_points = np.array([(sg['fpr'], sg['tpr']) for sg in candidate_subgroups])
        
        if len(roc_points) >= 3:
            hull_data = select_furthest_points_from_diagonal(roc_points, n_points, return_details=True, exclude_hull_points=True)
            hull_data['depth'] = depth
            hull_comparisons.append(hull_data)
            
            new_hull_points = hull_data.get('new_hull', np.array([]))
            
            if len(new_hull_points) > 0:
                # Find subgroups corresponding to selected points (optimized)
                kept_subgroups = []
                selected_points = hull_data.get('selected_points', new_hull_points)
                selected_set = set(map(lambda pt: (round(pt[0], 6), round(pt[1], 6)), selected_points))
                
                for sg in candidate_subgroups:
                    sg_point = (round(sg['fpr'], 6), round(sg['tpr'], 6))
                    if sg_point in selected_set:
                        kept_subgroups.append(sg)
                
                current_subgroups = kept_subgroups if kept_subgroups else candidate_subgroups[:n_points]
            else:
                current_subgroups = candidate_subgroups[:n_points]
        else:
            current_subgroups = candidate_subgroups[:n_points]
        
        width = len(current_subgroups)
        best_quality = max([sg['roc_quality'] for sg in current_subgroups]) if current_subgroups else 0
        avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0
        
        depth_analysis.append({
            'depth': depth,
            'subgroups_start': len(candidate_subgroups),
            'candidates_generated': len(candidates),
            'subgroups_after_pruning': len(current_subgroups),
            'width': width,
            'best_quality': best_quality,
            'avg_coverage': avg_coverage,
            'cumulative_candidates': candidates_explored
        })
        
        print(f"After furthest-from-diagonal selection: {len(candidate_subgroups)} -> {width} subgroups")
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    if all_subgroups:
        roc_points = [(sg['fpr'], sg['tpr']) for sg in all_subgroups]
        roc_metrics = calculate_roc_metrics(roc_points)
        best_sg = max(all_subgroups, key=lambda x: x['roc_quality'])
        
        result = {
            'algorithm': f'Furthest from Diagonal (n={n_points})',
            'adaptive_width': len(current_subgroups),
            'total_candidates': candidates_explored,
            'final_subgroups': len(all_subgroups),
            'search_time': elapsed_time,
            'auc_approx': roc_metrics['auc'],
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': all_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"\nCompleted Furthest from Diagonal Search:")
        print(f"  Final width: {result['adaptive_width']}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC: {roc_metrics['auc']:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
        
        return result
    
    return None


def below_hull_threshold_search(data, target_col, distance_percentage=1.0, max_depth=3, min_coverage=50):
    """
    ROC search using below-hull threshold strategy: at each depth, keep points
    within percentage threshold below the convex hull.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        distance_percentage: Percentage threshold (default 1.0%)
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with search results
    """
    print(f"\n=== Below Hull Threshold Search (threshold={distance_percentage}%) ===")
    start_time = time.time()
    
    # Initialize with population
    population_stats = calculate_subgroup_stats(data, [], target_col)
    if not population_stats or 'tpr' not in population_stats:
        print("Error: Could not calculate population statistics")
        return None
    
    population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], None)
    
    current_subgroups = [population_stats]
    all_subgroups = [population_stats]
    
    candidates_explored = 0
    depth_analysis = []
    hull_comparisons = []
    
    # Add depth 0
    depth_analysis.append({
        'depth': 0,
        'subgroups_start': 1,
        'candidates_generated': 0,
        'subgroups_after_pruning': 1,
        'width': 1,
        'best_quality': population_stats['roc_quality'],
        'avg_coverage': population_stats['coverage'],
        'cumulative_candidates': 0
    })
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Depth {depth} ---")
        
        # Generate candidates
        candidates, hull_comp = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
        candidates_explored += len(candidates)
        
        if not candidates:
            print(f"No candidates at depth {depth}")
            break
        
        print(f"Generated {len(candidates)} candidates")
        
        # Candidates already have stats calculated, just add roc_quality
        candidate_subgroups = []
        for stats in candidates:
            if stats and 'tpr' in stats:
                if 'roc_quality' not in stats:
                    stats['roc_quality'] = roc_quality_measure(stats['tpr'], stats['fpr'], None)
                candidate_subgroups.append(stats)
        
        all_subgroups.extend(candidate_subgroups)
        
        # Apply below-hull threshold pruning
        roc_points = np.array([(sg['fpr'], sg['tpr']) for sg in candidate_subgroups])
        
        if len(roc_points) >= 3:
            hull_data = select_points_below_hull(roc_points, distance_percentage, return_details=True, exclude_hull_points=True)
            hull_data['depth'] = depth
            hull_comparisons.append(hull_data)
            
            new_hull_points = hull_data.get('new_hull', np.array([]))
            
            if len(new_hull_points) > 0:
                # Find subgroups corresponding to selected points (optimized)
                kept_subgroups = []
                selected_points = hull_data.get('selected_points', new_hull_points)
                selected_set = set(map(lambda pt: (round(pt[0], 6), round(pt[1], 6)), selected_points))
                
                for sg in candidate_subgroups:
                    sg_point = (round(sg['fpr'], 6), round(sg['tpr'], 6))
                    if sg_point in selected_set:
                        kept_subgroups.append(sg)
                
                current_subgroups = kept_subgroups if kept_subgroups else candidate_subgroups[:10]
            else:
                current_subgroups = candidate_subgroups[:10]
        else:
            current_subgroups = candidate_subgroups
        
        width = len(current_subgroups)
        best_quality = max([sg['roc_quality'] for sg in current_subgroups]) if current_subgroups else 0
        avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0
        
        depth_analysis.append({
            'depth': depth,
            'subgroups_start': len(candidate_subgroups),
            'candidates_generated': len(candidates),
            'subgroups_after_pruning': len(current_subgroups),
            'width': width,
            'best_quality': best_quality,
            'avg_coverage': avg_coverage,
            'cumulative_candidates': candidates_explored
        })
        
        print(f"After below-hull threshold selection: {len(candidate_subgroups)} -> {width} subgroups")
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    if all_subgroups:
        roc_points = [(sg['fpr'], sg['tpr']) for sg in all_subgroups]
        roc_metrics = calculate_roc_metrics(roc_points)
        best_sg = max(all_subgroups, key=lambda x: x['roc_quality'])
        
        result = {
            'algorithm': f'Below Hull Threshold ({distance_percentage}%)',
            'adaptive_width': len(current_subgroups),
            'total_candidates': candidates_explored,
            'final_subgroups': len(all_subgroups),
            'search_time': elapsed_time,
            'auc_approx': roc_metrics['auc'],
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': all_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"\nCompleted Below Hull Threshold Search:")
        print(f"  Final width: {result['adaptive_width']}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC: {roc_metrics['auc']:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
        
        return result
    
    return None


def above_diagonal_threshold_search(data, target_col, distance_percentage=1.0, max_depth=3, min_coverage=50):
    """
    ROC search using above-diagonal threshold strategy: at each depth, keep points
    above percentage threshold from the diagonal.
    
    Args:
        data: DataFrame with the data
        target_col: Name of target column
        distance_percentage: Percentage threshold (default 1.0%)
        max_depth: Maximum search depth
        min_coverage: Minimum coverage for subgroups
    
    Returns:
        Dictionary with search results
    """
    print(f"\n=== Above Diagonal Threshold Search (threshold={distance_percentage}%) ===")
    start_time = time.time()
    
    # Initialize with population
    population_stats = calculate_subgroup_stats(data, [], target_col)
    if not population_stats or 'tpr' not in population_stats:
        print("Error: Could not calculate population statistics")
        return None
    
    population_stats['roc_quality'] = roc_quality_measure(population_stats['tpr'], population_stats['fpr'], None)
    
    current_subgroups = [population_stats]
    all_subgroups = [population_stats]
    
    candidates_explored = 0
    depth_analysis = []
    hull_comparisons = []
    
    # Add depth 0
    depth_analysis.append({
        'depth': 0,
        'subgroups_start': 1,
        'candidates_generated': 0,
        'subgroups_after_pruning': 1,
        'width': 1,
        'best_quality': population_stats['roc_quality'],
        'avg_coverage': population_stats['coverage'],
        'cumulative_candidates': 0
    })
    
    for depth in range(1, max_depth + 1):
        print(f"\n--- Depth {depth} ---")
        
        # Generate candidates
        candidates, hull_comp = generate_candidates(data, target_col, current_subgroups, depth, min_coverage)
        candidates_explored += len(candidates)
        
        if not candidates:
            print(f"No candidates at depth {depth}")
            break
        
        print(f"Generated {len(candidates)} candidates")
        
        # Candidates already have stats calculated, just add roc_quality
        candidate_subgroups = []
        for stats in candidates:
            if stats and 'tpr' in stats:
                if 'roc_quality' not in stats:
                    stats['roc_quality'] = roc_quality_measure(stats['tpr'], stats['fpr'], None)
                candidate_subgroups.append(stats)
        
        all_subgroups.extend(candidate_subgroups)
        
        # Apply above-diagonal threshold pruning
        roc_points = np.array([(sg['fpr'], sg['tpr']) for sg in candidate_subgroups])
        
        if len(roc_points) >= 3:
            hull_data = select_points_above_diagonal(roc_points, distance_percentage, return_details=True, exclude_hull_points=True)
            hull_data['depth'] = depth
            hull_comparisons.append(hull_data)
            
            new_hull_points = hull_data.get('new_hull', np.array([]))
            
            if len(new_hull_points) > 0:
                # Find subgroups corresponding to selected points (optimized)
                kept_subgroups = []
                selected_points = hull_data.get('selected_points', new_hull_points)
                selected_set = set(map(lambda pt: (round(pt[0], 6), round(pt[1], 6)), selected_points))
                
                for sg in candidate_subgroups:
                    sg_point = (round(sg['fpr'], 6), round(sg['tpr'], 6))
                    if sg_point in selected_set:
                        kept_subgroups.append(sg)
                
                current_subgroups = kept_subgroups if kept_subgroups else candidate_subgroups[:10]
            else:
                current_subgroups = candidate_subgroups[:10]
        else:
            current_subgroups = candidate_subgroups
        
        width = len(current_subgroups)
        best_quality = max([sg['roc_quality'] for sg in current_subgroups]) if current_subgroups else 0
        avg_coverage = np.mean([sg['coverage'] for sg in current_subgroups]) if current_subgroups else 0
        
        depth_analysis.append({
            'depth': depth,
            'subgroups_start': len(candidate_subgroups),
            'candidates_generated': len(candidates),
            'subgroups_after_pruning': len(current_subgroups),
            'width': width,
            'best_quality': best_quality,
            'avg_coverage': avg_coverage,
            'cumulative_candidates': candidates_explored
        })
        
        print(f"After above-diagonal threshold selection: {len(candidate_subgroups)} -> {width} subgroups")
    
    # Calculate final results
    elapsed_time = time.time() - start_time
    
    if all_subgroups:
        roc_points = [(sg['fpr'], sg['tpr']) for sg in all_subgroups]
        roc_metrics = calculate_roc_metrics(roc_points)
        best_sg = max(all_subgroups, key=lambda x: x['roc_quality'])
        
        result = {
            'algorithm': f'Above Diagonal Threshold ({distance_percentage}%)',
            'adaptive_width': len(current_subgroups),
            'total_candidates': candidates_explored,
            'final_subgroups': len(all_subgroups),
            'search_time': elapsed_time,
            'auc_approx': roc_metrics['auc'],
            'best_quality': best_sg['roc_quality'],
            'best_tpr': best_sg['tpr'],
            'best_fpr': best_sg['fpr'],
            'best_precision': best_sg['precision'],
            'best_coverage': best_sg['coverage'],
            'subgroups': all_subgroups,
            'depth_analysis': depth_analysis,
            'hull_comparisons': hull_comparisons
        }
        
        print(f"\nCompleted Above Diagonal Threshold Search:")
        print(f"  Final width: {result['adaptive_width']}")
        print(f"  Total candidates: {candidates_explored}")
        print(f"  AUC: {roc_metrics['auc']:.3f}")
        print(f"  Best quality: {best_sg['roc_quality']:.3f}")
        print(f"  Search time: {elapsed_time:.2f}s")
        
        return result
    
    return None


def create_depth_analysis_table(results, output_dir):
    """Create and save depth-by-depth analysis table."""
    # Collect all depth data across different alphas
    all_depth_data = []
    
    for alpha, result in results.items():
        # Use algorithm name from result if available, otherwise format from alpha
        alpha_str = result.get('algorithm', 'Pure ROC' if alpha == 'pure_roc' or alpha is None else f'α = {alpha}')
        
        for depth_info in result['depth_analysis']:
            depth_info_copy = depth_info.copy()
            depth_info_copy['algorithm'] = alpha_str
            depth_info_copy['alpha_value'] = alpha
            # Add width column explicitly (it should already be in depth_info)
            if 'width' not in depth_info_copy:
                depth_info_copy['width'] = depth_info_copy['subgroups_after_pruning']
            # Add AUC and width from final results
            depth_info_copy['final_auc'] = result['auc_approx']
            depth_info_copy['final_width'] = result['adaptive_width']
            all_depth_data.append(depth_info_copy)
    
    # Create DataFrame
    depth_df = pd.DataFrame(all_depth_data)
    
    # Reorder columns for better readability
    column_order = [
        'algorithm', 'alpha_value', 'depth', 'subgroups_start', 
        'candidates_generated', 'subgroups_after_pruning', 'width',
        'best_quality', 'avg_coverage', 'cumulative_candidates',
        'final_auc', 'final_width'
    ]
    depth_df = depth_df[column_order]
    
    # Save detailed depth analysis
    depth_path = output_dir / 'depth_analysis.csv'
    depth_df.to_csv(depth_path, index=False)
    print(f"Saved depth analysis to: {depth_path}")
    
    # Create summary table by depth (for display)
    print("\n=== Depth-by-Depth Analysis ===")
    
    # Create a pivot-style display
    unique_alphas = depth_df['algorithm'].unique()
    max_depth = depth_df['depth'].max()
    
    # Print header
    print(f"{'Depth':<6}", end='')
    for alpha in unique_alphas:
        print(f"{alpha:<25}", end='')
    print()
    
    print("-" * (6 + 25 * len(unique_alphas)))
    
    # Print each depth's statistics
    for depth in range(int(max_depth) + 1):
        depth_data = depth_df[depth_df['depth'] == depth]
        
        # Subgroups after pruning and candidates
        print(f"{depth:<6}", end='')
        for alpha in unique_alphas:
            alpha_data = depth_data[depth_data['algorithm'] == alpha]
            if not alpha_data.empty:
                subgroups = alpha_data.iloc[0]['subgroups_after_pruning']
                candidates = alpha_data.iloc[0]['candidates_generated']
                print(f"S:{subgroups:>3} C:{candidates:>4}        ", end='')
            else:
                print(f"{'N/A':<25}", end='')
        print()
    
    print("\nLegend: S = Subgroups after pruning, C = Candidates generated")
    
    # Print final AUC and width summary
    print("\n=== Final Results Summary ===")
    print(f"{'Algorithm':<15} {'AUC':<8} {'Width':<8}")
    print("-" * 35)
    for alpha in unique_alphas:
        alpha_data = depth_df[depth_df['algorithm'] == alpha].iloc[0]
        auc = alpha_data['final_auc']
        width = alpha_data['final_width']
        print(f"{alpha:<15} {auc:<8.3f} {width:<8}")
    
    # Create a more detailed comparison table
    comparison_data = []
    for depth in range(int(max_depth) + 1):
        depth_data = depth_df[depth_df['depth'] == depth]
        
        row = {'depth': depth}
        for alpha in unique_alphas:
            alpha_data = depth_data[depth_data['algorithm'] == alpha]
            if not alpha_data.empty:
                info = alpha_data.iloc[0]
                row[f'{alpha}_subgroups'] = info['subgroups_after_pruning']
                row[f'{alpha}_candidates'] = info['candidates_generated']
                row[f'{alpha}_quality'] = round(info['best_quality'], 3)
                row[f'{alpha}_coverage'] = round(info['avg_coverage'], 1)
                row[f'{alpha}_auc'] = round(info['final_auc'], 3)
                row[f'{alpha}_width'] = info['final_width']
            else:
                row[f'{alpha}_subgroups'] = 0
                row[f'{alpha}_candidates'] = 0
                row[f'{alpha}_quality'] = 0
                row[f'{alpha}_coverage'] = 0
                row[f'{alpha}_auc'] = 0
                row[f'{alpha}_width'] = 0
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = output_dir / 'depth_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved depth comparison to: {comparison_path}")

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
    
    # Create and save depth analysis table
    create_depth_analysis_table(results, output_dir)
    
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
        
        # Save hull comparison data
        if 'hull_comparisons' in result and result['hull_comparisons']:
            hull_dir = alpha_dir / 'hull_comparisons'
            hull_dir.mkdir(exist_ok=True)
            
            hull_summary_data = []
            for hull_data in result['hull_comparisons']:
                depth = hull_data.get('depth', 'unknown')
                
                # Save comprehensive statistics
                hull_summary_data.append({
                    'depth': depth,
                    # Point counts
                    'total_points': len(hull_data.get('all_points', [])),
                    'original_hull_points': len(hull_data.get('original_hull', [])),
                    'new_hull_points': len(hull_data.get('new_hull', [])),
                    'remaining_points': len(hull_data.get('remaining_points', [])),
                    'subgroups_removed': hull_data.get('subgroups_removed', 0),
                    # Hull area metrics
                    'original_hull_area': hull_data.get('original_hull_area', 0),
                    'new_hull_area': hull_data.get('new_hull_area', 0),
                    'hull_area_reduction': hull_data.get('hull_area_reduction', 0),
                    'area_reduction_percentage': hull_data.get('reduction_percentage', 0),
                    # AUC metrics
                    'original_auc': hull_data.get('original_auc', 0),
                    'new_auc': hull_data.get('new_auc', 0),
                    'all_points_auc': hull_data.get('all_points_auc', 0),
                    'auc_reduction': hull_data.get('auc_reduction', 0),
                    'auc_reduction_percentage': hull_data.get('auc_reduction_percentage', 0),
                    # Quality metrics
                    'original_max_quality': hull_data.get('original_max_quality', 0),
                    'original_avg_quality': hull_data.get('original_avg_quality', 0),
                    'new_max_quality': hull_data.get('new_max_quality', 0),
                    'new_avg_quality': hull_data.get('new_avg_quality', 0),
                    'quality_reduction': hull_data.get('quality_reduction', 0),
                    # Best subgroup metrics
                    'original_best_tpr': hull_data.get('original_best_tpr', 0),
                    'original_best_fpr': hull_data.get('original_best_fpr', 0),
                    'new_best_tpr': hull_data.get('new_best_tpr', 0),
                    'new_best_fpr': hull_data.get('new_best_fpr', 0)
                })
                
                # Create visualization for this depth
                if 'all_points' in hull_data and len(hull_data.get('original_hull', [])) > 0:
                    plot_path = hull_dir / f'hull_comparison_depth_{depth}.png'
                    plot_hull_comparison(hull_data, depth, output_path=plot_path, 
                                       title_suffix=f' (α={alpha})')
            
            # Save hull comparison summary
            if hull_summary_data:
                hull_summary_df = pd.DataFrame(hull_summary_data)
                hull_summary_path = hull_dir / 'hull_comparison_summary.csv'
                hull_summary_df.to_csv(hull_summary_path, index=False)
                print(f"Saved hull comparison summary to: {hull_summary_path}")
        
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

def get_dataset_info():
    """Define dataset information with target columns."""
    return {
        'adult.txt': 'target',
        'mushroom.txt': 'poisonous',
        'ionosphere.txt': 'Attribute35',
        'Credit-a.txt': 'A16',
        'tic-tac-toe.txt': 'class',
        'wisconsin.txt': 'Class',
        'Covertype.txt': 'Cover_Type',
        'YPMSD.txt': '',
    }

def preprocess_categorical_data(df):
    """
    Simple preprocessing to convert categorical data to numerical.
    Works without external dependencies by mapping unique values to integers.
    Also filters out rows with missing values ('?', NaN, empty strings).
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # First, filter out rows with missing values in ANY column
    # Common missing value indicators: '?', NaN, None, empty strings
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':  # String/categorical column
            df_processed = df_processed[df_processed[col].notna()]
            df_processed = df_processed[df_processed[col] != '?']
            df_processed = df_processed[df_processed[col] != '']
            df_processed = df_processed[df_processed[col].str.strip() != '']
    
    # Reset index after filtering
    df_processed = df_processed.reset_index(drop=True)
    
    # Now process each column to convert categorical to numerical
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':  # String/categorical column
            # Get unique values and create mapping
            unique_vals = df_processed[col].unique()
            # Create mapping: first unique value -> 0, second -> 1, etc.
            mapping = {val: i for i, val in enumerate(unique_vals)}
            # Apply mapping
            df_processed[col] = df_processed[col].map(mapping)
    
    return df_processed

def run_batch_analysis(data_dir, alphas=None, depth=3, min_coverage=50, output_dir='./runs/batch_analysis'):
    """
    Run ROC search on multiple datasets and consolidate results.
    
    Args:
        data_dir: Directory containing dataset files
        alphas: List of alpha values (None for pure ROC)
        depth: Maximum search depth
        min_coverage: Minimum subgroup coverage
        output_dir: Output directory for consolidated results
    
    Returns:
        Dictionary containing consolidated results
    """
    dataset_info = get_dataset_info()
    consolidated_results = {}
    all_depth_analysis = []
    all_summaries = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Batch ROC Analysis ===")
    print(f"Processing datasets from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process each dataset
    for filename, target_col in dataset_info.items():
        data_path = Path(data_dir) / filename
        
        if not data_path.exists():
            print(f"Dataset not found: {data_path}")
            continue
            
        print(f"\n=== Processing {filename} ===")
        print(f"Target column: {target_col}")
        
        # Load data
        data = load_data(str(data_path))
        if data is None:
            print(f"Failed to load {filename}")
            continue
        
        # Preprocess categorical data to numerical
        data = preprocess_categorical_data(data)
        print(f"Data shape after preprocessing: {data.shape}")
        
        dataset_name = filename.replace('.txt', '')
        dataset_results = {}
            
        # Run all 6 search methods
        print(f"\n{'='*60}")
        print(f"Running all search methods on {dataset_name}")
        print(f"{'='*60}")
        
        # Method 1: Pure ROC Search (original)
        if alphas is None:
            print("\n>>> Method 1: Pure ROC Search")
            result = true_roc_search(data, target_col, None, depth, min_coverage)
            if result:
                result['pure_roc']['dataset'] = dataset_name
                dataset_results['pure_roc'] = result['pure_roc']
        
        # Method 2: Hull Removal Search
        print("\n>>> Method 2: Hull Removal Search")
        result = hull_removal_search(data, target_col, depth, min_coverage)
        if result:
            result['dataset'] = dataset_name
            dataset_results['hull_removal'] = result
        
        # Method 3: Closest to Hull Search
        print("\n>>> Method 3: Closest to Hull Search")
        result = closest_to_hull_search(data, target_col, n_points=10, max_depth=depth, min_coverage=min_coverage)
        if result:
            result['dataset'] = dataset_name
            dataset_results['closest_to_hull'] = result
        
        # Method 4: Furthest from Diagonal Search
        print("\n>>> Method 4: Furthest from Diagonal Search")
        result = furthest_from_diagonal_search(data, target_col, n_points=10, max_depth=depth, min_coverage=min_coverage)
        if result:
            result['dataset'] = dataset_name
            dataset_results['furthest_diagonal'] = result
        
        # Method 5: Below Hull Threshold Search
        print("\n>>> Method 5: Below Hull Threshold Search")
        result = below_hull_threshold_search(data, target_col, distance_percentage=1.0, max_depth=depth, min_coverage=min_coverage)
        if result:
            result['dataset'] = dataset_name
            dataset_results['below_hull'] = result
        
        # Method 6: Above Diagonal Threshold Search
        print("\n>>> Method 6: Above Diagonal Threshold Search")
        result = above_diagonal_threshold_search(data, target_col, distance_percentage=1.0, max_depth=depth, min_coverage=min_coverage)
        if result:
            result['dataset'] = dataset_name
            dataset_results['above_diagonal'] = result
        
        if not dataset_results:
            print(f"No results for {dataset_name}")
            continue
            
        consolidated_results[dataset_name] = dataset_results
        
        # Save individual dataset results
        dataset_output_dir = Path(output_dir) / dataset_name
        save_results(dataset_results, str(dataset_output_dir))
        
        # Collect depth analysis and summaries for consolidated tables
        depth_file = dataset_output_dir / 'depth_analysis.csv'
        if depth_file.exists():
            depth_df = pd.read_csv(depth_file)
            depth_df['dataset'] = dataset_name
            all_depth_analysis.append(depth_df)
            
        # Extract summary info
        for algorithm_key, result in dataset_results.items():
            summary_row = {
                'dataset': dataset_name,
                'algorithm': result.get('algorithm', result.get('mode', algorithm_key)),
                'alpha': result.get('alpha', 'N/A'),
                'adaptive_width': result['adaptive_width'],
                'auc_approx': result['auc_approx'],
                'best_quality': result['best_quality'],
                'best_tpr': result.get('best_tpr', 0),
                'best_fpr': result.get('best_fpr', 0),
                'total_candidates': result['total_candidates'],
                'search_time': result['search_time'],
                'target_column': target_col,
                'data_size': len(data)
            }
            all_summaries.append(summary_row)
    
    # Create consolidated tables
    if all_depth_analysis:
        consolidated_depth_df = pd.concat(all_depth_analysis, ignore_index=True)
        consolidated_depth_path = Path(output_dir) / 'consolidated_depth_analysis.csv'
        consolidated_depth_df.to_csv(consolidated_depth_path, index=False)
        print(f"\nSaved consolidated depth analysis: {consolidated_depth_path}")
    
    if all_summaries:
        consolidated_summary_df = pd.DataFrame(all_summaries)
        consolidated_summary_path = Path(output_dir) / 'consolidated_summary.csv'
        consolidated_summary_df.to_csv(consolidated_summary_path, index=False)
        print(f"Saved consolidated summary: {consolidated_summary_path}")
        
        # Create consolidated visualization
        create_consolidated_plots(consolidated_summary_df, output_dir)
    
    return consolidated_results

def create_consolidated_plots(summary_df, output_dir):
    """Create consolidated visualization plots."""
    
    # 1. AUC comparison across datasets
    plt.figure(figsize=(12, 8))
    
    datasets = summary_df['dataset'].unique()
    algorithms = summary_df['algorithm'].unique()
    
    x = np.arange(len(datasets))
    width = 0.8 / len(algorithms)
    
    for i, algorithm in enumerate(algorithms):
        alg_data = summary_df[summary_df['algorithm'] == algorithm]
        aucs = []
        for dataset in datasets:
            dataset_auc = alg_data[alg_data['dataset'] == dataset]['auc_approx']
            aucs.append(dataset_auc.iloc[0] if len(dataset_auc) > 0 else 0)
        
        plt.bar(x + i * width, aucs, width, label=algorithm, alpha=0.8)
    
    plt.xlabel('Dataset')
    plt.ylabel('AUC Score')
    plt.title('AUC Comparison Across Datasets and Algorithms')
    plt.xticks(x + width * (len(algorithms) - 1) / 2, datasets, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    auc_plot_path = Path(output_dir) / 'consolidated_auc_comparison.png'
    plt.savefig(auc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved AUC comparison plot: {auc_plot_path}")
    
    # 2. Quality vs Width scatter plot
    plt.figure(figsize=(10, 8))
    
    for algorithm in algorithms:
        alg_data = summary_df[summary_df['algorithm'] == algorithm]
        plt.scatter(alg_data['adaptive_width'], alg_data['best_quality'], 
                   label=algorithm, alpha=0.7, s=100)
        
        # Add dataset labels
        for _, row in alg_data.iterrows():
            plt.annotate(row['dataset'], 
                        (row['adaptive_width'], row['best_quality']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Adaptive Width')
    plt.ylabel('Best Quality')
    plt.title('Quality vs Width Across Datasets')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    quality_plot_path = Path(output_dir) / 'consolidated_quality_width.png'
    plt.savefig(quality_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved quality vs width plot: {quality_plot_path}")
    
    # 3. Performance summary table
    create_performance_summary_table(summary_df, output_dir)

def demonstrate_hull_comparison(points, depth=1, output_dir=None):
    """
    Standalone demonstration of hull comparison functionality.
    
    Args:
        points: Array of (fpr, tpr) points
        depth: Depth label for visualization
        output_dir: Optional output directory for saving plots
    
    Returns:
        Dictionary with hull comparison results
    """
    print(f"\n=== Hull Comparison Demonstration (Depth {depth}) ===")
    
    # Get detailed hull comparison
    hull_data = remove_hull_points_and_recalculate(points, return_details=True)
    hull_data['depth'] = depth
    
    # Print statistics
    print(f"Total points: {len(hull_data.get('all_points', []))}")
    print(f"Original hull points: {len(hull_data['original_hull'])}")
    print(f"Points removed: {len(hull_data['removed_points'])}")
    print(f"Remaining points: {len(hull_data['remaining_points'])}")
    print(f"New hull points: {len(hull_data.get('new_hull', []))}")
    print(f"Original hull area: {hull_data.get('original_hull_area', 0):.4f}")
    print(f"New hull area: {hull_data.get('new_hull_area', 0):.4f}")
    print(f"Area reduction: {hull_data.get('hull_area_reduction', 0):.4f}")
    print(f"Reduction percentage: {hull_data.get('reduction_percentage', 0):.1f}%")
    
    # Create visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'hull_comparison_demo_depth_{depth}.png'
        plot_hull_comparison(hull_data, depth, output_path=output_path)
    else:
        plot_hull_comparison(hull_data, depth)
    
    return hull_data

def create_performance_summary_table(summary_df, output_dir):
    """Create a formatted performance summary table."""
    
    print("\n=== Consolidated Performance Summary ===")
    print("Dataset".ljust(15) + "Algorithm".ljust(20) + "AUC".ljust(8) + "Width".ljust(8) + "Quality".ljust(10) + "Time(s)".ljust(10))
    print("=" * 80)
    
    for _, row in summary_df.iterrows():
        dataset = str(row['dataset']).ljust(15)
        algorithm = str(row['algorithm']).ljust(20)
        auc = f"{row['auc_approx']:.3f}".ljust(8)
        width = str(int(row['adaptive_width'])).ljust(8)
        quality = f"{row['best_quality']:.3f}".ljust(10)
        time_str = f"{row['search_time']:.2f}".ljust(10)
        
        print(f"{dataset}{algorithm}{auc}{width}{quality}{time_str}")
    """Create a formatted performance summary table."""
    
    print("\n=== Consolidated Performance Summary ===")
    print("Dataset".ljust(15) + "Algorithm".ljust(20) + "AUC".ljust(8) + "Width".ljust(8) + "Quality".ljust(10) + "Time(s)".ljust(10))
    print("=" * 80)
    
    for _, row in summary_df.iterrows():
        dataset = str(row['dataset']).ljust(15)
        algorithm = str(row['algorithm']).ljust(20)
        auc = f"{row['auc_approx']:.3f}".ljust(8)
        width = str(int(row['adaptive_width'])).ljust(8)
        quality = f"{row['best_quality']:.3f}".ljust(10)
        time_str = f"{row['search_time']:.2f}".ljust(10)
        
        print(f"{dataset}{algorithm}{auc}{width}{quality}{time_str}")

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
    parser.add_argument('--batch', action='store_true', 
                       help='Run batch analysis on all datasets in tests directory')
    parser.add_argument('--data-dir', default='./tests', 
                       help='Directory containing datasets (for batch mode)')
    
    args = parser.parse_args()
    
    # Determine search mode
    if args.pure_roc or args.alphas is None:
        alphas = None
        search_mode = "Pure ROC Search (no alpha)"
    else:
        alphas = args.alphas
        search_mode = f"Alpha-ROC Search (alphas: {alphas})"
    
    if args.batch:
        # Run batch analysis
        print("=== Batch ROC Analysis ===")
        print(f"Data directory: {args.data_dir}")
        print(f"Search mode: {search_mode}")
        print(f"Max depth: {args.depth}")
        print(f"Min coverage: {args.min_coverage}")
        print(f"Output: {args.output}")
        
        results = run_batch_analysis(
            args.data_dir, 
            alphas, 
            args.depth, 
            args.min_coverage, 
            args.output
        )
        
        if results:
            print(f"\n=== Batch Analysis Complete ===")
            print(f"Processed {len(results)} datasets")
            print(f"Results saved to: {args.output}")
    else:
        # Run single dataset analysis
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