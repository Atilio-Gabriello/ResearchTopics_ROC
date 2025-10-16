"""
Comprehensive Test Script for All ROC Search Methods

This script runs all 6 ROC search methods on all datasets in the tests folder:
1. Basic True ROC Search (adaptive pruning)
2. Remove Hull Points
3. Select Closest Points to Hull
4. Select Furthest Points from Diagonal
5. Select Points Below Hull (percentage-based)
6. Select Points Above Diagonal (percentage-based)

Runs at depth 4 with all methods tested at each depth level (1-4) with multiple parameters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

# Import functions from true_roc_search
from true_roc_search import (
    load_data,
    calculate_subgroup_stats,
    preprocess_categorical_data,
    remove_hull_points_and_recalculate,
    select_closest_points_to_hull,
    select_furthest_points_from_diagonal,
    select_points_below_hull,
    select_points_above_diagonal,
    calculate_roc_metrics,
    adaptive_roc_pruning,
    generate_candidates,
    get_dataset_info
)


class ComprehensiveROCTester:
    """Comprehensive tester for all ROC search methods."""
    
    def __init__(self, data_dir='./tests', output_dir='./runs/comprehensive_all_methods', max_depth=4):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_depth = max_depth
        self.dataset_info = get_dataset_info()
        
        # Test parameters for each method
        self.test_params = {
            'closest_points': [10, 20, 50, 100],  # n_points values
            'furthest_points': [10, 20, 50, 100],  # n_points values
            'below_hull': [0.5, 1.0, 2.0, 5.0, 10.0],  # distance_percentage values
            'above_diagonal': [0.5, 1.0, 2.0, 5.0, 10.0]  # distance_percentage values
        }
        
        self.all_results = []
        
    def run_basic_roc_search(self, data, target_col, dataset_name):
        """Run basic ROC search (method 1) to depth 4."""
        print(f"\n{'='*60}")
        print(f"Method 1: Basic True ROC Search (Adaptive Pruning)")
        print(f"{'='*60}")
        
        results_by_depth = {}
        
        # Initialize with population
        population_stats = calculate_subgroup_stats(data, [], target_col)
        if not population_stats or 'tpr' not in population_stats:
            print("Failed to calculate population statistics")
            return results_by_depth
        
        population_stats['roc_quality'] = population_stats['tpr'] - population_stats['fpr']
        current_subgroups = [population_stats]
        all_subgroups = [population_stats]
        
        # Run through each depth
        for depth in range(1, self.max_depth + 1):
            print(f"\n--- Depth {depth} ---")
            start_time = time.time()
            
            # Generate candidates
            candidates, hull_comparison = generate_candidates(
                data, target_col, current_subgroups, depth, min_coverage=10
            )
            
            print(f"Generated {len(candidates)} candidates")
            
            if not candidates:
                print(f"No candidates at depth {depth}")
                break
            
            # Add to all subgroups
            all_subgroups.extend(candidates)
            
            # Apply adaptive pruning
            current_subgroups = adaptive_roc_pruning(candidates, alpha=None, quality_threshold=None)
            
            # Extract ROC points for analysis
            roc_points = np.array([[sg['fpr'], sg['tpr']] for sg in all_subgroups])
            
            # Calculate metrics
            metrics = calculate_roc_metrics(roc_points)
            
            elapsed = time.time() - start_time
            
            results_by_depth[depth] = {
                'method': 'basic_roc_search',
                'dataset': dataset_name,
                'depth': depth,
                'num_subgroups': len(all_subgroups),
                'num_candidates': len(candidates),
                'pruned_subgroups': len(current_subgroups),
                'auc': metrics['auc'],
                'best_tpr': metrics['best_tpr'],
                'best_fpr': metrics['best_fpr'],
                'max_quality': metrics['max_quality'],
                'avg_quality': metrics['avg_quality'],
                'time_seconds': elapsed,
                'all_points': roc_points
            }
            
            print(f"Subgroups: {len(all_subgroups)}, Pruned to: {len(current_subgroups)}")
            print(f"AUC: {metrics['auc']:.4f}, Max Quality: {metrics['max_quality']:.4f}")
            print(f"Time: {elapsed:.2f}s")
        
        return results_by_depth
    
    def run_remove_hull_points(self, base_results, dataset_name):
        """Run method 2: Remove hull points at each depth."""
        print(f"\n{'='*60}")
        print(f"Method 2: Remove Hull Points and Recalculate")
        print(f"{'='*60}")
        
        results = []
        
        for depth, base_result in base_results.items():
            print(f"\n--- Depth {depth} ---")
            start_time = time.time()
            
            points = base_result['all_points']
            
            if len(points) < 3:
                print(f"Insufficient points at depth {depth}")
                continue
            
            # Run hull removal
            hull_data = remove_hull_points_and_recalculate(points, return_details=True)
            
            elapsed = time.time() - start_time
            
            result = {
                'method': 'remove_hull_points',
                'dataset': dataset_name,
                'depth': depth,
                'param_type': None,
                'param_value': None,
                'total_points': len(points),
                'original_hull_points': len(hull_data.get('original_hull', [])),
                'removed_points': hull_data.get('subgroups_removed', 0),
                'new_hull_points': len(hull_data.get('new_hull', [])),
                'original_auc': hull_data.get('original_auc', 0),
                'new_auc': hull_data.get('new_auc', 0),
                'auc_reduction': hull_data.get('auc_reduction', 0),
                'original_area': hull_data.get('original_hull_area', 0),
                'new_area': hull_data.get('new_hull_area', 0),
                'area_reduction': hull_data.get('hull_area_reduction', 0),
                'original_max_quality': hull_data.get('original_max_quality', 0),
                'new_max_quality': hull_data.get('new_max_quality', 0),
                'time_seconds': elapsed
            }
            
            results.append(result)
            
            print(f"Original hull: {result['original_hull_points']} points, "
                  f"New hull: {result['new_hull_points']} points")
            print(f"AUC reduction: {result['auc_reduction']:.4f}, "
                  f"Area reduction: {result['area_reduction']:.4f}")
            print(f"Time: {elapsed:.2f}s")
        
        return results
    
    def run_closest_points_to_hull(self, base_results, dataset_name):
        """Run method 3: Select closest points to hull with multiple n_points."""
        print(f"\n{'='*60}")
        print(f"Method 3: Select Closest Points to Hull")
        print(f"{'='*60}")
        
        results = []
        
        for depth, base_result in base_results.items():
            points = base_result['all_points']
            
            if len(points) < 3:
                continue
            
            for n_points in self.test_params['closest_points']:
                print(f"\n--- Depth {depth}, n_points={n_points} ---")
                start_time = time.time()
                
                hull_data = select_closest_points_to_hull(
                    points, n_points, return_details=True, exclude_hull_points=True
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'method': 'closest_points_to_hull',
                    'dataset': dataset_name,
                    'depth': depth,
                    'param_type': 'n_points',
                    'param_value': n_points,
                    'total_points': len(points),
                    'selected_points': hull_data.get('num_selected', 0),
                    'new_hull_points': len(hull_data.get('new_hull', [])),
                    'original_auc': hull_data.get('original_auc', 0),
                    'new_auc': hull_data.get('new_auc', 0),
                    'auc_reduction': hull_data.get('auc_reduction', 0),
                    'original_area': hull_data.get('original_hull_area', 0),
                    'new_area': hull_data.get('new_hull_area', 0),
                    'area_reduction': hull_data.get('hull_area_reduction', 0),
                    'original_max_quality': hull_data.get('original_max_quality', 0),
                    'new_max_quality': hull_data.get('new_max_quality', 0),
                    'avg_distance_to_hull': hull_data.get('avg_distance', 0),
                    'time_seconds': elapsed
                }
                
                results.append(result)
                
                print(f"Selected: {result['selected_points']} → {result['new_hull_points']} hull points")
                print(f"AUC: {result['original_auc']:.4f} → {result['new_auc']:.4f}")
                print(f"Time: {elapsed:.2f}s")
        
        return results
    
    def run_furthest_points_from_diagonal(self, base_results, dataset_name):
        """Run method 4: Select furthest points from diagonal with multiple n_points."""
        print(f"\n{'='*60}")
        print(f"Method 4: Select Furthest Points from Diagonal")
        print(f"{'='*60}")
        
        results = []
        
        for depth, base_result in base_results.items():
            points = base_result['all_points']
            
            if len(points) < 3:
                continue
            
            for n_points in self.test_params['furthest_points']:
                print(f"\n--- Depth {depth}, n_points={n_points} ---")
                start_time = time.time()
                
                hull_data = select_furthest_points_from_diagonal(
                    points, n_points, return_details=True, exclude_hull_points=True
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'method': 'furthest_from_diagonal',
                    'dataset': dataset_name,
                    'depth': depth,
                    'param_type': 'n_points',
                    'param_value': n_points,
                    'total_points': len(points),
                    'selected_points': hull_data.get('num_selected', 0),
                    'new_hull_points': len(hull_data.get('new_hull', [])),
                    'original_auc': hull_data.get('original_auc', 0),
                    'new_auc': hull_data.get('new_auc', 0),
                    'auc_reduction': hull_data.get('auc_reduction', 0),
                    'original_area': hull_data.get('original_hull_area', 0),
                    'new_area': hull_data.get('new_hull_area', 0),
                    'area_reduction': hull_data.get('hull_area_reduction', 0),
                    'original_max_quality': hull_data.get('original_max_quality', 0),
                    'new_max_quality': hull_data.get('new_max_quality', 0),
                    'min_diagonal_distance': hull_data.get('min_diagonal_distance', 0),
                    'max_diagonal_distance': hull_data.get('max_diagonal_distance', 0),
                    'time_seconds': elapsed
                }
                
                results.append(result)
                
                print(f"Selected: {result['selected_points']} → {result['new_hull_points']} hull points")
                print(f"Quality range: [{result['min_diagonal_distance']:.4f}, {result['max_diagonal_distance']:.4f}]")
                print(f"Time: {elapsed:.2f}s")
        
        return results
    
    def run_select_points_below_hull(self, base_results, dataset_name):
        """Run method 5: Select points below hull with multiple distance percentages."""
        print(f"\n{'='*60}")
        print(f"Method 5: Select Points Below Hull (Percentage-based)")
        print(f"{'='*60}")
        
        results = []
        
        for depth, base_result in base_results.items():
            points = base_result['all_points']
            
            if len(points) < 3:
                continue
            
            for distance_pct in self.test_params['below_hull']:
                print(f"\n--- Depth {depth}, distance_percentage={distance_pct}% ---")
                start_time = time.time()
                
                hull_data = select_points_below_hull(
                    points, distance_percentage=distance_pct, 
                    return_details=True, exclude_hull_points=True
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'method': 'below_hull_percentage',
                    'dataset': dataset_name,
                    'depth': depth,
                    'param_type': 'distance_percentage',
                    'param_value': distance_pct,
                    'total_points': len(points),
                    'selected_points': hull_data.get('num_selected', 0),
                    'new_hull_points': len(hull_data.get('new_hull', [])),
                    'original_auc': hull_data.get('original_auc', 0),
                    'new_auc': hull_data.get('new_auc', 0),
                    'auc_reduction': hull_data.get('auc_reduction', 0),
                    'original_area': hull_data.get('original_hull_area', 0),
                    'new_area': hull_data.get('new_hull_area', 0),
                    'area_reduction': hull_data.get('hull_area_reduction', 0),
                    'original_max_quality': hull_data.get('original_max_quality', 0),
                    'new_max_quality': hull_data.get('new_max_quality', 0),
                    'threshold_used': hull_data.get('threshold', 0),
                    'max_diagonal_distance': hull_data.get('max_diagonal_distance', 0),
                    'time_seconds': elapsed
                }
                
                results.append(result)
                
                print(f"Selected: {result['selected_points']} → {result['new_hull_points']} hull points")
                print(f"Threshold: {result['threshold_used']:.4f} (from max: {result['max_diagonal_distance']:.4f})")
                print(f"Time: {elapsed:.2f}s")
        
        return results
    
    def run_select_points_above_diagonal(self, base_results, dataset_name):
        """Run method 6: Select points above diagonal with multiple distance percentages."""
        print(f"\n{'='*60}")
        print(f"Method 6: Select Points Above Diagonal (Percentage-based)")
        print(f"{'='*60}")
        
        results = []
        
        for depth, base_result in base_results.items():
            points = base_result['all_points']
            
            if len(points) < 3:
                continue
            
            for distance_pct in self.test_params['above_diagonal']:
                print(f"\n--- Depth {depth}, distance_percentage={distance_pct}% ---")
                start_time = time.time()
                
                hull_data = select_points_above_diagonal(
                    points, distance_percentage=distance_pct, 
                    return_details=True, exclude_hull_points=True
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'method': 'above_diagonal_percentage',
                    'dataset': dataset_name,
                    'depth': depth,
                    'param_type': 'distance_percentage',
                    'param_value': distance_pct,
                    'total_points': len(points),
                    'selected_points': hull_data.get('num_selected', 0),
                    'new_hull_points': len(hull_data.get('new_hull', [])),
                    'original_auc': hull_data.get('original_auc', 0),
                    'new_auc': hull_data.get('new_auc', 0),
                    'auc_reduction': hull_data.get('auc_reduction', 0),
                    'original_area': hull_data.get('original_hull_area', 0),
                    'new_area': hull_data.get('new_hull_area', 0),
                    'area_reduction': hull_data.get('hull_area_reduction', 0),
                    'original_max_quality': hull_data.get('original_max_quality', 0),
                    'new_max_quality': hull_data.get('new_max_quality', 0),
                    'threshold_used': hull_data.get('threshold', 0),
                    'max_diagonal_distance': hull_data.get('max_diagonal_distance', 0),
                    'time_seconds': elapsed
                }
                
                results.append(result)
                
                print(f"Selected: {result['selected_points']} → {result['new_hull_points']} hull points")
                print(f"Threshold: {result['threshold_used']:.4f} (from max: {result['max_diagonal_distance']:.4f})")
                print(f"Time: {elapsed:.2f}s")
        
        return results
    
    def run_all_methods_on_dataset(self, dataset_name, target_col):
        """Run all 6 methods on a single dataset."""
        print(f"\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset_name}")
        print(f"# Target column: {target_col}")
        print(f"{'#'*80}")
        
        # Load data
        data_path = self.data_dir / f"{dataset_name}.txt"
        if not data_path.exists():
            print(f"Dataset not found: {data_path}")
            return
        
        data = load_data(str(data_path))
        if data is None:
            print(f"Failed to load dataset: {dataset_name}")
            return
        
        # Preprocess
        data = preprocess_categorical_data(data)
        print(f"Preprocessed data shape: {data.shape}")
        
        dataset_start = time.time()
        
        # Method 1: Basic ROC Search (provides base results for other methods)
        base_results = self.run_basic_roc_search(data, target_col, dataset_name)
        
        if not base_results:
            print(f"No base results for {dataset_name}")
            return
        
        # Add basic search results
        for depth, result in base_results.items():
            self.all_results.append({
                'method': result['method'],
                'dataset': dataset_name,
                'depth': depth,
                'param_type': None,
                'param_value': None,
                'num_subgroups': result['num_subgroups'],
                'num_candidates': result['num_candidates'],
                'pruned_subgroups': result['pruned_subgroups'],
                'auc': result['auc'],
                'max_quality': result['max_quality'],
                'time_seconds': result['time_seconds']
            })
        
        # Method 2: Remove Hull Points
        method2_results = self.run_remove_hull_points(base_results, dataset_name)
        self.all_results.extend(method2_results)
        
        # Method 3: Closest Points to Hull
        method3_results = self.run_closest_points_to_hull(base_results, dataset_name)
        self.all_results.extend(method3_results)
        
        # Method 4: Furthest Points from Diagonal
        method4_results = self.run_furthest_points_from_diagonal(base_results, dataset_name)
        self.all_results.extend(method4_results)
        
        # Method 5: Below Hull Percentage
        method5_results = self.run_select_points_below_hull(base_results, dataset_name)
        self.all_results.extend(method5_results)
        
        # Method 6: Above Diagonal Percentage
        method6_results = self.run_select_points_above_diagonal(base_results, dataset_name)
        self.all_results.extend(method6_results)
        
        dataset_elapsed = time.time() - dataset_start
        print(f"\n{'='*80}")
        print(f"Completed {dataset_name} in {dataset_elapsed:.2f}s")
        print(f"{'='*80}")
    
    def run_all_datasets(self):
        """Run all methods on all datasets."""
        total_start = time.time()
        
        print(f"\n{'#'*80}")
        print(f"# COMPREHENSIVE ROC SEARCH TEST")
        print(f"# Testing all 6 methods on all datasets")
        print(f"# Max depth: {self.max_depth}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*80}")
        
        # Process each dataset
        for dataset_file, target_col in self.dataset_info.items():
            dataset_name = dataset_file.replace('.txt', '')
            
            # Skip datasets without target column
            if not target_col:
                print(f"Skipping {dataset_name} (no target column)")
                continue
            
            try:
                self.run_all_methods_on_dataset(dataset_name, target_col)
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        total_elapsed = time.time() - total_start
        
        print(f"\n{'#'*80}")
        print(f"# COMPLETED ALL DATASETS")
        print(f"# Total time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")
        print(f"# Total results: {len(self.all_results)}")
        print(f"{'#'*80}")
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
    
    def save_results(self):
        """Save all results to CSV files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)
        
        # Save complete results
        complete_path = self.output_dir / 'comprehensive_results.csv'
        df.to_csv(complete_path, index=False)
        print(f"\nSaved complete results to: {complete_path}")
        
        # Save summary by method
        summary_by_method = df.groupby('method').agg({
            'auc': ['mean', 'std', 'min', 'max'],
            'time_seconds': ['mean', 'sum']
        }).round(4)
        summary_method_path = self.output_dir / 'summary_by_method.csv'
        summary_by_method.to_csv(summary_method_path)
        print(f"Saved method summary to: {summary_method_path}")
        
        # Save summary by dataset
        summary_by_dataset = df.groupby('dataset').agg({
            'auc': ['mean', 'std'],
            'time_seconds': ['sum']
        }).round(4)
        summary_dataset_path = self.output_dir / 'summary_by_dataset.csv'
        summary_by_dataset.to_csv(summary_dataset_path)
        print(f"Saved dataset summary to: {summary_dataset_path}")
        
        # Save summary by depth
        summary_by_depth = df.groupby(['method', 'depth']).agg({
            'auc': ['mean', 'std'],
            'time_seconds': ['mean']
        }).round(4)
        summary_depth_path = self.output_dir / 'summary_by_depth.csv'
        summary_by_depth.to_csv(summary_depth_path)
        print(f"Saved depth summary to: {summary_depth_path}")
        
        # Save parameter analysis for methods 3-6
        param_methods = df[df['param_type'].notna()]
        if not param_methods.empty:
            param_summary = param_methods.groupby(['method', 'param_type', 'param_value']).agg({
                'auc': ['mean', 'std'],
                'new_auc': ['mean', 'std'],
                'auc_reduction': ['mean', 'std'],
                'selected_points': ['mean'],
                'time_seconds': ['mean']
            }).round(4)
            param_path = self.output_dir / 'parameter_analysis.csv'
            param_summary.to_csv(param_path)
            print(f"Saved parameter analysis to: {param_path}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\nCreating visualizations...")
        
        df = pd.DataFrame(self.all_results)
        
        # 1. AUC by method and depth
        self.plot_auc_by_method_depth(df)
        
        # 2. Parameter sensitivity analysis
        self.plot_parameter_sensitivity(df)
        
        # 3. Performance comparison
        self.plot_performance_comparison(df)
        
        # 4. Time analysis
        self.plot_time_analysis(df)
        
        print("Visualizations complete!")
    
    def plot_auc_by_method_depth(self, df):
        """Plot AUC by method and depth."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        methods = df['method'].unique()
        
        for idx, method in enumerate(methods):
            if idx >= 6:
                break
            
            ax = axes[idx]
            method_data = df[df['method'] == method]
            
            # Group by depth and calculate mean AUC
            if 'auc' in method_data.columns:
                depth_auc = method_data.groupby('depth')['auc'].mean()
            elif 'new_auc' in method_data.columns:
                depth_auc = method_data.groupby('depth')['new_auc'].mean()
            else:
                continue
            
            ax.plot(depth_auc.index, depth_auc.values, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Depth', fontsize=12)
            ax.set_ylabel('Mean AUC', fontsize=12)
            ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.0])
        
        plt.tight_layout()
        plot_path = self.output_dir / 'auc_by_method_depth.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved AUC plot to: {plot_path}")
    
    def plot_parameter_sensitivity(self, df):
        """Plot parameter sensitivity for methods 3-6."""
        param_methods = df[df['param_type'].notna()]
        
        if param_methods.empty:
            return
        
        unique_methods = param_methods['method'].unique()
        n_methods = len(unique_methods)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, method in enumerate(unique_methods):
            if idx >= 4:
                break
            
            ax = axes[idx]
            method_data = param_methods[param_methods['method'] == method]
            
            # Group by parameter value
            if 'new_auc' in method_data.columns:
                param_auc = method_data.groupby('param_value')['new_auc'].agg(['mean', 'std'])
            else:
                continue
            
            ax.errorbar(param_auc.index, param_auc['mean'], 
                       yerr=param_auc['std'], marker='o', 
                       linewidth=2, markersize=8, capsize=5)
            ax.set_xlabel(f"{method_data['param_type'].iloc[0]}", fontsize=12)
            ax.set_ylabel('Mean AUC', fontsize=12)
            ax.set_title(f'{method}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'parameter_sensitivity.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter sensitivity plot to: {plot_path}")
    
    def plot_performance_comparison(self, df):
        """Plot performance comparison across methods."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # AUC comparison
        methods = df['method'].unique()
        auc_means = []
        auc_stds = []
        
        for method in methods:
            method_data = df[df['method'] == method]
            if 'auc' in method_data.columns:
                auc_means.append(method_data['auc'].mean())
                auc_stds.append(method_data['auc'].std())
            elif 'new_auc' in method_data.columns:
                auc_means.append(method_data['new_auc'].mean())
                auc_stds.append(method_data['new_auc'].std())
            else:
                auc_means.append(0)
                auc_stds.append(0)
        
        x_pos = np.arange(len(methods))
        ax1.bar(x_pos, auc_means, yerr=auc_stds, capsize=5, alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
        ax1.set_ylabel('Mean AUC', fontsize=12)
        ax1.set_title('Mean AUC by Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Time comparison
        time_means = [df[df['method'] == m]['time_seconds'].mean() for m in methods]
        
        ax2.bar(x_pos, time_means, alpha=0.7, color='orange')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=45, ha='right')
        ax2.set_ylabel('Mean Time (seconds)', fontsize=12)
        ax2.set_title('Mean Execution Time by Method', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'performance_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved performance comparison to: {plot_path}")
    
    def plot_time_analysis(self, df):
        """Plot time analysis by depth and method."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = df['method'].unique()
        
        for method in methods:
            method_data = df[df['method'] == method]
            depth_time = method_data.groupby('depth')['time_seconds'].mean()
            
            ax.plot(depth_time.index, depth_time.values, 
                   'o-', linewidth=2, markersize=8, label=method, alpha=0.7)
        
        ax.set_xlabel('Depth', fontsize=12)
        ax.set_ylabel('Mean Time (seconds)', fontsize=12)
        ax.set_title('Execution Time by Depth and Method', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'time_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved time analysis to: {plot_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive test of all ROC search methods on all datasets'
    )
    parser.add_argument('--data-dir', default='./tests', 
                       help='Directory containing datasets')
    parser.add_argument('--output-dir', default='./runs/comprehensive_all_methods',
                       help='Output directory for results')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum search depth')
    
    args = parser.parse_args()
    
    # Create tester
    tester = ComprehensiveROCTester(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_depth=args.max_depth
    )
    
    # Run all tests
    tester.run_all_datasets()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST COMPLETE!")
    print(f"Results saved to: {tester.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
