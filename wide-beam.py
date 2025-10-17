"""
Wide Beam Search (pysubdisc) on All Datasets

Runs pysubdisc wide beam search with:
- Search Depth: 4
- Beam Width: 100
- All datasets in tests folder

Saves results to CSV for comparison with other methods.
"""

import pysubdisc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.spatial import ConvexHull


def get_dataset_info():
    """Define dataset information with target columns and positive class."""
    return {
        'adult.txt': {'target': 'target', 'positive_class': 'gr50K'},
        'mushroom.txt': {'target': 'poisonous', 'positive_class': 'p'},
        'ionosphere.txt': {'target': 'Attribute35', 'positive_class': 'g'},
        'wisconsin.txt': {'target': 'Class', 'positive_class': '4'},  # Fixed: '4' is malignant (as string)
        'tic-tac-toe.txt': {'target': 'class', 'positive_class': 'positive'},
        'Credit-a.txt': {'target': 'A16', 'positive_class': 'a'},  # Fixed: 'a' is approved
        'Covertype.txt': {'target': 'Cover_Type', 'positive_class': '1'},  # Fixed: '1' as string
    }



def calculate_roc_metrics(roc_points):
    """Calculate AUC and other ROC metrics from points."""
    if len(roc_points) == 0:
        return {
            'auc': 0.0,
            'num_points': 0,
            'best_tpr': 0.0,
            'best_fpr': 0.0,
            'max_quality': 0.0,
            'hull_points': 0
        }
    
    points_array = np.array(roc_points)
    
    # Calculate AUC using trapezoidal rule
    sorted_indices = np.argsort(points_array[:, 0])
    sorted_points = points_array[sorted_indices]
    
    # Add (0, 0) and (1, 1) for complete ROC curve
    roc_curve = np.vstack([[0, 0], sorted_points, [1, 1]])
    
    # Calculate AUC
    auc = 0.0
    for i in range(len(roc_curve) - 1):
        x1, y1 = roc_curve[i]
        x2, y2 = roc_curve[i + 1]
        auc += (x2 - x1) * (y1 + y2) / 2
    
    # Calculate quality (TPR - FPR)
    qualities = points_array[:, 1] - points_array[:, 0]
    best_idx = np.argmax(qualities)
    
    # Calculate convex hull
    hull_points = 0
    try:
        if len(points_array) >= 3:
            # Filter points above diagonal
            above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
            if len(above_diagonal) >= 3:
                extended_points = np.vstack([[0, 0], above_diagonal, [1, 1]])
                hull = ConvexHull(extended_points)
                hull_points = len(hull.vertices) - 2  # Exclude anchors
    except:
        pass
    
    return {
        'auc': auc,
        'num_points': len(roc_points),
        'best_tpr': points_array[best_idx, 1],
        'best_fpr': points_array[best_idx, 0],
        'max_quality': qualities[best_idx],
        'avg_quality': np.mean(qualities),
        'hull_points': hull_points
    }


def run_wide_beam_search(data_path, target_col, positive_class, depth=4, width=100):
    """Run wide beam search on a dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {data_path.stem}")
    print(f"Target: {target_col}, Positive Class: {positive_class}")
    print(f"Depth: {depth}, Width: {width}")
    print(f"{'='*60}")
    
    try:
        # Load data
        data = pd.read_csv(data_path)
        print(f"Loaded data: {data.shape}")
        
        # Convert target column to string if numeric (pysubdisc requires nominal target)
        target_dtype = data[target_col].dtype
        if pd.api.types.is_numeric_dtype(target_dtype):
            data[target_col] = data[target_col].astype(str)
        
        # Define ground-truth labels
        # Convert positive_class to string to match target column
        positive_class_str = str(positive_class)
        
        y_true = (data[target_col] == positive_class_str)
        P = y_true.sum()
        N = len(y_true) - P
        
        print(f"Positives: {P}, Negatives: {N}")
        
        if P == 0 or N == 0:
            print("Error: No positive or negative instances")
            return None
        
        # Run subgroup discovery with wide beam
        start_time = time.time()
        
        sd = pysubdisc.singleNominalTarget(data, target_col, positive_class_str)
        sd.searchDepth = depth
        sd.beamWidth = width
        sd.run()
        
        elapsed_time = time.time() - start_time
        
        results = sd.asDataFrame()
        print(f"Found {len(results)} subgroups in {elapsed_time:.2f}s")
        
        # Calculate TPR and FPR for each subgroup
        roc_points = []
        subgroup_details = []
        
        for i in range(len(results)):
            subgroup_members = sd.getSubgroupMembers(i)
            y_pred = subgroup_members.astype(bool)
            
            TP = (y_true & y_pred).sum()
            FP = ((~y_true) & y_pred).sum()
            FN = (y_true & ~y_pred).sum()
            TN = ((~y_true) & ~y_pred).sum()
            
            TPR = TP / P if P > 0 else 0
            FPR = FP / N if N > 0 else 0
            
            coverage = y_pred.sum()
            quality = TPR - FPR
            
            roc_points.append((FPR, TPR))
            
            subgroup_details.append({
                'subgroup_id': i,
                'coverage': coverage,
                'tp': TP,
                'fp': FP,
                'fn': FN,
                'tn': TN,
                'tpr': TPR,
                'fpr': FPR,
                'quality': quality,
                'description': str(results.iloc[i]['description']) if 'description' in results.columns else ''
            })
        
        # Calculate ROC metrics
        metrics = calculate_roc_metrics(roc_points)
        
        print(f"\nResults:")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  ROC Points: {metrics['num_points']}")
        print(f"  Hull Points: {metrics['hull_points']}")
        print(f"  Max Quality: {metrics['max_quality']:.4f}")
        print(f"  Best TPR: {metrics['best_tpr']:.4f}")
        print(f"  Best FPR: {metrics['best_fpr']:.4f}")
        
        return {
            'dataset': data_path.stem,
            'target_col': target_col,
            'positive_class': positive_class,
            'depth': depth,
            'width': width,
            'num_instances': len(data),
            'num_positives': P,
            'num_negatives': N,
            'num_subgroups': len(results),
            'time_seconds': elapsed_time,
            'auc': metrics['auc'],
            'num_roc_points': metrics['num_points'],
            'hull_points': metrics['hull_points'],
            'max_quality': metrics['max_quality'],
            'avg_quality': metrics.get('avg_quality', 0),
            'best_tpr': metrics['best_tpr'],
            'best_fpr': metrics['best_fpr'],
            'roc_points': roc_points,
            'subgroup_details': subgroup_details
        }
        
    except Exception as e:
        print(f"Error processing {data_path.stem}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_roc_comparison(all_results, output_dir):
    """Create ROC plots for all datasets."""
    n_datasets = len(all_results)
    
    if n_datasets == 0:
        return
    
    # Create individual plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(all_results):
        if idx >= 6:
            break
        
        ax = axes[idx]
        roc_points = result['roc_points']
        
        if len(roc_points) > 0:
            fpr_vals = [p[0] for p in roc_points]
            tpr_vals = [p[1] for p in roc_points]
            
            ax.scatter(fpr_vals, tpr_vals, alpha=0.6, s=50, c='blue', edgecolors='black')
            
            # Plot convex hull if possible
            if len(roc_points) >= 3:
                points_array = np.array(roc_points)
                above_diagonal = points_array[points_array[:, 1] > points_array[:, 0]]
                
                if len(above_diagonal) >= 3:
                    try:
                        extended = np.vstack([[0, 0], above_diagonal, [1, 1]])
                        hull = ConvexHull(extended)
                        
                        for simplex in hull.simplices:
                            ax.plot(extended[simplex, 0], extended[simplex, 1], 'r-', alpha=0.3)
                    except:
                        pass
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(f"{result['dataset']}\nAUC: {result['auc']:.4f}, Points: {result['num_roc_points']}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    # Hide unused subplots
    for idx in range(n_datasets, 6):
        axes[idx].axis('off')
    
    plt.suptitle(f'Wide Beam Search ROC Results (Depth={all_results[0]["depth"]}, Width={all_results[0]["width"]})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'wide_beam_roc_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved ROC plots to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("="*80)
    print("WIDE BEAM SEARCH (pysubdisc) - ALL DATASETS")
    print("="*80)
    print(f"Configuration: Depths=1-4, Width=100")
    
    data_dir = Path('./tests')
    output_dir = Path('./runs/wide_beam_search')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_info = get_dataset_info()
    all_results = []
    all_subgroup_details = []
    
    total_start = time.time()
    
    # Process each dataset at each depth
    for depth in range(1, 5):  # depths 1-4
        print(f"\n{'='*80}")
        print(f"PROCESSING DEPTH {depth}")
        print(f"{'='*80}")
        
        for filename, info in dataset_info.items():
            data_path = data_dir / filename
            
            if not data_path.exists():
                print(f"\nSkipping {filename} (file not found)")
                continue
            
            result = run_wide_beam_search(
                data_path,
                info['target'],
                info['positive_class'],
                depth=depth,
                width=100
            )
            
            if result:
                all_results.append(result)
                
                # Add dataset info to subgroup details
                for detail in result['subgroup_details']:
                    detail['dataset'] = result['dataset']
                    all_subgroup_details.append(detail)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"COMPLETED ALL DATASETS")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Datasets processed: {len(all_results)}")
    print(f"{'='*80}")
    
    # Save summary results
    if all_results:
        summary_data = []
        for result in all_results:
            summary_data.append({
                'dataset': result['dataset'],
                'depth': result['depth'],
                'width': result['width'],
                'num_instances': result['num_instances'],
                'num_positives': result['num_positives'],
                'num_negatives': result['num_negatives'],
                'num_subgroups': result['num_subgroups'],
                'num_roc_points': result['num_roc_points'],
                'hull_points': result['hull_points'],
                'auc': result['auc'],
                'max_quality': result['max_quality'],
                'avg_quality': result['avg_quality'],
                'best_tpr': result['best_tpr'],
                'best_fpr': result['best_fpr'],
                'time_seconds': result['time_seconds']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'wide_beam_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to: {summary_path}")
        
        # Display summary table
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Save detailed subgroup results
        if all_subgroup_details:
            details_df = pd.DataFrame(all_subgroup_details)
            details_path = output_dir / 'wide_beam_subgroups.csv'
            details_df.to_csv(details_path, index=False)
            print(f"\nSaved subgroup details to: {details_path}")
        
        # Create visualizations
        plot_roc_comparison(all_results, output_dir)
        
        # Print comparison statistics
        print(f"\n{'='*80}")
        print("COMPARISON STATISTICS")
        print(f"{'='*80}")
        print(f"Mean AUC: {summary_df['auc'].mean():.4f} ± {summary_df['auc'].std():.4f}")
        print(f"Best AUC: {summary_df['auc'].max():.4f} ({summary_df.loc[summary_df['auc'].idxmax(), 'dataset']})")
        print(f"Mean Time: {summary_df['time_seconds'].mean():.2f}s")
        print(f"Mean Subgroups: {summary_df['num_subgroups'].mean():.0f}")
        print(f"Mean ROC Points: {summary_df['num_roc_points'].mean():.0f}")
        
        print(f"\n{'='*80}")
        print("Results saved to: ./runs/wide_beam_search/")
        print("  - wide_beam_summary.csv (main results)")
        print("  - wide_beam_subgroups.csv (detailed subgroups)")
        print("  - wide_beam_roc_plots.png (visualizations)")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
