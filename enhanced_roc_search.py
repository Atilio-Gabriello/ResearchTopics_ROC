#!/usr/bin/env python3
"""
Enhanced ROC search implementation to replicate Table 2 results.
This version implements proper ROC quality measures with alpha parameter.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysubdisc
import json
from datetime import datetime

def roc_quality_measure(tp, fp, tn, fn, alpha=0.5):
    """
    Compute ROC-based quality measure.
    
    Args:
        tp, fp, tn, fn: Confusion matrix elements
        alpha: Trade-off parameter [0,1]
               alpha=0: minimize FPR (precision focus)
               alpha=1: maximize TPR (recall focus)
    
    Returns:
        Quality score combining TPR and FPR with alpha weighting
    """
    P = tp + fn  # Total positives
    N = fp + tn  # Total negatives
    
    if P == 0 or N == 0:
        return 0.0
    
    tpr = tp / P  # True Positive Rate (Sensitivity, Recall)
    fpr = fp / N  # False Positive Rate (1 - Specificity)
    
    # ROC quality: alpha * TPR + (1 - alpha) * (1 - FPR)
    # This rewards high TPR and low FPR according to alpha weighting
    quality = alpha * tpr + (1 - alpha) * (1 - fpr)
    
    return quality

def roc_distance_quality(tp, fp, tn, fn, alpha=0.5):
    """
    Alternative ROC quality: distance from ideal point (0,1) in ROC space.
    """
    P = tp + fn
    N = fp + tn
    
    if P == 0 or N == 0:
        return 0.0
    
    tpr = tp / P
    fpr = fp / N
    
    # Distance from ideal point (0,1): closer is better
    distance = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    
    # Convert to quality (higher is better)
    max_distance = np.sqrt(2)  # Maximum possible distance
    quality = 1 - (distance / max_distance)
    
    # Apply alpha weighting
    if alpha < 0.5:
        # Penalize high FPR more
        quality = quality * (1 - alpha * fpr)
    else:
        # Reward high TPR more
        quality = quality * (alpha * tpr + (1 - alpha))
    
    return quality

def manual_roc_search(data, target_col='target', positive_value='gr50K', 
                     alpha=0.5, depth=3, width=50, min_coverage=50,
                     quality_function='roc_combined'):
    """
    Manual implementation of ROC search since the Java backend alpha isn't working.
    """
    
    print(f"\\n=== Manual ROC Search (alpha={alpha}, depth={depth}, width={width}) ===")
    
    # Prepare data
    y_true = (data[target_col] == positive_value).values
    P = int(y_true.sum())
    N = int(len(y_true) - P)
    print(f"Dataset: {len(data)} samples, {P} positives, {N} negatives")
    
    # Get all possible single conditions using regular beam search first
    sd = pysubdisc.singleNominalTarget(data, target_col, positive_value)
    sd.searchStrategy = 'BEAM'
    sd.searchDepth = depth
    sd.searchStrategyWidth = width * 3  # Larger width to get more candidates
    sd.minimumCoverage = max(1, min_coverage // 2)  # More lenient coverage
    sd.maximumSubgroups = width * 10  # More subgroups to choose from
    sd.nrThreads = 1
    sd.postProcessingDoAutoRun = False
    sd.run()
    
    # Get all candidates
    candidates_df = sd.asDataFrame()
    if candidates_df.empty:
        print("No candidates found!")
        return pd.DataFrame(), []
    
    print(f"Found {len(candidates_df)} initial candidates")
    
    # Compute ROC quality for each candidate
    roc_candidates = []
    
    for i in range(len(candidates_df)):
        try:
            # Get subgroup membership
            members = sd.getSubgroupMembers(i).astype(bool)
            
            # Compute confusion matrix
            tp = int((y_true & members).sum())
            fp = int((~y_true & members).sum())
            fn = int((y_true & ~members).sum())
            tn = int((~y_true & ~members).sum())
            
            coverage = tp + fp
            if coverage < min_coverage:
                continue
            
            # Compute ROC quality based on alpha
            if quality_function == 'roc_combined':
                quality = roc_quality_measure(tp, fp, tn, fn, alpha)
            elif quality_function == 'roc_distance':
                quality = roc_distance_quality(tp, fp, tn, fn, alpha)
            else:
                # Fallback to original quality
                quality = float(candidates_df.iloc[i]['Quality'])
            
            tpr = tp / P if P > 0 else 0
            fpr = fp / N if N > 0 else 0
            
            roc_candidates.append({
                'index': i,
                'conditions': candidates_df.iloc[i]['Conditions'],
                'coverage': coverage,
                'original_quality': float(candidates_df.iloc[i]['Quality']),
                'roc_quality': quality,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'tpr': tpr, 'fpr': fpr,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tpr
            })
            
        except Exception as e:
            print(f"Error processing candidate {i}: {e}")
            continue
    
    if not roc_candidates:
        print("No valid ROC candidates found!")
        return pd.DataFrame(), []
    
    # Sort by ROC quality and select top width candidates
    roc_candidates.sort(key=lambda x: x['roc_quality'], reverse=True)
    selected = roc_candidates[:width]
    
    print(f"Selected top {len(selected)} subgroups based on ROC quality")
    print(f"Quality range: {selected[0]['roc_quality']:.4f} to {selected[-1]['roc_quality']:.4f}")
    
    # Convert to DataFrame
    result_df = pd.DataFrame([{
        'Conditions': c['conditions'],
        'Coverage': c['coverage'],
        'Quality': c['roc_quality'],
        'Original_Quality': c['original_quality'],
        'TPR': c['tpr'],
        'FPR': c['fpr'],
        'Precision': c['precision'],
        'Recall': c['recall'],
        'TP': c['tp'], 'FP': c['fp'],
        'TN': c['tn'], 'FN': c['fn']
    } for c in selected])
    
    # ROC points for plotting
    roc_points = [{
        'FPR': c['fpr'],
        'TPR': c['tpr'],
        'Coverage': c['coverage'],
        'Quality': c['roc_quality'],
        'Conditions': c['conditions']
    } for c in selected]
    
    return result_df, roc_points

def run_enhanced_roc_experiment(data_path, alphas=[0.0, 0.3, 0.5, 0.7, 1.0], 
                               depth=4, width=50, min_coverage=100):
    """
    Run enhanced ROC experiments with working alpha parameter.
    """
    
    data = pd.read_csv(data_path)
    results = {}
    
    out_dir = Path('./runs/enhanced_roc')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\\n{'='*60}")
    print(f"ENHANCED ROC SEARCH EXPERIMENT")
    print(f"{'='*60}")
    print(f"Data: {data_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"Target distribution: {data['target'].value_counts().to_dict()}")
    print(f"Alphas: {alphas}")
    print(f"Search params: depth={depth}, width={width}, min_coverage={min_coverage}")
    
    summary_rows = []
    
    for alpha in alphas:
        print(f"\\n--- Running alpha = {alpha} ---")
        
        # Run manual ROC search
        subgroups_df, roc_points = manual_roc_search(
            data, alpha=alpha, depth=depth, width=width, 
            min_coverage=min_coverage, quality_function='roc_combined'
        )
        
        if subgroups_df.empty:
            print(f"No results for alpha={alpha}")
            continue
        
        # Save results
        alpha_dir = out_dir / f'alpha_{alpha}'
        alpha_dir.mkdir(exist_ok=True)
        
        subgroups_df.to_csv(alpha_dir / 'subgroups.csv', index=False)
        
        roc_df = pd.DataFrame(roc_points)
        roc_df.to_csv(alpha_dir / 'roc_points.csv', index=False)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.scatter(roc_df['FPR'], roc_df['TPR'], alpha=0.6, s=60)
        
        # Envelope curve
        roc_sorted = roc_df.sort_values('FPR')
        envelope_tpr = []
        max_tpr_so_far = 0
        envelope_fpr = []
        
        for _, row in roc_sorted.iterrows():
            if row['TPR'] > max_tpr_so_far:
                envelope_tpr.append(row['TPR'])
                envelope_fpr.append(row['FPR'])
                max_tpr_so_far = row['TPR']
        
        if envelope_fpr:
            envelope_fpr = [0] + envelope_fpr + [1]
            envelope_tpr = [0] + envelope_tpr + [1]
            plt.plot(envelope_fpr, envelope_tpr, 'r-', linewidth=2, label='ROC Envelope')
            
            # Compute AUC
            auc = np.trapezoid(envelope_tpr, envelope_fpr)
        else:
            auc = 0.5
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (α={alpha}, AUC={auc:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(alpha_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Summary statistics
        summary = {
            'alpha': alpha,
            'depth': depth,
            'width': width,
            'num_subgroups': len(subgroups_df),
            'best_quality': float(subgroups_df['Quality'].max()),
            'mean_quality': float(subgroups_df['Quality'].mean()),
            'median_quality': float(subgroups_df['Quality'].median()),
            'mean_coverage': float(subgroups_df['Coverage'].mean()),
            'median_coverage': float(subgroups_df['Coverage'].median()),
            'mean_tpr': float(subgroups_df['TPR'].mean()),
            'mean_fpr': float(subgroups_df['FPR'].mean()),
            'mean_precision': float(subgroups_df['Precision'].mean()),
            'AUC': float(auc),
            'max_TPR': float(subgroups_df['TPR'].max()),
            'min_FPR': float(subgroups_df['FPR'].min()),
        }
        
        # TPR at FPR thresholds
        for fpr_thresh in [0.05, 0.10, 0.20]:
            valid_points = roc_df[roc_df['FPR'] <= fpr_thresh]
            tpr_at_fpr = float(valid_points['TPR'].max()) if not valid_points.empty else 0.0
            summary[f'TPR@FPR<={fpr_thresh:.2f}'] = tpr_at_fpr
        
        summary_rows.append(summary)
        results[alpha] = {
            'subgroups': subgroups_df,
            'roc_points': roc_points,
            'summary': summary
        }
        
        print(f"  Results: {summary['num_subgroups']} subgroups, "
              f"AUC={summary['AUC']:.3f}, "
              f"Quality={summary['best_quality']:.3f}")
        
        # Save config
        config = {
            'timestamp': datetime.now().isoformat(),
            'alpha': alpha,
            'depth': depth,
            'width': width,
            'min_coverage': min_coverage,
            'quality_function': 'roc_combined',
            'method': 'manual_roc_search'
        }
        with open(alpha_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    # Save summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(out_dir / 'alpha_comparison.csv', index=False)
        
        print(f"\\n{'='*80}")
        print("ALPHA COMPARISON RESULTS")
        print(f"{'='*80}")
        print(f"Search parameters: depth={depth}, width={width}")
        print(summary_df[['alpha', 'num_subgroups', 'AUC', 'best_quality', 'mean_tpr', 'mean_fpr']].to_string(index=False))
        
        # Create overlay plot
        plt.figure(figsize=(10, 8))
        for alpha in alphas:
            if alpha in results:
                roc_df = pd.DataFrame(results[alpha]['roc_points'])
                auc = results[alpha]['summary']['AUC']
                
                # Plot envelope
                roc_sorted = roc_df.sort_values('FPR')
                envelope_tpr = []
                max_tpr_so_far = 0
                envelope_fpr = []
                
                for _, row in roc_sorted.iterrows():
                    if row['TPR'] > max_tpr_so_far:
                        envelope_tpr.append(row['TPR'])
                        envelope_fpr.append(row['FPR'])
                        max_tpr_so_far = row['TPR']
                
                if envelope_fpr:
                    envelope_fpr = [0] + envelope_fpr + [1]
                    envelope_tpr = [0] + envelope_tpr + [1]
                    plt.plot(envelope_fpr, envelope_tpr, linewidth=2, 
                            label=f'α={alpha} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves: Alpha Parameter Effect')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'alpha_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\\nResults saved to: {out_dir}")
        print(f"Summary table: {out_dir / 'alpha_comparison.csv'}")
        print(f"Overlay plot: {out_dir / 'alpha_overlay.png'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Enhanced ROC Search Implementation')
    parser.add_argument('--data', default='./tests/adult.txt', help='Dataset path')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.3, 0.5, 0.7, 1.0], 
                       help='Alpha values to test')
    parser.add_argument('--depth', type=int, default=4, help='Search depth')
    parser.add_argument('--width', type=int, default=50, help='Beam width')
    parser.add_argument('--min-coverage', type=int, default=100, help='Minimum coverage')
    
    args = parser.parse_args()
    
    results = run_enhanced_roc_experiment(
        data_path=args.data,
        alphas=args.alphas,
        depth=args.depth,
        width=args.width,
        min_coverage=args.min_coverage
    )
    
    return results

if __name__ == '__main__':
    main()