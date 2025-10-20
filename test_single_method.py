"""
Test script to run individual methods on a single dataset
and verify output format before full batch run.
"""

import sys
import argparse
from pathlib import Path
from true_roc_search import (
    load_data,
    preprocess_categorical_data,
    true_roc_search,
    hull_removal_search,
    closest_to_hull_search,
    furthest_from_diagonal_search,
    below_hull_threshold_search,
    above_diagonal_threshold_search,
    save_results
)

def test_single_method(method_name, data_file='./tests/adult.txt', target_col='target', 
                       depth=4, min_coverage=50, output_dir='./runs/test_single'):
    """
    Test a single method on a single dataset.
    
    Args:
        method_name: Name of method to test (1-6)
        data_file: Path to data file
        target_col: Target column name
        depth: Maximum depth
        min_coverage: Minimum coverage
        output_dir: Output directory
    """
    print("="*80)
    print(f"Testing Single Method: {method_name}")
    print("="*80)
    print(f"Dataset: {data_file}")
    print(f"Target: {target_col}")
    print(f"Depth: {depth}")
    print(f"Min Coverage: {min_coverage}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data = load_data(data_file)
    if data is None:
        print("ERROR: Failed to load data")
        return None
    
    # Preprocess
    print("Preprocessing data...")
    data = preprocess_categorical_data(data)
    print(f"Data shape: {data.shape}")
    
    # Run the selected method
    result = None
    dataset_name = Path(data_file).stem
    
    if method_name == '1' or method_name.lower() == 'pure_roc':
        print("\n" + "="*60)
        print("METHOD 1: Pure ROC Search")
        print("="*60)
        results = true_roc_search(data, target_col, None, depth, min_coverage)
        if results and 'pure_roc' in results:
            result = {'pure_roc': results['pure_roc']}
            result['pure_roc']['dataset'] = dataset_name
    
    elif method_name == '2' or method_name.lower() == 'hull_removal':
        print("\n" + "="*60)
        print("METHOD 2: Hull Removal Search")
        print("="*60)
        result_data = hull_removal_search(data, target_col, depth, min_coverage)
        if result_data:
            result_data['dataset'] = dataset_name
            result = {'hull_removal': result_data}
    
    elif method_name == '3' or method_name.lower() == 'closest_to_hull':
        print("\n" + "="*60)
        print("METHOD 3: Closest to Hull Search")
        print("="*60)
        result_data = closest_to_hull_search(data, target_col, n_points=10, max_depth=depth, min_coverage=min_coverage)
        if result_data:
            result_data['dataset'] = dataset_name
            result = {'closest_to_hull': result_data}
    
    elif method_name == '4' or method_name.lower() == 'furthest_diagonal':
        print("\n" + "="*60)
        print("METHOD 4: Furthest from Diagonal Search")
        print("="*60)
        result_data = furthest_from_diagonal_search(data, target_col, n_points=10, max_depth=depth, min_coverage=min_coverage)
        if result_data:
            result_data['dataset'] = dataset_name
            result = {'furthest_diagonal': result_data}
    
    elif method_name == '5' or method_name.lower() == 'below_hull':
        print("\n" + "="*60)
        print("METHOD 5: Below Hull Threshold Search")
        print("="*60)
        result_data = below_hull_threshold_search(data, target_col, distance_percentage=1.0, max_depth=depth, min_coverage=min_coverage)
        if result_data:
            result_data['dataset'] = dataset_name
            result = {'below_hull': result_data}
    
    elif method_name == '6' or method_name.lower() == 'above_diagonal':
        print("\n" + "="*60)
        print("METHOD 6: Above Diagonal Threshold Search")
        print("="*60)
        result_data = above_diagonal_threshold_search(data, target_col, distance_percentage=1.0, max_depth=depth, min_coverage=min_coverage)
        if result_data:
            result_data['dataset'] = dataset_name
            result = {'above_diagonal': result_data}
    
    else:
        print(f"ERROR: Unknown method '{method_name}'")
        print("Valid methods: 1-6, pure_roc, hull_removal, closest_to_hull, furthest_diagonal, below_hull, above_diagonal")
        return None
    
    if result:
        # Save results
        print("\n" + "="*60)
        print("Saving Results")
        print("="*60)
        output_path = Path(output_dir) / dataset_name
        save_results(result, str(output_path))
        
        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        for key, res in result.items():
            print(f"\nMethod: {res.get('algorithm', key)}")
            print(f"  Dataset: {res.get('dataset', 'N/A')}")
            print(f"  Final Width: {res.get('adaptive_width', 'N/A')}")
            print(f"  Total Candidates: {res.get('total_candidates', 'N/A')}")
            print(f"  AUC: {res.get('auc_approx', 0):.4f}")
            print(f"  Best Quality: {res.get('best_quality', 0):.4f}")
            print(f"  Best TPR: {res.get('best_tpr', 0):.4f}")
            print(f"  Best FPR: {res.get('best_fpr', 0):.4f}")
            print(f"  Search Time: {res.get('search_time', 0):.2f}s")
            
            # Show depth analysis
            if 'depth_analysis' in res:
                print(f"\n  Depth-by-Depth Analysis:")
                print(f"    {'Depth':<8} {'Start':<8} {'Candidates':<12} {'After Prune':<12} {'Width':<8} {'Best Quality':<12}")
                print(f"    {'-'*70}")
                for d in res['depth_analysis']:
                    print(f"    {d['depth']:<8} {d['subgroups_start']:<8} {d['candidates_generated']:<12} "
                          f"{d['subgroups_after_pruning']:<12} {d.get('width', 'N/A'):<8} {d['best_quality']:<12.4f}")
        
        print("\n" + "="*60)
        print(f"Results saved to: {output_path}")
        print("="*60)
        
        return result
    else:
        print("\nERROR: Method returned no results")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test individual search methods')
    parser.add_argument('method', help='Method to test: 1-6 or method name')
    parser.add_argument('--data', default='./tests/adult.txt', help='Path to data file')
    parser.add_argument('--target', default='target', help='Target column name')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth')
    parser.add_argument('--min-coverage', type=int, default=50, help='Minimum subgroup coverage')
    parser.add_argument('--output', default='./runs/test_single', help='Output directory')
    
    args = parser.parse_args()
    
    result = test_single_method(
        args.method,
        args.data,
        args.target,
        args.depth,
        args.min_coverage,
        args.output
    )
    
    if result:
        print("\n✓ Test completed successfully!")
        return 0
    else:
        print("\n✗ Test failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
