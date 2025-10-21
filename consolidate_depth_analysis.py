"""
Consolidate depth_analysis.csv files from all dataset folders into one file.
"""
import pandas as pd
from pathlib import Path

def consolidate_depth_analysis(results_dir='./results', output_file='consolidated_depth_analysis.csv'):
    """
    Read all depth_analysis.csv files from dataset subdirectories and combine them
    into a single consolidated file with a 'dataset' column.
    
    Args:
        results_dir: Directory containing dataset subdirectories
        output_file: Name of the output consolidated file
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found")
        return
    
    all_data = []
    datasets_found = []
    datasets_missing = []
    
    # Get all subdirectories (dataset folders)
    dataset_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    print(f"Scanning {len(dataset_dirs)} dataset directories...\n")
    
    for dataset_dir in sorted(dataset_dirs):
        dataset_name = dataset_dir.name
        depth_file = dataset_dir / 'depth_analysis.csv'
        
        if depth_file.exists():
            try:
                # Read the CSV
                df = pd.read_csv(depth_file)
                
                # Add dataset column
                df['dataset'] = dataset_name
                
                all_data.append(df)
                datasets_found.append(dataset_name)
                print(f"✓ Loaded {dataset_name}: {len(df)} rows")
                
            except Exception as e:
                print(f"✗ Error reading {dataset_name}: {e}")
                datasets_missing.append(dataset_name)
        else:
            datasets_missing.append(dataset_name)
            print(f"✗ Missing depth_analysis.csv in {dataset_name}")
    
    if not all_data:
        print("\n❌ No depth_analysis.csv files found!")
        return
    
    # Combine all DataFrames
    print(f"\n{'='*60}")
    print("Consolidating data...")
    consolidated_df = pd.concat(all_data, ignore_index=True)
    
    # Save to file
    output_path = results_path / output_file
    consolidated_df.to_csv(output_path, index=False)
    
    print(f"✅ Saved consolidated file: {output_path}")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Total rows: {len(consolidated_df)}")
    print(f"  Datasets included: {len(datasets_found)}")
    print(f"  Datasets missing: {len(datasets_missing)}")
    
    if datasets_found:
        print(f"\n  Included datasets:")
        for ds in sorted(datasets_found):
            count = len(consolidated_df[consolidated_df['dataset'] == ds])
            print(f"    - {ds}: {count} rows")
    
    if datasets_missing:
        print(f"\n  ⚠️  Missing depth_analysis.csv:")
        for ds in sorted(datasets_missing):
            print(f"    - {ds}")
    
    # Show unique algorithms
    algorithms = consolidated_df['algorithm'].unique()
    print(f"\n  Algorithms found: {len(algorithms)}")
    for alg in sorted(algorithms):
        print(f"    - {alg}")
    
    print(f"\n{'='*60}")

if __name__ == '__main__':
    import sys
    
    # Allow custom results directory from command line
    results_dir = sys.argv[1] if len(sys.argv) > 1 else './results'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'consolidated_depth_analysis.csv'
    
    print("=" * 60)
    print("CONSOLIDATE DEPTH ANALYSIS")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Output file: {output_file}")
    print("=" * 60)
    print()
    
    consolidate_depth_analysis(results_dir, output_file)
