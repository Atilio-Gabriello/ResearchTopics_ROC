"""
Run all datasets in parallel, each in a separate PowerShell terminal.
This dramatically reduces total execution time by processing datasets simultaneously.

ENHANCED VERSION: Supports multiple parameter configurations per dataset.
"""
import subprocess
import sys
from pathlib import Path
import time

# Dataset configurations with multiple parameter combinations
# Each dataset can have multiple configurations that will run sequentially
DATASET_CONFIGS = {
    'adult.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
        # Add more configurations to run different parameter combinations
        # {'n_points': 15, 'distance_below': 1.5, 'distance_above': 8.0},
    ],
    # 'Covertype.txt': [
    #     {'n_points': 20, 'distance_below': 2.0, 'distance_above': 5.0},
    #     # {'n_points': 25, 'distance_below': 2.5, 'distance_above': 4.0},
    # ],
    'Credit-a.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
    ],
    'ionosphere.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
    ],
    'mushroom.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
    ],
    'tic-tac-toe.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
    ],
    'wisconsin.txt': [
        {'n_points': 5, 'distance_below': 1.0, 'distance_above': 1.0},
        {'n_points': 10, 'distance_below': 2.0, 'distance_above': 5.0},
        {'n_points': 25, 'distance_below': 5.0, 'distance_above': 10.0},
    ],
    # 'YPMSD.txt': [
    #     {'n_points': 25, 'distance_below': 2.5, 'distance_above': 5.0},
    # ]
}

# Default configuration for datasets not in DATASET_CONFIGS
DEFAULT_CONFIG = {'n_points': 10, 'distance_below': 1.0, 'distance_above': 10.0}

# Get all unique datasets
DATASETS = list(DATASET_CONFIGS.keys())

def create_single_dataset_dir(dataset_file):
    """
    Create a temporary directory with just one dataset for batch processing.
    """
    temp_dir = Path('./temp_dataset_dirs')
    temp_dir.mkdir(exist_ok=True)
    
    dataset_name = Path(dataset_file).stem
    dataset_dir = temp_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    # Copy the dataset file to the temp directory
    source = Path('./tests') / dataset_file
    dest = dataset_dir / dataset_file
    
    if not dest.exists() and source.exists():
        import shutil
        shutil.copy2(source, dest)
    
    return dataset_dir

def run_dataset_in_terminal(dataset_file, output_dir='./results', 
                            depth=4, min_coverage=50,
                            n_points=None, distance_below=None, distance_above=None,
                            config_index=0):
    """
    Launch a new PowerShell terminal to run analysis on a specific dataset.
    Uses batch mode with a single-dataset directory.
    
    Args:
        dataset_file: Dataset filename
        output_dir: Output directory
        depth: Maximum search depth
        min_coverage: Minimum subgroup coverage
        n_points: n_points parameter for methods 3&4
        distance_below: distance_percentage for method 5
        distance_above: distance_percentage for method 6
        config_index: Index for multiple configurations (for output naming)
    """
    dataset_name = Path(dataset_file).stem
    if(dataset_name == 'YPMSD' or dataset_name == 'Covertype'):
        print(f"  Using min_coverage=50000 for {dataset_name}")
        min_coverage = 50000
    dataset_dir = create_single_dataset_dir(dataset_file)
    
    # Create unique output path if multiple configs
    if config_index > 0:
        output_path = Path(output_dir) / f"config_{config_index}"
    else:
        output_path = Path(output_dir)
    
    # Build parameter string for display
    params_str = f"n={n_points}, below={distance_below}%, above={distance_above}%"
    
    # Build the Python command - use batch mode with single dataset directory
    python_cmd = (
        f'try {{ '
        f'python true_roc_search.py '
        f'--batch '
        f'--data-dir "{dataset_dir}" '
        f'--depth {depth} '
        f'--min-coverage {min_coverage} '
        f'--output "{output_path}" '
        f'--start-from-pure-roc'
    )
    
    # Add custom parameters if provided
    if n_points is not None:
        python_cmd += f' --n-points {n_points}'
    if distance_below is not None:
        python_cmd += f' --distance-below {distance_below}'
    if distance_above is not None:
        python_cmd += f' --distance-above {distance_above}'
    
    # Complete the command with error handling
    python_cmd += (
        f'; '
        f'if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: Python script failed with exit code $LASTEXITCODE" -ForegroundColor Red }} '
        f'else {{ Write-Host "SUCCESS: {dataset_name} completed!" -ForegroundColor Green }} '
        f'}} catch {{ Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red }}'
    )
    
    # PowerShell command that keeps window open and shows dataset name
    window_title = f"{dataset_name}" + (f" [Config {config_index+1}]" if config_index > 0 else "")
    ps_cmd = [
        'powershell.exe',
        '-NoExit',
        '-Command',
        f'$host.UI.RawUI.WindowTitle = "Processing: {window_title}"; {python_cmd}; Write-Host "`nPress any key to close..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
    ]
    
    print(f"🚀 Launching: {dataset_name} ({params_str})")
    
    # Start the process in a new window
    subprocess.Popen(ps_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    # Small delay to avoid overwhelming the system
    time.sleep(0.5)

def main():
    """Main function to launch all dataset processing in parallel."""
    print("=" * 70)
    print("PARALLEL DATASET PROCESSING WITH CUSTOM PARAMETERS")
    print("=" * 70)
    
    # Count total runs
    total_runs = sum(len(configs) for configs in DATASET_CONFIGS.values())
    
    print(f"\nLaunching {len(DATASETS)} datasets with {total_runs} total configurations...")
    print("Each terminal will process 6 methods on one dataset with specific parameters.")
    
    print("\n" + "Dataset Configurations:".center(70, "-"))
    for dataset, configs in DATASET_CONFIGS.items():
        print(f"\n📊 {dataset}:")
        for i, config in enumerate(configs, 1):
            n = config.get('n_points', 'default')
            below = config.get('distance_below', 'default')
            above = config.get('distance_above', 'default')
            print(f"   Config {i}: n_points={n}, distance_below={below}%, distance_above={above}%")
    
    print("\n" + "=" * 70)
    print("This will create temporary dataset directories.")
    print("Multiple configurations will run SEQUENTIALLY (one after another).")
    input("Press ENTER to start parallel processing...")
    print()
    
    # Launch all datasets with their configurations
    for dataset in DATASETS:
        configs = DATASET_CONFIGS.get(dataset, [DEFAULT_CONFIG])
        
        for config_idx, config in enumerate(configs):
            run_dataset_in_terminal(
                dataset,
                n_points=config.get('n_points'),
                distance_below=config.get('distance_below'),
                distance_above=config.get('distance_above'),
                config_index=config_idx
            )
            
            # Add extra delay between multiple configs of same dataset
            if len(configs) > 1 and config_idx < len(configs) - 1:
                time.sleep(1.0)
    
    print("\n" + "=" * 70)
    print("✅ All terminals launched!")
    print("=" * 70)
    print(f"\n📊 Total: {total_runs} runs across {len(DATASETS)} datasets")
    print("\nMonitor the individual terminal windows for progress.")
    print("Each window shows its dataset name and config number in the title bar.")
    print("\n⏱️  Total expected time: ~5-10 minutes (all running in parallel)")
    print("\n💾 Results will be saved in ./results/")
    print("   - Multiple configs save to ./results/config_N/ subdirectories")
    print("\n📋 After completion, run 'python consolidate_results.py'")
    print("   to merge all results into consolidated CSV files.")
    print("=" * 70)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
