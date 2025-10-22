"""
Run all datasets in parallel, each in a separate PowerShell terminal.
This dramatically reduces total execution time by processing datasets simultaneously.
"""
import subprocess
import sys
from pathlib import Path
import time

# All datasets to process
DATASETS = [
    'adult.txt',
    # 'Covertype.txt',
    # 'Credit-a.txt',
    # 'ionosphere.txt',
    # 'mushroom.txt',
    # 'tic-tac-toe.txt',
    # 'wisconsin.txt',
    # 'YPMSD.txt'
]

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
                            depth=4, min_coverage=50):
    """
    Launch a new PowerShell terminal to run analysis on a specific dataset.
    Uses batch mode with a single-dataset directory.
    """
    dataset_name = Path(dataset_file).stem
    dataset_dir = create_single_dataset_dir(dataset_file)
    output_path = Path(output_dir)
    
    # Build the Python command - use batch mode with single dataset directory
    # Wrap in try-catch to capture errors
    python_cmd = (
        f'try {{ '
        f'python true_roc_search.py '
        f'--batch '
        f'--data-dir "{dataset_dir}" '
        f'--depth {depth} '
        f'--min-coverage {min_coverage} '
        f'--output "{output_path}" '
        f'--start-from-pure-roc; '
        f'if ($LASTEXITCODE -ne 0) {{ Write-Host "ERROR: Python script failed with exit code $LASTEXITCODE" -ForegroundColor Red }} '
        f'else {{ Write-Host "SUCCESS: {dataset_name} completed!" -ForegroundColor Green }} '
        f'}} catch {{ Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red }}'
    )
    
    # PowerShell command that keeps window open and shows dataset name
    ps_cmd = [
        'powershell.exe',
        '-NoExit',
        '-Command',
        f'$host.UI.RawUI.WindowTitle = "Processing: {dataset_name}"; {python_cmd}; Write-Host "`nPress any key to close..."; $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")'
    ]
    
    print(f"🚀 Launching terminal for: {dataset_name}")
    
    # Start the process in a new window
    subprocess.Popen(ps_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    # Small delay to avoid overwhelming the system
    time.sleep(0.5)

def main():
    """Main function to launch all dataset processing in parallel."""
    print("=" * 60)
    print("PARALLEL DATASET PROCESSING")
    print("=" * 60)
    print(f"\nLaunching {len(DATASETS)} separate terminals...")
    print("Each terminal will process 6 methods on one dataset.")
    print("\nDatasets:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i}. {dataset}")
    
    print("\n" + "=" * 60)
    print("This will create temporary dataset directories.")
    input("Press ENTER to start parallel processing...")
    print()
    
    # Launch all datasets in parallel
    for dataset in DATASETS:
        run_dataset_in_terminal(dataset)
    
    print("\n" + "=" * 60)
    print("✅ All terminals launched!")
    print("=" * 60)
    print("\nMonitor the individual terminal windows for progress.")
    print("Each window will show its dataset name in the title bar.")
    print("\nTotal expected time: ~5-10 minutes (all running in parallel)")
    print("\nAfter completion, run 'python consolidate_results.py'")
    print("to merge all results into consolidated CSV files.")
    print("=" * 60)

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
