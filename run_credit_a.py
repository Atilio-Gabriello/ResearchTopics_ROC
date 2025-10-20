"""
Run Credit-a dataset only - all 6 methods.
"""
import subprocess
import sys
from pathlib import Path
import shutil

def run_credit_a():
    """Run analysis on Credit-a dataset."""
    dataset_file = 'Credit-a.txt'
    dataset_name = 'Credit-a'
    
    # Create temporary directory for this dataset
    temp_dir = Path('./temp_dataset_dirs')
    temp_dir.mkdir(exist_ok=True)
    
    dataset_dir = temp_dir / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    
    # Copy the dataset file
    source = Path('./tests') / dataset_file
    dest = dataset_dir / dataset_file
    
    if not dest.exists() and source.exists():
        shutil.copy2(source, dest)
        print(f"✓ Copied {dataset_file} to {dataset_dir}")
    
    # Build the Python command
    python_cmd = (
        f'python true_roc_search.py '
        f'--batch '
        f'--data-dir "{dataset_dir}" '
        f'--depth 4 '
        f'--min-coverage 50 '
        f'--output "./runs/all_methods_depth4" '
        f'--pure-roc'
    )
    
    # PowerShell command that keeps window open
    ps_cmd = [
        'powershell.exe',
        '-NoExit',
        '-Command',
        f'$host.UI.RawUI.WindowTitle = "Processing: {dataset_name}"; {python_cmd}'
    ]
    
    print(f"🚀 Launching terminal for: {dataset_name}")
    print(f"   Processing all 6 methods with depth=4, min_coverage=50")
    print()
    
    # Start the process in a new window
    subprocess.Popen(ps_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
    
    print("=" * 60)
    print("✅ Terminal launched!")
    print("=" * 60)
    print("\nMonitor the terminal window for progress.")
    print("Results will be saved to: ./runs/all_methods_depth4/Credit-a/")
    print()

if __name__ == '__main__':
    try:
        run_credit_a()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
