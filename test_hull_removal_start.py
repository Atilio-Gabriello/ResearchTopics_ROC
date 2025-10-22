"""
Test script for the new Pure ROC start implementation with hull_removal_search
"""

import subprocess
import sys

print("=" * 80)
print("TESTING NEW PURE ROC START IMPLEMENTATION - Hull Removal Only")
print("=" * 80)
print()
print("This will run:")
print("1. Pure ROC search on Credit-a dataset (depth 4)")
print("2. Hull Removal search starting from Pure ROC depth 1 candidates")
print()
print("Expected Results:")
print("- Depth 1 candidates_generated: SAME for both methods")
print("- Depth 1 width: DIFFERENT (Pure ROC: ~13, Hull Removal: ~15-20)")
print("- Depth 2+ candidates: DIFFERENT (based on depth 1 width)")
print()
print("=" * 80)
print()

response = input("Run test? (y/n): ")
if response.lower() != 'y':
    print("Test cancelled.")
    sys.exit(0)

print()
print("Running test on Credit-a dataset...")
print()

# Run the test with Pure ROC and Hull Removal only
cmd = [
    "python",
    "true_roc_search.py",
    "--data", "tests/Credit-a.txt",
    "--target", "A16",
    "--depth", "4",
    "--start-from-pure-roc",
    "--output", "results/test_hull_removal_pure_roc_start"
]

try:
    result = subprocess.run(cmd, check=True, capture_output=False, text=True)
    print()
    print("=" * 80)
    print("TEST COMPLETED!")
    print("=" * 80)
    print()
    print("Check the results:")
    print("1. results/test_hull_removal_pure_roc_start/consolidated_depth_analysis.csv")
    print()
    print("What to look for:")
    print("- Find rows for Pure ROC and Hull Removal")
    print("- At depth=1: Check 'candidates_generated' column - should be IDENTICAL")
    print("- At depth=1: Check 'width' column - should be DIFFERENT")
    print("- At depth=2,3,4: 'candidates_generated' should differ")
    print()
    print("Opening the file for you...")
    print()
    
    # Try to open the file
    import os
    csv_path = "results/test_hull_removal_pure_roc_start/consolidated_depth_analysis.csv"
    if os.path.exists(csv_path):
        # Open in default CSV viewer
        if sys.platform == 'win32':
            os.startfile(csv_path)
        print(f"File opened: {csv_path}")
    else:
        print(f"File not found: {csv_path}")
        print("Check the console output above for any errors.")
    
except subprocess.CalledProcessError as e:
    print()
    print("=" * 80)
    print("TEST FAILED!")
    print("=" * 80)
    print(f"Error code: {e.returncode}")
    print()
    print("Check the error messages above for details.")
    sys.exit(1)
except Exception as e:
    print()
    print("=" * 80)
    print("ERROR!")
    print("=" * 80)
    print(f"Error: {e}")
    sys.exit(1)