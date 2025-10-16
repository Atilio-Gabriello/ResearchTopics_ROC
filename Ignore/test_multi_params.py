"""
Quick test script to verify multiple parameter functionality
"""
import subprocess
import sys

print("="*70)
print("Testing Multiple Parameter Support")
print("="*70)

# Test 1: Single dataset with multiple n-points
print("\nTest 1: Multiple n-points values")
print("Command: --depth 1 --n-points 5 10 --datasets adult")
cmd1 = [
    sys.executable,
    "run_all_methods_all_datasets.py",
    "--depth", "1",
    "--n-points", "5", "10",
    "--datasets", "adult",
    "--output", "./runs/test_multi_params"
]
print(f"Running: {' '.join(cmd1)}")
print("-"*70)

# Test 2: Single dataset with multiple percentages
print("\nTest 2: Multiple distance percentages")
print("Command: --depth 1 --distance-percentage 0.5 1.0 2.0 --datasets adult")
cmd2 = [
    sys.executable,
    "run_all_methods_all_datasets.py",
    "--depth", "1",
    "--distance-percentage", "0.5", "1.0", "2.0",
    "--datasets", "adult",
    "--output", "./runs/test_multi_params"
]
print(f"Running: {' '.join(cmd2)}")
print("-"*70)

# Test 3: Both multiple parameters
print("\nTest 3: Multiple n-points AND percentages")
print("Command: --depth 1 --n-points 5 10 --distance-percentage 1.0 2.0 --datasets adult")
cmd3 = [
    sys.executable,
    "run_all_methods_all_datasets.py",
    "--depth", "1",
    "--n-points", "5", "10",
    "--distance-percentage", "1.0", "2.0",
    "--datasets", "adult",
    "--output", "./runs/test_multi_params"
]
print(f"Running: {' '.join(cmd3)}")
print("-"*70)

print("\n" + "="*70)
print("Choose a test to run (1, 2, 3, or 'all'):")
print("="*70)
choice = input("Enter choice: ").strip()

if choice == "1":
    subprocess.run(cmd1)
elif choice == "2":
    subprocess.run(cmd2)
elif choice == "3":
    subprocess.run(cmd3)
elif choice.lower() == "all":
    print("\nRunning Test 1...")
    subprocess.run(cmd1)
    print("\nRunning Test 2...")
    subprocess.run(cmd2)
    print("\nRunning Test 3...")
    subprocess.run(cmd3)
else:
    print("Invalid choice. Exiting.")
