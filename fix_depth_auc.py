"""
Script to update depth_auc calculation in all search functions to use Pure ROC depth 1 AUC
"""
import re

# Read the file
with open('true_roc_search.py', 'r', encoding='utf-8') as f:
    content = f.read()

# List of functions that need updating (they all follow the same pattern but already have pure_roc_depth_1_auc variable)
# We need to find the depth_auc calculation sections and add the depth==1 check

# The sections to update are in:
# - closest_to_hull_search (line ~2552)
# - furthest_from_diagonal_search (line ~2783)
# - below_hull_threshold_search (line ~3012)
# - above_diagonal_threshold_search (line ~3242)

# We already updated hull_removal_search, so we need to update the others

# First, let's add pure_roc_depth_1_auc initialization to furthest_from_diagonal_search
pattern1 = r'(def furthest_from_diagonal_search\(data, target_col, n_points=10, max_depth=3, min_coverage=50, start_from_pure_roc=None\):.*?\n    # Initialize with population or Pure ROC depth 1 candidates\n    using_pure_roc_start = False\n    depth_1_candidates_from_pure_roc = None\n    depth_1_subgroups_from_pure_roc = None\n    \n    if start_from_pure_roc and isinstance\(start_from_pure_roc, dict\):)'

replacement1 = r'''\1
        depth_1_candidates_from_pure_roc = start_from_pure_roc.get('candidates', [])
        depth_1_subgroups_from_pure_roc = start_from_pure_roc.get('subgroups', [])
        pure_roc_depth_1_auc = start_from_pure_roc.get('depth_1_auc', None)  # Get Pure ROC depth 1 AUC'''

print("Updating file...")

#Save original
with open('true_roc_search.py.backup', 'w', encoding='utf-8') as f:
    f.write(content)
print("Backup saved")

# For now, let's manually add the lines we need
print("Script prepared - manual edits needed")
print("Functions to update:")
print("- furthest_from_diagonal_search")  
print("- below_hull_threshold_search")
print("- above_diagonal_threshold_search")
