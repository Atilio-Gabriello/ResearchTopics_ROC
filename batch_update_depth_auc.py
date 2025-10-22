"""
Batch update script to add pure_roc_depth_1_auc handling to remaining search functions
"""

# Read file
with open('true_roc_search.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Track changes
changes_made = []

# Find and update each function
i = 0
while i < len(lines):
    line = lines[i]
    
    # Look for functions that need updating
    if 'def furthest_from_diagonal_search' in line or \
       'def below_hull_threshold_search' in line or \
       'def above_diagonal_threshold_search' in line:
        
        func_name = line.split('(')[0].replace('def ', '').strip()
        print(f"Processing {func_name}...")
        
        # Look for the depth_auc calculation section within next 300 lines
        for j in range(i, min(i + 300, len(lines))):
            # Find the pattern: "# Calculate AUC at this depth"
            if '# Calculate AUC at this depth' in lines[j]:
                # Check if it already has the depth==1 check
                if 'if depth == 1 and pure_roc_depth_1_auc is not None:' in lines[j+1]:
                    print(f"  {func_name} already updated, skipping")
                    break
                
                # Insert the new logic
                # Original lines should be:
                # j: # Calculate AUC at this depth
                # j+1: if current_subgroups:
                # j+2:     depth_points = ...
                # j+3:     depth_auc = calculate_roc_metrics...
                # j+4: else:
                # j+5:     depth_auc = 0.0
                
                # Replace with:
                new_lines = [
                    lines[j],  # Keep comment
                    "        # At depth 1, use Pure ROC's depth 1 AUC if available (to ensure consistency)\n",
                    '        if depth == 1 and pure_roc_depth_1_auc is not None:\n',
                    '            depth_auc = pure_roc_depth_1_auc\n',
                    '            print(f"Using Pure ROC depth 1 AUC: {depth_auc:.4f}")\n',
                    '        elif current_subgroups:\n',
                    lines[j+2],  # depth_points line
                    lines[j+3],  # depth_auc calculation
                    lines[j+4],  # else:
                    lines[j+5],  # depth_auc = 0.0
                ]
                
                # Replace the 6 lines (j through j+5) with new_lines
                lines[j:j+6] = new_lines
                changes_made.append(f"{func_name} at line {j+1}")
                print(f"  Updated {func_name}")
                break
        
    i += 1

# Write back
with open('true_roc_search.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\nCompleted! Made {len(changes_made)} changes:")
for change in changes_made:
    print(f"  - {change}")
