import pysubdisc as subdisc
import pandas as pd

import pysubdisc
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./tests/adult.txt')

# Binarize the target variable for clarity (assuming 'gr50K' is the positive class)
data['target_binary'] = (data['target'] == 'gr50K').astype(int)

# 1. Perform Subgroup Discovery
sd = pysubdisc.singleNominalTarget(data, 'target', 'gr50K')
sd.run()
results = sd.asDataFrame()

# Data for ROC plot
roc_points = []

# Ground truth
y_true = data['target_binary']
P = y_true.sum()
N = len(y_true) - P

# 2. & 3. Calculate TPR and FPR for each subgroup
for i in range(len(results)):
    subgroup_members = sd.getSubgroupMembers(i)
    
    # Predictions for this subgroup
    y_pred = subgroup_members.astype(int)
    
    TP = (y_true & y_pred).sum()
    FP = (~y_true.astype(bool) & y_pred).sum()
    
    TPR = TP / P if P > 0 else 0
    FPR = FP / N if N > 0 else 0
    
    roc_points.append((FPR, TPR))

# 4. Plot the ROC points
plt.figure(figsize=(8, 6))
for i, (fpr, tpr) in enumerate(roc_points):
    plt.scatter(fpr, tpr, label=f'Subgroup {i}')
    plt.text(fpr, tpr, f' {i}', fontsize=9)

plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Space of Discovered Subgroups')
plt.legend()
plt.grid(True)
plt.show()

