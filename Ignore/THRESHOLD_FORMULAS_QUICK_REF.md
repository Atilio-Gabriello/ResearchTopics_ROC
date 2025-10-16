# Quick Reference: New Threshold Formulas

## Unified Reference Point

Both functions use the **same reference distance**:

```python
max_diagonal_distance = max(TPR - FPR)  # across all points above diagonal
```

This is the distance from the diagonal to the **furthest point**.

---

## Function 4: select_points_below_hull

### Formula
```python
threshold = max_diagonal_distance × (distance_percentage / 100)
```

### Selection
```python
select points where: vertical_distance_below_hull ≤ threshold
```

### Examples

| Percentage | Max Distance | Threshold | Interpretation |
|------------|--------------|-----------|----------------|
| 0.5% | 0.8 | 0.004 | Within 0.4% of max distance from hull |
| 1.0% | 0.8 | 0.008 | Within 0.8% of max distance from hull |
| 5.0% | 0.8 | 0.040 | Within 4% of max distance from hull |
| 10.0% | 0.8 | 0.080 | Within 8% of max distance from hull |
| 20.0% | 0.8 | 0.160 | Within 16% of max distance from hull |

### Behavior
- **Higher percentage** → **Larger threshold** → **More points** selected
- Expanding tolerance: allows points further below hull as percentage increases

---

## Function 5: select_points_above_diagonal

### Formula
```python
threshold = max_diagonal_distance × ((100 - distance_percentage) / 100)
```

### Selection
```python
select points where: diagonal_distance ≥ threshold
```

### Examples

| Percentage | Max Distance | Threshold | Interpretation |
|------------|--------------|-----------|----------------|
| 0.5% | 0.8 | 0.796 | At least 99.5% as far as the furthest |
| 1.0% | 0.8 | 0.792 | At least 99% as far as the furthest |
| 5.0% | 0.8 | 0.760 | At least 95% as far as the furthest |
| 10.0% | 0.8 | 0.720 | At least 90% as far as the furthest |
| 20.0% | 0.8 | 0.640 | At least 80% as far as the furthest |
| 50.0% | 0.8 | 0.400 | At least 50% as far as the furthest |

### Behavior
- **Higher percentage** → **Lower threshold** → **More points** selected
- Relaxing quality: allows points closer to diagonal as percentage increases

---

## Visual Comparison

### Function 4: Growing Threshold
```
Percentage:     1%    5%    10%   20%   50%
Threshold:      |     ||    |||   ||||  ||||||||||
Effect:         ├─────┼─────┼─────┼─────┼──────────→  More points
                Tight              Relaxed
```

### Function 5: Shrinking Threshold  
```
Percentage:     1%    5%    10%   20%   50%
Threshold:      ||||||||||  ||||  |||   ||    |
Effect:         ├──────────┼─────┼─────┼─────┼─────→  More points
                Very selective            Relaxed
```

---

## Practical Example

**Dataset**: ionosphere  
**Max diagonal distance**: 0.5375

### At 10% parameter:

**Function 4**:
- Threshold: 0.5375 × 0.10 = **0.0538**
- Selects points with hull_distance ≤ 0.0538
- Result: **9 points** selected → 3 hull points → 3.26% AUC reduction

**Function 5**:
- Threshold: 0.5375 × 0.90 = **0.4838**
- Selects points with diagonal_distance ≥ 0.4838  
- Result: **0 points** (all points have diagonal_distance < 0.4838)

---

## Mathematical Relationship

The two thresholds are **complementary**:

```
Function 4: t₄ = ref × p
Function 5: t₅ = ref × (1 - p)

Sum: t₄ + t₅ = ref × (p + (1-p)) = ref
```

Where:
- `ref` = max_diagonal_distance
- `p` = distance_percentage / 100

---

## When to Use Each Function

### Use Function 4 when:
- You want points **close to the convex hull**
- You're exploring the **suboptimal region** near the hull
- You want to see how much quality degrades with small deviations

### Use Function 5 when:
- You want **high-quality points** far from diagonal
- You're focusing on the **top performers**
- You want a **stricter quality threshold**

---

## Tuning Guide

### Function 4: Recommended Ranges
- **Tight selection**: 1-5% (very few points)
- **Moderate selection**: 5-15% (reasonable subset)
- **Relaxed selection**: 15-30% (many points)

### Function 5: Recommended Ranges  
- **Tight selection**: 1-10% (very few points, near max)
- **Moderate selection**: 10-30% (reasonable subset)
- **Relaxed selection**: 30-50% (many points)

---

## Code Usage

```python
from true_roc_search import select_points_below_hull, select_points_above_diagonal

# Function 4: Points close to hull
result4 = select_points_below_hull(
    points, 
    distance_percentage=10.0,  # 10% of max diagonal distance
    return_details=True,
    exclude_hull_points=True
)

print(f"Reference distance: {result4['reference_distance']:.4f}")
print(f"Threshold: {result4['threshold_distance']:.4f}")
print(f"Points selected: {result4['n_selected']}")

# Function 5: High-quality points
result5 = select_points_above_diagonal(
    points,
    distance_percentage=20.0,  # At least 80% as far as max
    return_details=True,
    exclude_hull_points=True
)

print(f"Reference distance: {result5['reference_distance']:.4f}")
print(f"Threshold: {result5['threshold_distance']:.4f}")  
print(f"Points selected: {result5['n_selected']}")
```
