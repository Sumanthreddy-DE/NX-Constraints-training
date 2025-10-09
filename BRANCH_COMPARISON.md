# Model Branch Comparison

This document describes the three model branches created to address class imbalance and improve model performance.

## Branch Overview

### 1. **baseline-model** (Original Branch)
**Branch name:** `baseline-model`

**Description:** The original neural network model without any special handling for class imbalance or feature selection.

**Characteristics:**
- Uses all 41 features from the dataset
- No class weights applied
- Standard training approach with early stopping and learning rate reduction
- Architecture: 256 → 128 → 64 → output layers with BatchNorm and Dropout

**Use case:** Baseline for comparison with the improved models.

---

### 2. **model-with-class-weights** (Handles Class Imbalance)
**Branch name:** `model-with-class-weights`

**Description:** Enhanced version that addresses the severe class imbalance problem using class weights.

**Key improvements:**
- **Class weights calculation:** Automatically computes balanced class weights using scikit-learn's `compute_class_weight`
- **Weighted training:** Applies class weights during model training to give more importance to minority classes
- **Same architecture:** Uses all 41 features with the same neural network architecture

**Why this helps:**
The dataset has a severe class imbalance (79:1 ratio between most and least common classes):
- Revolute: 79 samples (most common)
- Cylinder: 37 samples
- Planar: 20 samples
- Fixed: 9 samples
- Parallel: 3 samples
- Inplane: 3 samples
- Slider: 1 sample (least common)
- Perpendicular: 1 sample
- Inline: 1 sample

Class weights ensure the model doesn't just learn to predict the majority class (Revolute) and ignores minority classes.

**Expected benefit:** Better performance on minority classes, more balanced predictions across all joint types.

---

### 3. **model-with-feature-selection** (Optimized Feature Set)
**Branch name:** `model-with-feature-selection`

**Description:** Uses Random Forest to identify and select only the most important features for training.

**Key improvements:**
- **Feature selection:** Uses Random Forest classifier to rank features by importance
- **Top 20 features:** Selects only the top 20 most important features from the original 41
- **Reduced dimensionality:** Trains a simpler model with fewer inputs

**How it works:**
1. Trains a Random Forest on all features to compute feature importance scores
2. Ranks features by importance
3. Selects top 20 features
4. Trains the neural network only on these selected features
5. Saves selected feature information for inference

**Expected benefits:**
- **Reduced overfitting:** Fewer features mean less chance of learning noise
- **Faster training:** Smaller input dimension speeds up training
- **Better generalization:** Focuses on the most discriminative features
- **Improved performance:** Removes noisy/irrelevant features that may confuse the model

---

## Testing the Branches

### Switch to a branch
```bash
# Test baseline model
git checkout baseline-model
python train_joint_predictor.py

# Test class weights model
git checkout model-with-class-weights
python train_joint_predictor.py

# Test feature selection model
git checkout model-with-feature-selection
python train_joint_predictor.py
```

### Compare Results
After training each model, compare:
1. **Overall accuracy** on test set
2. **Per-class performance** in classification report
3. **Confusion matrix** - check if minority classes are being predicted
4. **Training time** and convergence behavior

---

## Recommendations

### For Severely Imbalanced Data
**Use:** `model-with-class-weights`
- Best when you need to predict all classes including rare ones
- Ensures minority classes get adequate attention during training

### For High-Dimensional Noisy Data
**Use:** `model-with-feature-selection`
- Best when you suspect many features are irrelevant
- Improves model robustness and reduces overfitting
- Faster training and inference

### For Best Results
**Consider combining both approaches:**
You could create a fourth branch that combines class weights AND feature selection for potentially the best performance.

---

## Class Distribution Summary

| Joint Type | Count | Percentage |
|------------|-------|------------|
| Revolute | 79 | 51.3% |
| Cylinder | 37 | 24.0% |
| Planar | 20 | 13.0% |
| Fixed | 9 | 5.8% |
| Parallel | 3 | 1.9% |
| Inplane | 3 | 1.9% |
| Slider | 1 | 0.6% |
| Perpendicular | 1 | 0.6% |
| Inline | 1 | 0.6% |
| **Total** | **154** | **100%** |

**Imbalance Ratio:** 79:1 (most common to least common)

---

## Next Steps

1. **Train all three models** and compare their performance metrics
2. **Analyze confusion matrices** to see which model handles minority classes better
3. **Consider hyperparameter tuning** on the best-performing approach
4. **Optional:** Create a combined branch with both class weights and feature selection
