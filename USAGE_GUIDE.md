# Joint Type Predictor - Quick Start Guide

## Overview

This ML system predicts joint types in NX assembly based on constraint data. It uses **TensorFlow** neural networks with **iterative retraining** until convergence.

## 📁 Project Structure

```
/workspace/
├── Constraint_dataset.xlsx          # Original dataset
├── train_joint_predictor.py         # Main training script
├── predict_joint.py                 # Inference/prediction script
├── requirements.txt                 # Python dependencies
├── ML_README.md                    # Detailed documentation
├── USAGE_GUIDE.md                  # This quick start guide
│
├── joint_predictor_model.keras     # Trained model (generated)
├── preprocessing_objects.pkl       # Scaler & encoder (generated)
├── confusion_matrix.png           # Evaluation visualization (generated)
├── training_history.png           # Training progress (generated)
└── predictions.csv                # Prediction results (generated)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_joint_predictor.py
```

**What it does:**
- Loads dataset (154 samples, 41 features)
- Splits: 120 training, 34 testing
- Trains neural network iteratively
- Stops when loss change < tolerance (default: 0.01)
- Saves model and generates visualizations

**Output:**
```
✓ joint_predictor_model.keras - Trained model
✓ preprocessing_objects.pkl - Feature transformers
✓ confusion_matrix.png - Confusion matrix
✓ training_history.png - Loss/accuracy plots
```

### 3. Make Predictions

```bash
python predict_joint.py
```

**What it does:**
- Loads trained model
- Predicts joint types for dataset
- Shows accuracy and misclassifications
- Saves results to `predictions.csv`

## 📊 Dataset Information

### Input Features (41 columns)

1. **Classifier_1, Classifier_2**: Part identifiers
2. **Constraint Features** (39 columns):
   - Touch constraints (Plane-Plane, Cylinder-Cylinder, etc.)
   - Align constraints
   - Parallel/Perpendicular constraints
   - Distance constraints
   - And more...
3. **Count**: Number of constraints

### Target Variable

**Joint** - Joint type to predict:
- Revolute (79 samples)
- Cylinder (37 samples)
- Planar (20 samples)
- Fixed (9 samples)
- Parallel, Inplane, Slider, etc. (7 samples)

## 🔧 Customization

### Adjust Training Parameters

Edit `train_joint_predictor.py`:

```python
# Line 378-381
TRAIN_SIZE = 120               # Training samples
CONVERGENCE_TOLERANCE = 0.01   # Loss change threshold
MAX_ITERATIONS = 20            # Max retraining iterations
```

### Modify Model Architecture

Edit `build_model()` method in `train_joint_predictor.py`:

```python
# Lines 141-162
def build_model(self, input_dim):
    model = keras.Sequential([
        layers.Dense(256, activation='relu'),  # Adjust layer sizes
        # ... add/remove layers
    ])
```

## 📈 Understanding Results

### Training Output

```
ITERATION 1/20
Results:
  Train Loss: 0.464019 | Train Accuracy: 0.9250
  Test Loss:  0.724826 | Test Accuracy:  0.8824
  
Loss change from previous iteration: inf
Convergence tolerance: 0.01

✓ CONVERGENCE REACHED after 5 iterations!
```

### Classification Report

```
               precision    recall  f1-score   support
     Cylinder       0.20      0.11      0.14         9
     Revolute       1.00      0.67      0.80        18
```

- **Precision**: How many predictions were correct
- **Recall**: How many actual cases were found
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of samples

### Visualizations

1. **confusion_matrix.png**: Shows predicted vs actual joint types
2. **training_history.png**: Loss and accuracy across iterations

## 🔍 Using the Model Programmatically

### Load and Predict

```python
from predict_joint import JointPredictor
import pandas as pd

# Initialize predictor
predictor = JointPredictor(
    model_path='joint_predictor_model.keras',
    preprocessing_path='preprocessing_objects.pkl'
)

# Predict from DataFrame
df = pd.read_excel('new_data.xlsx')
predictions, confidence, probabilities = predictor.predict(df)

# Or predict single sample
joint, conf = predictor.predict_single(
    Classifier_1=0,
    Classifier_2=1,
    Touch_Plane_Plane=1,
    Touch_Cylinder_Cylinder=0,
    # ... all 41 features
    Count=2
)

print(f"Predicted Joint: {joint} (Confidence: {conf:.2%})")
```

## 🎯 Training Strategy

The model uses **iterative retraining with convergence**:

1. **Initial Training**: Train model for up to 150 epochs
2. **Evaluation**: Calculate test loss
3. **Convergence Check**: If loss change < tolerance → STOP
4. **Retrain**: Rebuild model and train again
5. **Repeat**: Until converged or max iterations reached

This ensures the model finds a stable, reliable solution.

## 📉 Improving Accuracy

If accuracy is low, try:

1. **Increase training data**: More samples = better learning
2. **Adjust model complexity**: More layers/neurons
3. **Tune hyperparameters**: Learning rate, dropout, batch size
4. **Feature engineering**: Create new features from existing ones
5. **Handle class imbalance**: Use class weights or resampling

### Example: Add Class Weights

```python
# In train_with_convergence() method
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

history = self.model.fit(
    X_train, y_train,
    class_weight=class_weight_dict,  # Add this
    # ... other parameters
)
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Low accuracy** | Increase `MAX_ITERATIONS`, adjust model architecture, collect more data |
| **Overfitting** (high train acc, low test acc) | Increase dropout, add L2 regularization, reduce model complexity |
| **Not converging** | Decrease learning rate, increase `MAX_ITERATIONS` |
| **Memory errors** | Reduce batch size, simplify model |
| **Import errors** | Run `pip install -r requirements.txt` |

## 📝 Example Workflow

```bash
# 1. Train model
python train_joint_predictor.py

# Output:
# ✓ CONVERGENCE REACHED after 5 iterations!
# Final test accuracy: 0.3824 (38.24%)

# 2. Check visualizations
# Open: confusion_matrix.png, training_history.png

# 3. Make predictions
python predict_joint.py

# Output:
# Overall Accuracy: 0.3831 (38.31%)
# ✓ Full predictions saved to: predictions.csv

# 4. Analyze results
# Open: predictions.csv
```

## 🔬 Model Architecture

```
Input (41 features)
       ↓
Dense(256) + ReLU + L2(0.001)
       ↓
BatchNormalization
       ↓
Dropout(0.4)
       ↓
Dense(128) + ReLU + L2(0.001)
       ↓
BatchNormalization
       ↓
Dropout(0.3)
       ↓
Dense(64) + ReLU
       ↓
Dropout(0.2)
       ↓
Dense(9) + Softmax
       ↓
Output (9 joint types)
```

**Regularization Techniques:**
- L2 regularization (prevents overfitting)
- Batch normalization (stabilizes training)
- Dropout (prevents co-adaptation)
- Early stopping (stops when validation loss plateaus)
- Learning rate reduction (adapts learning rate)

## 📚 Additional Resources

- **Detailed Documentation**: See `ML_README.md`
- **TensorFlow Docs**: https://www.tensorflow.org/
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html

## ✅ Success Criteria

Your model is working well if:
- ✓ Convergence reached (loss stabilizes)
- ✓ Test accuracy > 70% (depends on data complexity)
- ✓ Precision/recall balanced (no extreme bias)
- ✓ Training and test accuracy similar (no overfitting)

---

**Happy Predicting! 🚀**

For questions or issues, refer to `ML_README.md` for detailed explanations.
