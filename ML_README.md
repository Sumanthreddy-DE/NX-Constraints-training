# Joint Type Prediction Model

A machine learning system to predict joint types in NX assembly based on constraint data between components.

## Overview

This project trains a neural network model using TensorFlow to predict joint types (Revolute, Cylinder, Planar, Fixed, etc.) based on constraint features between assembly components.

### Key Features

- **Iterative Training with Convergence**: Automatically retrains the model until error converges to tolerance levels
- **120/34 Train-Test Split**: Uses 120 samples for training and remaining samples for testing
- **Deep Neural Network**: Multi-layer architecture with batch normalization and dropout for robust predictions
- **Comprehensive Evaluation**: Provides accuracy metrics, classification reports, and confusion matrices
- **Visualization**: Generates training history plots and confusion matrix visualizations

## Dataset Structure

The dataset contains the following information:

- **Part1, Part2**: Component names between which constraints are established
- **Classifier_1, Classifier_2**: Classifiers to identify the parts
- **Constraint Features** (40+ columns): Various constraint types like:
  - Touch constraints (Plane-Plane, Cylinder-Cylinder, etc.)
  - Align constraints
  - Parallel/Perpendicular constraints
  - Distance constraints
  - And more...
- **Count**: Number of constraints between components
- **Joint**: Target variable - the joint type to predict

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

Run the training script with iterative retraining until convergence:

```bash
python train_joint_predictor.py
```

**Configuration parameters** (edit in `train_joint_predictor.py`):
- `TRAIN_SIZE = 120`: Number of samples for training
- `CONVERGENCE_TOLERANCE = 0.001`: Error tolerance for convergence
- `MAX_ITERATIONS = 50`: Maximum retraining iterations

**Output files**:
- `joint_predictor_model.keras`: Trained TensorFlow model
- `preprocessing_objects.pkl`: Scaler and label encoder
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Loss and accuracy plots across iterations

### 2. Make Predictions

Use the trained model to predict joint types:

```bash
python predict_joint.py
```

Or use it programmatically:

```python
from predict_joint import JointPredictor

# Initialize predictor
predictor = JointPredictor(
    model_path='joint_predictor_model.keras',
    preprocessing_path='preprocessing_objects.pkl'
)

# Predict from file
predictions, confidence, probabilities = predictor.predict('new_data.xlsx')

# Or predict single sample
joint, confidence = predictor.predict_single(
    Classifier_1=0,
    Classifier_2=1,
    Touch_Plane_Plane=1,
    Touch_Cylinder_Cylinder=0,
    Align_Plane_Plane=0,
    # ... other features
    Count=2
)
```

## Model Architecture

The neural network consists of:

```
Input Layer (41 features)
    ↓
Dense Layer (128 units, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (64 units, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (32 units, ReLU) + Dropout(0.2)
    ↓
Output Layer (N classes, Softmax)
```

## Training Process

1. **Data Loading**: Reads Excel dataset and preprocesses features
2. **Data Splitting**: 120 samples for training, rest for testing (stratified)
3. **Feature Scaling**: StandardScaler normalization
4. **Iterative Training**:
   - Trains model for up to 100 epochs with early stopping
   - Evaluates on test set
   - Checks if loss change < convergence tolerance
   - Retrains if not converged
   - Stops when converged or max iterations reached
5. **Evaluation**: Generates classification report and confusion matrix
6. **Model Saving**: Saves trained model and preprocessing objects

## Results Interpretation

After training, you'll see:

- **Training/Test Accuracy**: Model performance on both sets
- **Classification Report**: Precision, recall, F1-score for each joint type
- **Confusion Matrix**: Visual representation of prediction accuracy
- **Convergence History**: How loss and accuracy evolved across iterations

## Files Generated

| File | Description |
|------|-------------|
| `joint_predictor_model.keras` | Trained neural network model |
| `preprocessing_objects.pkl` | Feature scaler and label encoder |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `training_history.png` | Training metrics across iterations |
| `predictions.csv` | Prediction results on dataset |

## Customization

### Adjust Training Parameters

In `train_joint_predictor.py`:

```python
# Change train/test split
TRAIN_SIZE = 120  # Number of training samples

# Adjust convergence criteria
CONVERGENCE_TOLERANCE = 0.001  # Lower = stricter convergence
MAX_ITERATIONS = 50  # Maximum retraining attempts
```

### Modify Model Architecture

In the `build_model` method:

```python
def build_model(self, input_dim):
    model = keras.Sequential([
        # Add/remove layers as needed
        layers.Dense(128, activation='relu'),
        # ... customize architecture
    ])
    return model
```

## Troubleshooting

**Issue**: "ModuleNotFoundError"
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: "KeyError: Column not found"
- **Solution**: Ensure input data has all required feature columns

**Issue**: Low accuracy
- **Solution**: Try increasing `MAX_ITERATIONS`, adjusting model architecture, or collecting more training data

## License

This project is provided as-is for educational and research purposes.
