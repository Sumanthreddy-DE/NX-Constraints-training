# NX Constraints to Motion Joints

Predicting which NX Motion joint belongs between two components of a Siemens NX assembly,
using only the mating constraints already defined between them.

Work carried out as a research assistant at the Chair of Engineering Design (KTmfk),
Friedrich-Alexander-Universität Erlangen-Nürnberg.

## The problem

When a CAD assembly is handed over for multibody simulation, the geometry and the mating
constraints come across, but the kinematic joints do not. Someone has to look at each pair of
mated components and decide whether that pairing is a revolute joint, a cylindrical joint, a
planar joint, a fixed connection, and so on. On a large assembly this is slow, repetitive, and
easy to get wrong in a way that only shows up once the simulation misbehaves.

The constraints themselves already carry most of the answer. Two parts touching plane to plane
and aligned cylinder to cylinder describe a different degree of freedom than two parts fixed
face to face. So the task can be posed as supervised classification: given the constraint
pattern between a component pair, predict the joint.

## Approach

Each component pair becomes one row. The features are the counts of each constraint type
between the two parts, roughly forty columns covering touch, align, parallel, perpendicular,
distance and related constraint kinds, plus two part classifiers and a total constraint count.
The target is the joint type.

The classifier is a small fully connected network in TensorFlow:

```
Input (41 features)
  Dense 128, ReLU, BatchNorm, Dropout 0.3
  Dense  64, ReLU, BatchNorm, Dropout 0.3
  Dense  32, ReLU, Dropout 0.2
  Dense  N classes, Softmax
```

Features are standardised before training. Training runs up to 100 epochs with early stopping,
then repeats the whole fit until the change in test loss falls below a tolerance or a retry
limit is reached, which keeps a single unlucky initialisation from deciding the result.

## Dataset

`Constraint_dataset.xlsx` holds 154 component pairs drawn from assemblies, split 120 for
training and the remainder for testing, stratified by joint type.

The class distribution is strongly imbalanced. Revolute accounts for 79 of the 154 pairs and
Cylinder for 37, while Slider, Perpendicular and Inline appear once each:

| Joint | Pairs |
|---|---|
| Revolute | 79 |
| Cylinder | 37 |
| Planar | 20 |
| Fixed | 9 |
| Parallel | 3 |
| Inplane | 3 |
| Slider | 1 |
| Perpendicular | 1 |
| Inline | 1 |

## Reading the results

A single accuracy figure is not meaningful on this dataset. With three classes represented by
one sample each, whether those land in train or test moves the number more than the model does,
and always predicting Revolute already scores above half.

`confusion_matrix.png` is the honest view. It shows which joint types the model separates
cleanly and which it confuses, so the frequent classes and the rare ones can be judged
separately. The per-class precision, recall and F1 in the classification report printed at the
end of training serve the same purpose. `training_history.png` shows loss and accuracy across
the retraining iterations, which is where non-convergence would show up.

`predictions.csv` lists every component pair with its actual joint, predicted joint and
confidence, so individual disagreements can be traced back to specific parts.

## Running it

```bash
pip install -r requirements.txt
```

Train, which writes the model, the preprocessing objects and both plots:

```bash
python train_joint_predictor.py
```

Predict with the committed model:

```bash
python predict_joint.py
```

Or from your own code:

```python
from predict_joint import JointPredictor

predictor = JointPredictor(
    model_path='joint_predictor_model.keras',
    preprocessing_path='preprocessing_objects.pkl'
)

predictions, confidence, probabilities = predictor.predict('new_data.xlsx')
```

Input data must carry the same feature columns as `Constraint_dataset.xlsx`.

## Files

| File | What it is |
|---|---|
| `Constraint_dataset.xlsx` | The 154 labelled component pairs |
| `train_joint_predictor.py` | Training with iterative refitting until convergence |
| `predict_joint.py` | `JointPredictor` class, file and single-sample prediction |
| `joint_predictor_model.keras` | Trained model |
| `preprocessing_objects.pkl` | Fitted scaler and label encoder |
| `confusion_matrix.png` | Per-class confusion matrix |
| `training_history.png` | Loss and accuracy across retraining iterations |
| `predictions.csv` | Actual against predicted joint, per component pair |

`ML_README.md`, `USAGE_GUIDE.md`, `PROJECT_SUMMARY.md` and `FILES_OVERVIEW.txt` hold the longer
working notes, including how to change the architecture and the convergence settings.

## Limitations

The dataset is small and imbalanced, and it comes from a limited set of assemblies, so the model
should be treated as a first pass that a human confirms rather than an automatic converter. Rare
joint types have too few examples to learn from. Growing the dataset, particularly for the
underrepresented joints, is the change that would help most.
