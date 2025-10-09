"""
Joint Type Prediction Model using TensorFlow
Trains on 120 samples and iteratively retrains until convergence
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class JointPredictor:
    def __init__(self, dataset_path, train_size=120, convergence_tolerance=0.001, max_iterations=50):
        """
        Initialize the Joint Predictor
        
        Args:
            dataset_path: Path to the dataset Excel file
            train_size: Number of samples to use for training (default: 120)
            convergence_tolerance: Tolerance for convergence checking (default: 0.001)
            max_iterations: Maximum number of retraining iterations (default: 50)
        """
        self.dataset_path = dataset_path
        self.train_size = train_size
        self.convergence_tolerance = convergence_tolerance
        self.max_iterations = max_iterations
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.training_history = []
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("=" * 80)
        print("Loading and preprocessing data...")
        print("=" * 80)
        
        # Load dataset
        df = pd.read_excel(self.dataset_path)
        print(f"\nDataset shape: {df.shape}")
        print(f"Total samples: {len(df)}")
        
        # Standardize target variable (handle case variations)
        df['Joint'] = df['Joint'].str.strip().str.capitalize()
        
        # Display target distribution
        print("\nTarget variable distribution:")
        print(df['Joint'].value_counts())
        
        # Identify feature columns (exclude Part1, Part2, and Joint)
        # Part1 and Part2 are component identifiers, we'll use Classifier columns instead
        exclude_cols = ['Part1', 'Part2', 'Joint']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features (Classifier_1, Classifier_2, Touch_Line_Line if object)
        df_processed = df.copy()
        for col in self.feature_columns:
            if df[col].dtype == 'object':
                # Convert to numeric if possible, otherwise use label encoding
                try:
                    df_processed[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df[col].astype(str))
        
        # Fill any NaN values with 0
        df_processed = df_processed.fillna(0)
        
        # Extract features and target
        X = df_processed[self.feature_columns].values
        y = df_processed['Joint'].values
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        
        print(f"\nNumber of features: {len(self.feature_columns)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        return X, y_encoded
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        print("\n" + "=" * 80)
        print("Splitting data...")
        print("=" * 80)
        
        # Calculate test size
        test_size = len(X) - self.train_size
        
        # Check if stratification is possible
        from collections import Counter
        class_counts = Counter(y)
        min_class_count = min(class_counts.values())
        
        # Only use stratification if all classes have at least 2 samples
        if min_class_count >= 2:
            print(f"Using stratified split (all classes have >= 2 samples)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=self.train_size,
                test_size=test_size,
                random_state=42,
                stratify=y
            )
        else:
            print(f"Using random split (some classes have only 1 sample)")
            print(f"Classes with single samples: {[k for k, v in class_counts.items() if v == 1]}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                train_size=self.train_size,
                test_size=test_size,
                random_state=42,
                shuffle=True
            )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Training set class distribution: {Counter(y_train)}")
        print(f"Test set class distribution: {Counter(y_test)}")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim):
        """Build a neural network model"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_convergence(self, X_train, X_test, y_train, y_test):
        """Train model iteratively until convergence"""
        print("\n" + "=" * 80)
        print("Starting iterative training with convergence checking...")
        print("=" * 80)
        
        iteration = 0
        previous_loss = float('inf')
        convergence_reached = False
        
        # Early stopping and learning rate reduction callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        while iteration < self.max_iterations and not convergence_reached:
            iteration += 1
            print(f"\n{'*' * 80}")
            print(f"ITERATION {iteration}/{self.max_iterations}")
            print(f"{'*' * 80}")
            
            # Build or rebuild the model
            self.model = self.build_model(X_train.shape[1])
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=8,
                validation_data=(X_test, y_test),
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
            
            # Store training history
            iteration_info = {
                'iteration': iteration,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'epochs_trained': len(history.history['loss'])
            }
            self.training_history.append(iteration_info)
            
            # Print results
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.6f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"  Test Loss:  {test_loss:.6f} | Test Accuracy:  {test_accuracy:.4f}")
            print(f"  Epochs trained: {iteration_info['epochs_trained']}")
            
            # Check for convergence
            loss_change = abs(previous_loss - test_loss)
            print(f"\nLoss change from previous iteration: {loss_change:.6f}")
            print(f"Convergence tolerance: {self.convergence_tolerance}")
            
            if loss_change < self.convergence_tolerance:
                convergence_reached = True
                print(f"\n✓ CONVERGENCE REACHED after {iteration} iterations!")
            
            previous_loss = test_loss
            
        if not convergence_reached:
            print(f"\nMax iterations ({self.max_iterations}) reached.")
        
        return self.training_history
    
    def evaluate_final_model(self, X_test, y_test):
        """Evaluate the final model and display detailed metrics"""
        print("\n" + "=" * 80)
        print("FINAL MODEL EVALUATION")
        print("=" * 80)
        
        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nFinal Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report:")
        print("=" * 80)
        # Get unique labels present in both test and predictions
        unique_labels = sorted(list(set(y_test) | set(y_pred)))
        target_names = [self.label_encoder.classes_[i] for i in unique_labels]
        print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_names, 
                    yticklabels=target_names)
        plt.title('Confusion Matrix - Joint Type Prediction')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/workspace/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved to: confusion_matrix.png")
        plt.close()
        
        return accuracy, y_pred
    
    def plot_training_history(self):
        """Plot the training history across iterations"""
        print("\n" + "=" * 80)
        print("Generating training history plots...")
        print("=" * 80)
        
        iterations = [h['iteration'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        test_losses = [h['test_loss'] for h in self.training_history]
        train_accuracies = [h['train_accuracy'] for h in self.training_history]
        test_accuracies = [h['test_accuracy'] for h in self.training_history]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(iterations, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=8)
        axes[0].plot(iterations, test_losses, 's-', label='Test Loss', linewidth=2, markersize=8)
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Loss vs Iteration', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(iterations, train_accuracies, 'o-', label='Train Accuracy', linewidth=2, markersize=8)
        axes[1].plot(iterations, test_accuracies, 's-', label='Test Accuracy', linewidth=2, markersize=8)
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Accuracy vs Iteration', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Training history plot saved to: training_history.png")
        plt.close()
    
    def save_model(self):
        """Save the trained model"""
        model_path = '/workspace/joint_predictor_model.keras'
        self.model.save(model_path)
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save preprocessing objects
        import pickle
        with open('/workspace/preprocessing_objects.pkl', 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }, f)
        print("✓ Preprocessing objects saved to: preprocessing_objects.pkl")
    
    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "╔" + "═" * 78 + "╗")
        print("║" + " " * 20 + "JOINT TYPE PREDICTOR - TRAINING PIPELINE" + " " * 18 + "║")
        print("╚" + "═" * 78 + "╝\n")
        
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train with convergence
        self.train_with_convergence(X_train, X_test, y_train, y_test)
        
        # Evaluate final model
        accuracy, y_pred = self.evaluate_final_model(X_test, y_test)
        
        # Plot training history
        self.plot_training_history()
        
        # Save model
        self.save_model()
        
        # Summary
        print("\n" + "=" * 80)
        print("TRAINING SUMMARY")
        print("=" * 80)
        print(f"Total iterations: {len(self.training_history)}")
        print(f"Final test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Training samples: {self.train_size}")
        print(f"Test samples: {len(X_test)}")
        print(f"Convergence tolerance: {self.convergence_tolerance}")
        print("=" * 80)
        
        return self.model


def main():
    """Main function to run the training"""
    # Configuration
    DATASET_PATH = '/workspace/Constraint_dataset.xlsx'
    TRAIN_SIZE = 120
    CONVERGENCE_TOLERANCE = 0.01  # Relaxed tolerance for faster convergence
    MAX_ITERATIONS = 20  # Reduced for faster training
    
    # Create and run predictor
    predictor = JointPredictor(
        dataset_path=DATASET_PATH,
        train_size=TRAIN_SIZE,
        convergence_tolerance=CONVERGENCE_TOLERANCE,
        max_iterations=MAX_ITERATIONS
    )
    
    # Run complete training pipeline
    model = predictor.run_complete_training_pipeline()
    
    print("\n✓ Training completed successfully!")
    print("\nGenerated files:")
    print("  1. joint_predictor_model.keras - Trained model")
    print("  2. preprocessing_objects.pkl - Feature scaling and label encoding objects")
    print("  3. confusion_matrix.png - Confusion matrix visualization")
    print("  4. training_history.png - Training progress across iterations")


if __name__ == "__main__":
    main()
