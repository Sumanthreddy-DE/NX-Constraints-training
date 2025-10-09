"""
Joint Type Prediction - Inference Script
Use this to make predictions on new data using the trained model
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')


class JointPredictor:
    def __init__(self, model_path, preprocessing_path):
        """
        Initialize the predictor with trained model and preprocessing objects
        
        Args:
            model_path: Path to the saved Keras model
            preprocessing_path: Path to the preprocessing objects pickle file
        """
        # Load the model
        self.model = keras.models.load_model(model_path)
        
        # Load preprocessing objects
        with open(preprocessing_path, 'rb') as f:
            preprocess_obj = pickle.load(f)
            self.scaler = preprocess_obj['scaler']
            self.label_encoder = preprocess_obj['label_encoder']
            self.feature_columns = preprocess_obj['feature_columns']
        
        print("✓ Model and preprocessing objects loaded successfully!")
        print(f"  Features: {len(self.feature_columns)}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
    
    def preprocess_input(self, df):
        """Preprocess input data"""
        # Ensure all feature columns are present
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select and order features correctly
        df_features = df[self.feature_columns].copy()
        
        # Handle categorical features
        for col in self.feature_columns:
            if df_features[col].dtype == 'object':
                try:
                    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
                except:
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_features[col] = le.fit_transform(df_features[col].astype(str))
        
        # Fill NaN values
        df_features = df_features.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.transform(df_features.values)
        
        return X_scaled
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Args:
            data: DataFrame with the same features as training data, or path to Excel/CSV file
        
        Returns:
            predictions: Array of predicted joint types
            probabilities: Array of prediction probabilities
        """
        # Load data if path is provided
        if isinstance(data, str):
            if data.endswith('.xlsx'):
                df = pd.read_excel(data)
            elif data.endswith('.csv'):
                df = pd.read_csv(data)
            else:
                raise ValueError("Unsupported file format. Use .xlsx or .csv")
        else:
            df = data.copy()
        
        # Preprocess
        X = self.preprocess_input(df)
        
        # Predict
        probabilities = self.model.predict(X, verbose=0)
        predictions_encoded = np.argmax(probabilities, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores, probabilities
    
    def predict_single(self, **features):
        """
        Predict for a single sample using keyword arguments
        
        Example:
            predictor.predict_single(
                Classifier_1=0,
                Classifier_2=1,
                Touch_Plane_Plane=1,
                ...
            )
        """
        df = pd.DataFrame([features])
        predictions, confidence, _ = self.predict(df)
        return predictions[0], confidence[0]


def main():
    """Example usage of the predictor"""
    # Paths to model and preprocessing objects
    MODEL_PATH = '/workspace/joint_predictor_model.keras'
    PREPROCESSING_PATH = '/workspace/preprocessing_objects.pkl'
    
    # Initialize predictor
    predictor = JointPredictor(MODEL_PATH, PREPROCESSING_PATH)
    
    # Example: Predict on test dataset
    print("\n" + "=" * 80)
    print("EXAMPLE: Making predictions on the original dataset")
    print("=" * 80)
    
    # Load the dataset
    df = pd.read_excel('/workspace/Constraint_dataset.xlsx')
    
    # Make predictions
    predictions, confidence_scores, probabilities = predictor.predict(df)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Part1': df['Part1'],
        'Part2': df['Part2'],
        'Actual_Joint': df['Joint'].str.strip().str.capitalize(),
        'Predicted_Joint': predictions,
        'Confidence': confidence_scores
    })
    
    # Calculate accuracy
    accuracy = (results['Actual_Joint'] == results['Predicted_Joint']).mean()
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nSample predictions:")
    print(results.head(10).to_string(index=False))
    
    # Save predictions
    results.to_csv('/workspace/predictions.csv', index=False)
    print("\n✓ Full predictions saved to: predictions.csv")
    
    # Show misclassifications
    misclassified = results[results['Actual_Joint'] != results['Predicted_Joint']]
    if len(misclassified) > 0:
        print(f"\nMisclassified samples: {len(misclassified)}")
        print("\nTop misclassifications:")
        print(misclassified.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
