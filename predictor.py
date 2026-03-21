"""
Rockfall Prediction Model Implementation
=======================================

This module contains the main prediction model for rockfall hazard assessment.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from typing import Dict, Any, Union, Tuple
import joblib
import logging
from pathlib import Path

class RockfallPredictor:
    """Main class for rockfall hazard prediction."""

    def __init__(self, config: Dict = None):
        """Initialize the predictor with configuration."""
        self.config = config or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)

        # Initialize model based on configuration
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ML model based on configuration."""
        algorithm = self.config.get('algorithm', 'random_forest')
        hyperparams = self.config.get('hyperparameters', {}).get(algorithm, {})

        if algorithm == 'random_forest':
            self.model = RandomForestClassifier(**hyperparams)
        elif algorithm == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**hyperparams)
        else:
            self.logger.warning(f"Unknown algorithm: {algorithm}, using Random Forest")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model on the provided data."""
        self.logger.info("Starting model training...")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Calculate training metrics
        y_pred_train = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_pred_train)

        self.logger.info(f"Model training completed. Training accuracy: {train_accuracy:.4f}")

        return {
            'train_accuracy': train_accuracy,
            'n_features': len(self.feature_names),
            'n_samples': len(X_train)
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

        # Get classification report as dictionary
        class_report = classification_report(y_test, y_pred, output_dict=True)

        metrics = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        self.logger.info(f"Model evaluation completed. Test accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

        return metrics

    def predict(self, sample_data: Union[Dict, pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction for a single sample or batch."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Convert single sample to DataFrame if needed
        if isinstance(sample_data, dict):
            # Ensure all required features are present
            feature_array = []
            for feature in self.feature_names:
                if feature in sample_data:
                    feature_array.append(sample_data[feature])
                else:
                    self.logger.warning(f"Feature {feature} not found in input, using default value")
                    feature_array.append(0)  # Default value

            X = pd.DataFrame([feature_array], columns=self.feature_names)
        else:
            X = sample_data[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        # Prepare results
        results = []
        for i in range(len(X)):
            result = {
                'predicted_class': 'High Risk' if predictions[i] else 'Low/Medium Risk',
                'high_risk_probability': probabilities[i][1],
                'low_medium_probability': probabilities[i][0],
                'risk_score': probabilities[i][1] * 100,
                'confidence': max(probabilities[i])
            }
            results.append(result)

        return results[0] if isinstance(sample_data, dict) else results

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            return importance_df
        else:
            self.logger.warning("Model does not have feature importance attribute")
            return pd.DataFrame()

    def predict_from_file(self, file_path: str, output_path: str = None) -> pd.DataFrame:
        """Make predictions for data from a CSV file."""
        # Load data
        df = pd.read_csv(file_path)

        # Make predictions
        predictions = self.predict(df)

        # Add predictions to dataframe
        df['predicted_class'] = [pred['predicted_class'] for pred in predictions]
        df['risk_score'] = [pred['risk_score'] for pred in predictions]
        df['confidence'] = [pred['confidence'] for pred in predictions]

        # Save results if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Predictions saved to {output_path}")

        return df

    def interactive_predict(self) -> Dict[str, Any]:
        """Interactive prediction mode for user input."""
        print("\n=== Interactive Rockfall Risk Prediction ===")
        print("Please enter the following parameters:")

        sample_data = {}

        # Define feature descriptions and typical ranges
        feature_info = {
            'rock_mass_rating': ('Rock Mass Rating (0-100)', (0, 100)),
            'geological_strength_index': ('Geological Strength Index (0-100)', (0, 100)),
            'slope_height_m': ('Slope Height (meters)', (10, 200)),
            'slope_angle_deg': ('Slope Angle (degrees)', (30, 80)),
            'weathering_grade': ('Weathering Grade (1-5)', (1, 5)),
            'rainfall_mm_annual': ('Annual Rainfall (mm)', (200, 3000)),
            'joint_spacing_m': ('Joint Spacing (meters)', (0.1, 3.0)),
            'friction_angle_deg': ('Friction Angle (degrees)', (15, 50)),
            'cohesion_kpa': ('Cohesion (kPa)', (5, 100)),
            'pore_pressure_kpa': ('Pore Pressure (kPa)', (0, 200)),
            'vibration_intensity_mm_s': ('Vibration Intensity (mm/s)', (0, 30)),
            'block_volume_m3': ('Block Volume (cubic meters)', (0.1, 20))
        }

        for feature in self.feature_names:
            if feature in feature_info:
                description, (min_val, max_val) = feature_info[feature]
                while True:
                    try:
                        value = float(input(f"{description} ({min_val}-{max_val}): "))
                        if min_val <= value <= max_val:
                            sample_data[feature] = value
                            break
                        else:
                            print(f"Please enter a value between {min_val} and {max_val}")
                    except ValueError:
                        print("Please enter a valid number")

        # Make prediction
        result = self.predict(sample_data)

        print("\n=== Prediction Result ===")
        print(f"Predicted Risk Level: {result['predicted_class']}")
        print(f"Risk Score: {result['risk_score']:.1f}%")
        print(f"Confidence: {result['confidence']:.3f}")

        return result

    def save_model(self, save_path: str):
        """Save the trained model and scaler."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_path / self.config.get('model_file', 'rockfall_predictor.pkl')
        joblib.dump(self.model, model_path)

        # Save scaler
        scaler_path = save_path / self.config.get('scaler_file', 'feature_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'model_type': type(self.model).__name__
        }
        metadata_path = save_path / 'model_metadata.pkl'
        joblib.dump(metadata, metadata_path)

        self.logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: str):
        """Load a previously trained model."""
        model_path = Path(model_path)

        # Load model
        model_file = model_path / self.config.get('model_file', 'rockfall_predictor.pkl')
        self.model = joblib.load(model_file)

        # Load scaler
        scaler_file = model_path / self.config.get('scaler_file', 'feature_scaler.pkl')
        self.scaler = joblib.load(scaler_file)

        # Load metadata
        metadata_file = model_path / 'model_metadata.pkl'
        metadata = joblib.load(metadata_file)

        self.feature_names = metadata['feature_names']
        self.is_trained = metadata['is_trained']

        self.logger.info(f"Model loaded from {model_path}")
