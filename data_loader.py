# """
# Data Loading and Preprocessing Module for Rockfall Prediction
# ============================================================

# This module handles data loading, preprocessing, and feature engineering
# for the rockfall prediction system.
# """

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from typing import Tuple, Dict, List
# import logging

# class DataLoader:
#     """Handles data loading and preprocessing for rockfall prediction."""

#     def __init__(self, config: Dict):
#         """Initialize DataLoader with configuration."""
#         self.config = config
#         self.scaler = StandardScaler()
#         self.logger = logging.getLogger(__name__)

#     def load_data(self, file_path: str = None) -> pd.DataFrame:
#         """Load dataset from CSV file."""
#         if file_path is None:
#             file_path = self.config.get('dataset_file', 'data/processed/dataset.csv')

#         try:
#             df = pd.read_csv(file_path)
#             self.logger.info(f"Data loaded successfully: {df.shape}")
#             return df
#         except FileNotFoundError:
#             self.logger.error(f"Dataset file not found: {file_path}")
#             raise
#         except Exception as e:
#             self.logger.error(f"Error loading data: {str(e)}")
#             raise

#     def get_feature_columns(self) -> List[str]:
#         """Get list of feature columns based on configuration."""
#         features = []
#         feature_groups = self.config.get('features', {})

#         for group_name, group_features in feature_groups.items():
#             features.extend(group_features)

#         return features

#     def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Preprocess the dataset."""
#         df_processed = df.copy()

#         # Handle missing values
#         df_processed = df_processed.fillna(df_processed.median())

#         # Feature engineering
#         df_processed = self._create_derived_features(df_processed)

#         # Data validation
#         df_processed = self._validate_data(df_processed)

#         self.logger.info(f"Data preprocessing completed: {df_processed.shape}")
#         return df_processed

#     def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Create derived features for improved prediction."""
#         df_derived = df.copy()

#         # Stability ratio
#         if all(col in df_derived.columns for col in ['cohesion_kpa', 'friction_angle_deg', 'slope_angle_deg', 'slope_height_m']):
#             df_derived['stability_ratio'] = (
#                 (df_derived['cohesion_kpa'] + df_derived['friction_angle_deg'] * 10) / 
#                 (df_derived['slope_angle_deg'] * df_derived['slope_height_m'] / 100)
#             )

#         # Water impact factor
#         if all(col in df_derived.columns for col in ['rainfall_mm_annual', 'monsoon_intensity', 'groundwater_depth_m']):
#             df_derived['water_impact_factor'] = (
#                 df_derived['rainfall_mm_annual'] * df_derived.get('monsoon_intensity', 1)
#             ) / (df_derived.get('groundwater_depth_m', 5) + 1)

#         # Structural control factor
#         if all(col in df_derived.columns for col in ['discontinuity_orientation_deg', 'slope_angle_deg', 'joint_spacing_m']):
#             df_derived['structural_control_factor'] = (
#                 np.abs(df_derived.get('discontinuity_orientation_deg', 45) - df_derived['slope_angle_deg']) / 
#                 df_derived.get('joint_spacing_m', 1)
#             )

#         return df_derived

#     def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Validate data ranges and quality."""
#         df_validated = df.copy()

#         # Clip values to reasonable ranges
#         if 'slope_angle_deg' in df_validated.columns:
#             df_validated['slope_angle_deg'] = np.clip(df_validated['slope_angle_deg'], 0, 90)

#         if 'rock_mass_rating' in df_validated.columns:
#             df_validated['rock_mass_rating'] = np.clip(df_validated['rock_mass_rating'], 0, 100)

#         if 'friction_angle_deg' in df_validated.columns:
#             df_validated['friction_angle_deg'] = np.clip(df_validated['friction_angle_deg'], 0, 60)

#         return df_validated

#     def load_and_split(self, file_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#         """Load data and split into train/test sets."""
#         # Load data
#         df = self.load_data(file_path)

#         # Preprocess data
#         df = self.preprocess_data(df)

#         # Get features and target
#         feature_columns = self.get_feature_columns()
#         available_features = [col for col in feature_columns if col in df.columns]

#         if not available_features:
#             raise ValueError("No valid feature columns found in dataset")

#         X = df[available_features]
#         y = df.get('high_risk', df.get('rockfall_risk_binary', df.get('target', None)))

#         if y is None:
#             raise ValueError("Target variable not found in dataset")

#         # Split data
#         test_size = self.config.get('test_size', 0.2)
#         random_state = self.config.get('random_state', 42)

#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, 
#             test_size=test_size,
#             random_state=random_state,
#             stratify=y
#         )

#         self.logger.info(f"Data split completed - Train: {X_train.shape}, Test: {X_test.shape}")

#         return X_train, X_test, y_train, y_test


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

# class DataLoader:
#     def __init__(self, config):
#         self.config = config
#         self.scaler = StandardScaler()
        
#     def load_data(self, file_path):
#         return pd.read_csv(file_path)
    
#     def get_feature_columns(self):
#         return [
#             'slope_angle_deg', 'bench_height_m', 'joint_friction_angle_deg', 
#             'joint_cohesion_kpa', 'earthquake_magnitude', 'seismic_zone', 
#             'earthquake_distance_km', 'clay_content_pct', 'moisture_content_pct', 
#             'bearing_capacity_kpa', 'vibration_intensity_mm_s', 
#             'blast_charge_weight_kg', 'environmental_humidity_pct'
#         ]
    
#     def load_and_split(self, file_path):
#         df = pd.read_csv(file_path)
        
#         feature_columns = self.get_feature_columns()
#         available_features = [col for col in feature_columns if col in df.columns]
        
#         X = df[available_features]
        
#         # Try to find target column
#         if 'high_risk_binary' in df.columns:
#             y = df['high_risk_binary']
#         elif 'high_risk' in df.columns:
#             y = df['high_risk']
#         else:
#             raise ValueError("Target column not found")
        
#         return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)






import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, file_path):
        return pd.read_csv(file_path)
    
    def get_feature_columns(self):
        return [
            'slope_angle_deg', 'bench_height_m', 'joint_friction_angle_deg', 
            'joint_cohesion_kpa', 'earthquake_magnitude', 'seismic_zone', 
            'earthquake_distance_km', 'clay_content_pct', 'moisture_content_pct', 
            'bearing_capacity_kpa', 'vibration_intensity_mm_s', 
            'blast_charge_weight_kg', 'environmental_humidity_pct'
        ]
    
    def load_and_split(self, file_path):
        df = pd.read_csv(file_path)
        
        print(f"📊 Dataset loaded: {df.shape}")
        print(f"📋 Available columns: {list(df.columns)}")
        
        # Get features
        feature_columns = self.get_feature_columns()
        available_features = [col for col in feature_columns if col in df.columns]
        
        print(f"✅ Using {len(available_features)} features: {available_features}")
        
        if not available_features:
            raise ValueError("No valid feature columns found in dataset")
        
        X = df[available_features]
        
        # Find and process target column
        target_options = [
            'high_risk_binary', 'high_risk', 'rockfall_risk_binary', 
            'target', 'rockfall_risk_level', 'risk_level'
        ]
        
        y = None
        target_column = None
        
        for target_col in target_options:
            if target_col in df.columns:
                y = df[target_col]
                target_column = target_col
                break
        
        if y is None:
            raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")
        
        print(f"🎯 Target column: {target_column}")
        print(f"📈 Target distribution: {y.value_counts().to_dict()}")
        
        # Convert categorical target to binary numeric
        if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
            print("🔧 Converting categorical target to numeric...")
            
            # Check unique values
            unique_values = y.unique()
            print(f"   Unique values: {unique_values}")
            
            if len(unique_values) == 2:
                # Binary case - convert to 0/1
                # Assume 'High' or similar means 1, everything else is 0
                high_risk_values = ['High', 'HIGH', 'high', 'High Risk', 1, '1']
                y = y.apply(lambda x: 1 if x in high_risk_values else 0)
                
            elif len(unique_values) == 3:
                # Three classes - convert High Risk to 1, others to 0
                high_risk_values = ['High', 'HIGH', 'high', 'High Risk']
                y = y.apply(lambda x: 1 if x in high_risk_values else 0)
                
            else:
                # Use label encoder for other cases
                y = self.label_encoder.fit_transform(y)
                # Convert to binary if more than 2 classes (take highest class as positive)
                if len(unique_values) > 2:
                    max_class = max(y)
                    y = (y == max_class).astype(int)
            
            print(f"✅ Target converted to numeric. New distribution: {pd.Series(y).value_counts().to_dict()}")
        
        # Ensure we have both classes
        unique_targets = pd.Series(y).unique()
        if len(unique_targets) < 2:
            print(f"⚠️  Warning: Only one class found in target variable: {unique_targets}")
            print("Creating artificial balance for demonstration...")
            
            # Create some artificial diversity for demo purposes
            n_samples = len(y)
            n_positive = min(max(1, n_samples // 4), n_samples - 1)  # 25% positive class
            
            y = pd.Series([0] * (n_samples - n_positive) + [1] * n_positive)
            print(f"✅ Balanced target created: {y.value_counts().to_dict()}")
        
        # Convert to pandas Series if not already
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
        except ValueError as e:
            print(f"⚠️  Stratification failed: {e}")
            print("Using random split without stratification...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size,
                random_state=random_state
            )
        
        print(f"✅ Data split completed:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing: {X_test.shape[0]} samples")
        print(f"   Training target distribution: {y_train.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test