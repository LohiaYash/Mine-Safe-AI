# Create a comprehensive dataset structure for rockfall prediction in Indian open pit mines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Generate synthetic but realistic dataset based on research findings
np.random.seed(42)

# Number of samples for the synthetic dataset
n_samples = 1000

# Generate features based on research literature
data = {
    # Geological features
    'rock_mass_rating': np.random.normal(50, 15, n_samples),  # RMR values
    'geological_strength_index': np.random.normal(45, 12, n_samples),  # GSI values
    'ucs_mpa': np.random.lognormal(3.5, 0.8, n_samples),  # Unconfined compressive strength
    'joint_spacing_m': np.random.exponential(0.5, n_samples),  # Joint spacing in meters
    'joint_persistence_m': np.random.exponential(2.0, n_samples),  # Joint persistence
    'joint_aperture_mm': np.random.exponential(2.0, n_samples),  # Joint aperture
    'weathering_grade': np.random.randint(1, 6, n_samples),  # 1-5 scale
    
    # Slope geometry features
    'slope_height_m': np.random.uniform(10, 150, n_samples),  # Slope height
    'slope_angle_deg': np.random.uniform(30, 70, n_samples),  # Slope angle
    'bench_height_m': np.random.uniform(8, 15, n_samples),  # Bench height
    'bench_width_m': np.random.uniform(5, 20, n_samples),  # Bench width
    'overall_slope_angle_deg': np.random.uniform(25, 45, n_samples),  # Overall slope angle
    
    # Environmental features
    'rainfall_mm_annual': np.random.normal(1200, 400, n_samples),  # Annual rainfall (India specific)
    'groundwater_depth_m': np.random.exponential(5, n_samples),  # Groundwater depth
    'pore_pressure_kpa': np.random.exponential(50, n_samples),  # Pore water pressure
    'monsoon_intensity': np.random.randint(1, 5, n_samples),  # 1-4 scale for monsoon intensity
    
    # Operational features
    'blast_distance_m': np.random.exponential(20, n_samples),  # Distance from blast
    'vibration_intensity_mm_s': np.random.exponential(10, n_samples),  # Peak particle velocity
    'mining_rate_tons_day': np.random.lognormal(8, 0.5, n_samples),  # Daily mining rate
    'equipment_loading_kn': np.random.exponential(500, n_samples),  # Equipment loading
    
    # Structural features
    'discontinuity_orientation_deg': np.random.uniform(0, 180, n_samples),  # Discontinuity strike
    'dip_angle_deg': np.random.uniform(20, 90, n_samples),  # Dip angle
    'block_volume_m3': np.random.lognormal(1, 1.5, n_samples),  # Potential block volume
    'friction_angle_deg': np.random.normal(32, 5, n_samples),  # Internal friction angle
    'cohesion_kpa': np.random.exponential(25, n_samples),  # Cohesion
}

# Ensure realistic bounds
data['rock_mass_rating'] = np.clip(data['rock_mass_rating'], 0, 100)
data['geological_strength_index'] = np.clip(data['geological_strength_index'], 0, 100)
data['weathering_grade'] = np.clip(data['weathering_grade'], 1, 5)
data['monsoon_intensity'] = np.clip(data['monsoon_intensity'], 1, 4)
data['friction_angle_deg'] = np.clip(data['friction_angle_deg'], 15, 50)

# Create DataFrame
df = pd.DataFrame(data)

# Calculate composite stability indicators
df['stability_ratio'] = (df['cohesion_kpa'] + df['friction_angle_deg'] * 10) / (df['slope_angle_deg'] * df['slope_height_m'] / 100)
df['water_impact_factor'] = (df['rainfall_mm_annual'] * df['monsoon_intensity']) / (df['groundwater_depth_m'] + 1)
df['structural_control_factor'] = np.abs(df['discontinuity_orientation_deg'] - df['slope_angle_deg']) / df['joint_spacing_m']

# Create rockfall risk classification
# Using a complex formula based on multiple factors
risk_score = (
    (100 - df['rock_mass_rating']) * 0.3 +
    (df['slope_angle_deg'] - 30) * 0.2 +
    (df['slope_height_m'] / 150 * 100) * 0.2 +
    (df['weathering_grade'] - 1) * 25 * 0.1 +
    (df['water_impact_factor'] / df['water_impact_factor'].max() * 100) * 0.1 +
    (df['block_volume_m3'] / df['block_volume_m3'].max() * 100) * 0.1
)

# Define risk categories
df['rockfall_risk'] = pd.cut(risk_score, 
                            bins=[0, 30, 60, 100], 
                            labels=['Low', 'Medium', 'High'],
                            include_lowest=True)

# Create binary classification for high-risk scenarios
df['high_risk'] = (df['rockfall_risk'] == 'High').astype(int)

print("Rockfall Prediction Dataset Created Successfully!")
print(f"Dataset shape: {df.shape}")
print("\nFeature summary:")
print(df.describe())

print("\nRockfall Risk Distribution:")
print(df['rockfall_risk'].value_counts())

# Save the dataset
df.to_csv('rockfall_prediction_dataset.csv', index=False)
print("\nDataset saved as 'rockfall_prediction_dataset.csv'")

# Adjust the risk calculation to create a more balanced dataset
# Recalculate risk score with adjusted weights and thresholds

# Create more realistic risk score with higher variance
risk_score_adjusted = (
    (100 - df['rock_mass_rating']) * 0.4 +
    (df['slope_angle_deg'] - 25) * 0.8 +  # Increased weight
    (df['slope_height_m'] / 100) * 50 +    # Increased impact
    (df['weathering_grade'] - 1) * 15 +
    np.log1p(df['water_impact_factor']) * 5 +
    np.log1p(df['block_volume_m3']) * 8 +
    (df['vibration_intensity_mm_s'] / 5) +
    ((100 - df['geological_strength_index']) * 0.3)
)

# Adjust thresholds for more realistic distribution
df['rockfall_risk'] = pd.cut(risk_score_adjusted, 
                            bins=[0, 60, 90, 200], 
                            labels=['Low', 'Medium', 'High'],
                            include_lowest=True)

# Create binary classification for high-risk scenarios
df['high_risk'] = (df['rockfall_risk'] == 'High').astype(int)

print("Updated Rockfall Risk Distribution:")
print(df['rockfall_risk'].value_counts())
print(f"\nPercentage distribution:")
for risk_level in df['rockfall_risk'].value_counts().index:
    pct = df['rockfall_risk'].value_counts()[risk_level] / len(df) * 100
    print(f"{risk_level}: {pct:.1f}%")

# Basic ML model demonstration
# Select key features for the ML model
feature_columns = [
    'rock_mass_rating', 'geological_strength_index', 'slope_height_m', 
    'slope_angle_deg', 'weathering_grade', 'rainfall_mm_annual',
    'joint_spacing_m', 'friction_angle_deg', 'cohesion_kpa',
    'pore_pressure_kpa', 'vibration_intensity_mm_s', 'block_volume_m3'
]

X = df[feature_columns]
y = df['high_risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest Model Accuracy: {accuracy:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save updated dataset
df.to_csv('rockfall_prediction_dataset_balanced.csv', index=False)
print("\nUpdated dataset saved as 'rockfall_prediction_dataset_balanced.csv'")

