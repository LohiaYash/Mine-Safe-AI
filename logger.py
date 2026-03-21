"""
Utility Functions for Rockfall Prediction System
===============================================
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

def setup_logging(config_path: str = "config.yaml"):
    """Setup logging configuration."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        log_config = config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file', 'logs/rockfall_prediction.log')

        # Create logs directory if it doesn't exist
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    except Exception as e:
        # Fallback logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Could not load logging configuration: {str(e)}")

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean input data."""
    validated_data = {}

    # Define validation rules
    validation_rules = {
        'rock_mass_rating': (0, 100),
        'geological_strength_index': (0, 100),
        'slope_height_m': (1, 500),
        'slope_angle_deg': (0, 90),
        'weathering_grade': (1, 5),
        'rainfall_mm_annual': (0, 10000),
        'joint_spacing_m': (0.01, 10),
        'friction_angle_deg': (0, 60),
        'cohesion_kpa': (0, 1000),
        'pore_pressure_kpa': (0, 1000),
        'vibration_intensity_mm_s': (0, 100),
        'block_volume_m3': (0.01, 1000)
    }

    for key, value in data.items():
        if key in validation_rules:
            min_val, max_val = validation_rules[key]
            validated_data[key] = np.clip(value, min_val, max_val)
        else:
            validated_data[key] = value

    return validated_data

def generate_risk_report(prediction_result: Dict[str, Any], mine_data: Dict[str, Any]) -> str:
    """Generate a comprehensive risk assessment report."""

    risk_level = prediction_result['predicted_class']
    risk_score = prediction_result['risk_score']
    confidence = prediction_result['confidence']

    report = f"""
    ROCKFALL RISK ASSESSMENT REPORT
    ===============================

    MINE PARAMETERS:
    - Slope Height: {mine_data.get('slope_height_m', 'N/A')} meters
    - Slope Angle: {mine_data.get('slope_angle_deg', 'N/A')} degrees  
    - Rock Mass Rating: {mine_data.get('rock_mass_rating', 'N/A')}
    - Annual Rainfall: {mine_data.get('rainfall_mm_annual', 'N/A')} mm

    RISK ASSESSMENT:
    - Risk Level: {risk_level}
    - Risk Score: {risk_score:.1f}%
    - Confidence: {confidence:.1%}

    RECOMMENDATIONS:
    """

    # Add recommendations based on risk level
    if 'High' in risk_level:
        report += """
    ⚠️  HIGH RISK - IMMEDIATE ACTION REQUIRED:
    1. Implement comprehensive slope monitoring system
    2. Install rockfall barriers and catch benches
    3. Restrict personnel access to hazardous areas
    4. Conduct detailed geotechnical investigation
    5. Consider slope angle reduction or height limitation
    6. Establish emergency response procedures
    """
    elif 'Medium' in risk_level:
        report += """
    ⚡ MEDIUM RISK - ENHANCED MONITORING RECOMMENDED:
    1. Install basic slope monitoring equipment
    2. Conduct regular visual inspections
    3. Implement controlled access procedures
    4. Monitor weather conditions closely
    5. Prepare contingency plans
    """
    else:
        report += """
    ✅ LOW RISK - STANDARD MONITORING SUFFICIENT:
    1. Continue routine safety inspections
    2. Maintain good drainage systems
    3. Monitor for any changes in conditions
    4. Follow standard mining safety protocols
    """

    return report
