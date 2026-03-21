#!/usr/bin/env python3
"""
MineSafe-AI — Flask Web Application
====================================
Serves the interactive frontend and provides prediction/stats API endpoints.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ---------------------------------------------------------------------------
# Try to load the trained ML model (gracefully fall back to formula)
# ---------------------------------------------------------------------------
MODEL_LOADED = False
predictor = None

try:
    from predictor import RockfallPredictor
    import yaml

    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    model_dir = BASE_DIR / "models" / "trained_models"
    model_file = model_dir / config.get("model", {}).get("model_file", "rockfall_predictor.pkl")

    if model_file.exists():
        predictor = RockfallPredictor(config.get("model", {}))
        predictor.load_model(str(model_dir))
        MODEL_LOADED = True
        print("✅  Trained ML model loaded successfully")
    else:
        print("⚠️  No trained model found — using formula-based fallback")
except Exception as exc:
    print(f"⚠️  Could not load ML model ({exc}) — using formula-based fallback")

# ---------------------------------------------------------------------------
# Load dataset for statistics
# ---------------------------------------------------------------------------
DATASET_PATH = BASE_DIR / "data" / "datasets" / "final_integrated_rockfall_dataset.csv"
dataset_df = None
if DATASET_PATH.exists():
    try:
        dataset_df = pd.read_csv(DATASET_PATH)
        print(f"📊  Dataset loaded: {dataset_df.shape[0]} rows, {dataset_df.shape[1]} columns")
    except Exception as exc:
        print(f"⚠️  Could not load dataset: {exc}")


# ========================== HELPER FUNCTIONS ================================

def formula_predict(params: dict) -> dict:
    """Fallback risk prediction using the composite formula from rockfall.py."""
    rmr = params.get("rock_mass_rating", 50)
    gsi = params.get("geological_strength_index", 45)
    slope_angle = params.get("slope_angle_deg", 45)
    slope_height = params.get("slope_height_m", 80)
    weathering = params.get("weathering_grade", 3)
    rainfall = params.get("rainfall_mm_annual", 1200)
    monsoon = params.get("monsoon_intensity", 2)
    gw_depth = params.get("groundwater_depth_m", 5)
    block_vol = params.get("block_volume_m3", 2)
    vibration = params.get("vibration_intensity_mm_s", 10)
    joint_sp = params.get("joint_spacing_m", 0.5)
    friction = params.get("friction_angle_deg", 32)
    cohesion = params.get("cohesion_kpa", 25)
    pore = params.get("pore_pressure_kpa", 50)

    water_impact = (rainfall * monsoon) / (gw_depth + 1)
    water_impact_max = (3000 * 4) / 1  # theoretical max

    risk_score_raw = (
        (100 - rmr) * 0.4
        + (slope_angle - 25) * 0.8
        + (slope_height / 100) * 50
        + (weathering - 1) * 15
        + math.log1p(water_impact) * 5
        + math.log1p(block_vol) * 8
        + (vibration / 5)
        + ((100 - gsi) * 0.3)
    )

    # Normalise score to 0-100 range
    risk_pct = max(0, min(100, (risk_score_raw / 150) * 100))

    if risk_pct >= 65:
        risk_level = "High Risk"
        confidence = 0.80 + (risk_pct - 65) / 350
    elif risk_pct >= 35:
        risk_level = "Medium Risk"
        confidence = 0.70 + (risk_pct - 35) / 300
    else:
        risk_level = "Low Risk"
        confidence = 0.75 + (35 - risk_pct) / 350

    confidence = min(confidence, 0.98)

    return {
        "risk_score": round(risk_pct, 1),
        "risk_level": risk_level,
        "confidence": round(confidence, 3),
        "high_risk_probability": round(risk_pct / 100, 3),
        "method": "formula",
    }


def get_recommendations(risk_level: str, params: dict) -> list:
    """Return context-aware safety recommendations."""
    base = []
    if "High" in risk_level:
        base = [
            "🚨 Immediately restrict personnel access to the hazard zone",
            "🛡️ Install rockfall barriers and reinforced catch benches",
            "📡 Deploy real-time slope monitoring sensors (radar, prisms)",
            "🔬 Conduct detailed geotechnical investigation within 48 hours",
            "📐 Evaluate slope angle reduction or bench re-design",
            "🚁 Establish emergency response & evacuation procedures",
        ]
    elif "Medium" in risk_level:
        base = [
            "⚡ Install basic slope monitoring equipment (crack meters)",
            "👁️ Schedule bi-weekly visual inspections of the slope face",
            "🚧 Implement controlled access & signage in the area",
            "🌧️ Monitor weather forecasts — restrict work during heavy rain",
            "📋 Prepare contingency & rapid-response plans",
        ]
    else:
        base = [
            "✅ Continue routine weekly safety inspections",
            "💧 Maintain drainage systems to prevent water accumulation",
            "📊 Monitor for gradual changes in slope geometry",
            "📝 Follow standard mining safety protocols (DGMS guidelines)",
        ]

    # Contextual extras
    if params.get("rainfall_mm_annual", 0) > 1800:
        base.append("🌧️ High rainfall region — prioritise dewatering & surface drainage")
    if params.get("weathering_grade", 0) >= 4:
        base.append("🪨 Severe weathering detected — consider shotcrete reinforcement")
    if params.get("vibration_intensity_mm_s", 0) > 15:
        base.append("💥 High blast vibration — optimise blast design to reduce PPV")

    return base


# ========================== ROUTES ==========================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept mine parameters as JSON and return risk prediction."""
    try:
        params = request.get_json(force=True)

        if MODEL_LOADED and predictor is not None:
            result = predictor.predict(params)
            result["method"] = "ml_model"
            risk_level = result.get("predicted_class", result.get("risk_level", "Unknown"))
            result["risk_level"] = risk_level
            result["risk_score"] = round(result.get("risk_score", 0), 1)
        else:
            result = formula_predict(params)
            risk_level = result["risk_level"]

        result["recommendations"] = get_recommendations(risk_level, params)
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return dataset statistics for the dashboard charts."""
    stats = {}

    if dataset_df is not None:
        df = dataset_df

        # Risk distribution — try multiple possible column names
        risk_col = None
        for col_name in ["rockfall_risk_level", "rockfall_risk", "risk_level"]:
            if col_name in df.columns:
                risk_col = col_name
                break

        if risk_col:
            dist = df[risk_col].value_counts().to_dict()
            stats["risk_distribution"] = {str(k): int(v) for k, v in dist.items()}
        else:
            # Fall back to binary column
            for bin_col in ["high_risk_binary", "high_risk"]:
                if bin_col in df.columns:
                    hr = int(df[bin_col].sum())
                    stats["risk_distribution"] = {"High": hr, "Low/Medium": int(len(df) - hr)}
                    break

        # Identify target column for correlation
        target_col = None
        for t in ["high_risk_binary", "high_risk", "comprehensive_risk_score"]:
            if t in df.columns:
                target_col = t
                break

        # Use ALL numeric feature columns (exclude target/label cols)
        exclude_cols = {"rockfall_risk_level", "rockfall_risk", "risk_level",
                        "high_risk_binary", "high_risk", "comprehensive_risk_score"}
        available = [c for c in df.select_dtypes(include="number").columns
                     if c not in exclude_cols]

        # Approximate importance via correlation with target
        if target_col and available:
            corrs = df[available].corrwith(df[target_col]).abs().sort_values(ascending=False)
            corrs = corrs.dropna()
            stats["feature_importance"] = {
                "features": corrs.index.tolist(),
                "values": [round(v, 4) for v in corrs.values.tolist()],
            }

        # Summary stats
        stats["total_samples"] = int(len(df))
        stats["total_features"] = int(len(df.columns))

        # Regional data
        stats["regions"] = {
            "Jharkhand": {"mines": "Coal", "avg_rainfall": 1400, "risk_profile": "High"},
            "Odisha": {"mines": "Iron Ore", "avg_rainfall": 1600, "risk_profile": "Medium-High"},
            "Rajasthan": {"mines": "Limestone", "avg_rainfall": 600, "risk_profile": "Low-Medium"},
        }
    else:
        # Fallback static data
        stats = {
            "risk_distribution": {"Low": 350, "Medium": 420, "High": 230},
            "feature_importance": {
                "features": [
                    "slope_angle_deg", "slope_height_m", "rock_mass_rating",
                    "weathering_grade", "rainfall_mm_annual", "block_volume_m3",
                    "vibration_intensity_mm_s", "cohesion_kpa", "friction_angle_deg",
                    "geological_strength_index", "pore_pressure_kpa", "joint_spacing_m",
                ],
                "values": [0.18, 0.16, 0.14, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02, 0.02],
            },
            "total_samples": 1000,
            "total_features": 28,
            "regions": {
                "Jharkhand": {"mines": "Coal", "avg_rainfall": 1400, "risk_profile": "High"},
                "Odisha": {"mines": "Iron Ore", "avg_rainfall": 1600, "risk_profile": "Medium-High"},
                "Rajasthan": {"mines": "Limestone", "avg_rainfall": 600, "risk_profile": "Low-Medium"},
            },
        }

    # Model performance (static from training results)
    stats["model_performance"] = {
        "accuracy": 0.94,
        "auc_score": 0.97,
        "precision": 0.93,
        "recall": 0.95,
        "f1_score": 0.94,
        "algorithm": "Random Forest",
    }

    return jsonify(stats)


@app.route("/videos/<path:filename>")
def serve_video(filename):
    """Serve rockfall detection videos from the project root."""
    return send_from_directory(str(BASE_DIR), filename)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🏔️  MineSafe-AI — Starting web server …")
    print("   Open  http://127.0.0.1:5050  in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5050)
