#!/usr/bin/env python3
"""
Rockfall Prediction System for Indian Open Pit Mines
==================================================== 

Fixed main.py file with correct imports and path handling
"""

import argparse
import yaml
import logging
import sys
from pathlib import Path
import os

# Fix Python path issues
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))

# Now import our custom modules
try:
    from data_loader import DataLoader
    from predictor import RockfallPredictor
    print("✅ Modules imported successfully!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please check that all files are in the correct locations:")
    print("- src/data_loader.py")
    print("- src/predictor.py")
    print("- src/dashboard.py")
    print("- src/logger.py")
    sys.exit(1)

def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/rockfall_prediction.log')
        ]
    )

def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default configuration...")
        return {
            'data': {
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'algorithm': 'gradient_boosting',
                'save_path': 'models/trained_models/',
                'model_file': 'rockfall_predictor.pkl',
                'scaler_file': 'feature_scaler.pkl'
            }
        }

def check_file_structure():
    """Check if required files and folders exist."""
    required_files = [
        'src/data_loader.py',
        'src/predictor.py',
        'src/dashboard.py',
        'src/logger.py'
    ]
    
    required_dirs = [
        'src/',
        'data/datasets/',
        'models/trained_models/',
        'logs/'
    ]
    
    # Create missing directories
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check for missing files
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    return True

def main():
    """Main execution function."""
    print("🏔️  ROCKFALL PREDICTION SYSTEM FOR INDIAN OPEN PIT MINES")
    print("="*65)
    
    # Check file structure
    if not check_file_structure():
        print("\n❌ Please ensure all required files are in place before running.")
        return
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Rockfall Prediction System")
    parser.add_argument("--mode", choices=["train", "predict", "demo"], 
                       default="demo", help="Operation mode")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--input-file", help="Input CSV file for batch predictions")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded for mode: {args.mode}")
    
    if args.mode == "train":
        print("\n🔧 TRAINING MODE")
        print("="*20)
        
        try:
            # Check for dataset
            dataset_files = [
                "data/datasets/final_integrated_rockfall_dataset.csv",
                "data/datasets/integrated_real_rockfall_dataset.csv",
                "data/datasets/indian_rockfall_prediction_dataset.csv"
            ]
            
            dataset_path = None
            for path in dataset_files:
                if Path(path).exists():
                    dataset_path = path
                    break
            
            if not dataset_path:
                print("❌ No dataset found. Please add one of these files:")
                for path in dataset_files:
                    print(f"   - {path}")
                return
            
            print(f"📊 Using dataset: {dataset_path}")
            
            # Initialize data loader
            data_loader = DataLoader(config.get('data', {}))
            
            # Load and split data
            print("📥 Loading and preprocessing data...")
            X_train, X_test, y_train, y_test = data_loader.load_and_split(dataset_path)
            
            print(f"   Training samples: {len(X_train)}")
            print(f"   Testing samples: {len(X_test)}")
            print(f"   Features: {len(X_train.columns)}")
            
            # Initialize and train model
            predictor = RockfallPredictor(config.get('model', {}))
            print("🤖 Training machine learning model...")
            
            training_metrics = predictor.train(X_train, y_train)
            print(f"✅ Training completed!")
            print(f"   Training accuracy: {training_metrics.get('train_accuracy', 'N/A'):.3f}")
            
            # Evaluate model
            print("📈 Evaluating model performance...")
            eval_metrics = predictor.evaluate(X_test, y_test)
            print(f"✅ Evaluation completed!")
            print(f"   Test accuracy: {eval_metrics['accuracy']:.3f}")
            print(f"   AUC score: {eval_metrics.get('auc_score', 'N/A'):.3f}")
            
            # Save trained model
            save_path = config.get('model', {}).get('save_path', 'models/trained_models/')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            predictor.save_model(save_path)
            print(f"💾 Model saved to: {save_path}")
            
            # Show feature importance
            if hasattr(predictor, 'get_feature_importance'):
                importance = predictor.get_feature_importance()
                if not importance.empty:
                    print(f"\n🎯 Top 5 Most Important Features:")
                    for i, (_, row) in enumerate(importance.head().iterrows(), 1):
                        print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            logger.error(f"Training failed: {str(e)}")
    
    elif args.mode == "predict":
        print("\n🔮 PREDICTION MODE")
        print("="*20)
        
        try:
            # Load trained model
            predictor = RockfallPredictor(config.get('model', {}))
            model_path = config.get('model', {}).get('save_path', 'models/trained_models/')
            
            if not Path(model_path).exists():
                print("❌ No trained model found.")
                print("Please run training first: python main.py --mode train")
                return
            
            print("📥 Loading trained model...")
            predictor.load_model(model_path)
            print("✅ Model loaded successfully!")
            
            if args.input_file:
                # Batch predictions from file
                if not Path(args.input_file).exists():
                    print(f"❌ Input file not found: {args.input_file}")
                    return
                
                print(f"📊 Making predictions for: {args.input_file}")
                output_file = args.input_file.replace('.csv', '_predictions.csv')
                results_df = predictor.predict_from_file(args.input_file, output_file)
                
                print(f"✅ Predictions completed!")
                print(f"💾 Results saved to: {output_file}")
                
                # Show summary
                high_risk_count = (results_df['predicted_class'] == 'High Risk').sum()
                total_count = len(results_df)
                print(f"📈 Summary: {high_risk_count}/{total_count} samples classified as High Risk")
                
            else:
                # Interactive prediction
                print("🤖 Interactive Prediction Mode")
                print("-" * 35)
                result = predictor.interactive_predict()
                
            logger.info("Predictions completed successfully")
            
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            logger.error(f"Prediction failed: {str(e)}")
    
    elif args.mode == "demo":
        print("\n🚀 DEMO MODE")
        print("="*15)
        
        print("Welcome to the Rockfall Prediction System for Indian Open Pit Mines!")
        print()
        print("This system integrates data from multiple real Kaggle datasets:")
        print("✓ Slope Stability Analysis Dataset")
        print("✓ Indian Earthquakes Dataset (2018 onwards)")  
        print("✓ Comprehensive Soil Classification Dataset")
        print("✓ Multimodal Sensor Fusion Dataset")
        print()
        
        # Demo mining scenarios
        demo_scenarios = {
            "🏭 High Risk - Jharkhand Coal Mine (Monsoon Season)": {
                "description": "Steep slopes, high moisture, active seismic zone",
                "risk_factors": ["Slope: 55°", "Moisture: 34%", "Earthquake: M4.2", "Clay: 52%"]
            },
            "⛰️  Medium Risk - Odisha Iron Ore Mine": {
                "description": "Moderate conditions with controlled operations",  
                "risk_factors": ["Slope: 42°", "Moisture: 22%", "Earthquake: M3.2", "Clay: 28%"]
            },
            "🏗️  Low Risk - Rajasthan Limestone Mine": {
                "description": "Gentle slopes, dry conditions, stable geology",
                "risk_factors": ["Slope: 32°", "Moisture: 12%", "Earthquake: M2.5", "Clay: 15%"]
            }
        }
        
        print("📊 Demo Mining Scenarios:")
        for scenario, info in demo_scenarios.items():
            print(f"\n{scenario}")
            print(f"   {info['description']}")
            print(f"   Key factors: {', '.join(info['risk_factors'])}")
        
        print(f"\n📝 Available Commands:")
        print(f"   python main.py --mode train    # Train the ML model")
        print(f"   python main.py --mode predict  # Make predictions")
        print(f"   streamlit run src/dashboard.py # Launch web interface")
        
        print(f"\n🔧 System Status:")
        model_path = Path('models/trained_models/rockfall_predictor.pkl')
        if model_path.exists():
            print("   ✅ Trained model: Available")
        else:
            print("   ❌ Trained model: Not found (run training first)")
        
        dataset_found = False
        for path in ["data/datasets/final_integrated_rockfall_dataset.csv", 
                    "data/datasets/integrated_real_rockfall_dataset.csv"]:
            if Path(path).exists():
                print(f"   ✅ Dataset: {Path(path).name}")
                dataset_found = True
                break
        
        if not dataset_found:
            print("   ❌ Dataset: Not found")

if __name__ == "__main__":
    main()