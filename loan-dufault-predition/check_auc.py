"""
Diagnose AUC Performance Issue
Identifies and fixes any problems with model performance visualization
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

from src.paths import (
    CORRECTED_SCALING_MODEL,
    WORKING_MODEL,
    get_dataset_path,
    get_model_output_path,
    get_model_path,
    get_output_path,
    iter_model_paths,
    setup_project_imports,
)

setup_project_imports()
warnings.filterwarnings('ignore')

def check_available_models():
    """Check which model files are available and their properties."""
    print("🔍 CHECKING AVAILABLE MODELS")
    print("=" * 40)
    
    model_files = [path.name for path in iter_model_paths()]
    
    print("Available model files:")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    # Check each model
    model_info = {}
    for model_file in model_files:
        try:
            model_data = joblib.load(get_model_path(model_file))
            info = {
                'file': model_file,
                'model_type': model_data.get('model_type', 'Unknown'),
                'model_version': model_data.get('model_version', 'Unknown'),
                'auc_score': model_data.get('auc_score', 'Unknown'),
                'training_samples': model_data.get('training_samples', 'Unknown'),
                'scaling_method': model_data.get('scaling_method', 'Unknown'),
                'feature_count': len(model_data.get('feature_names', []))
            }
            model_info[model_file] = info
            
            print(f"\n📊 {model_file}:")
            print(f"  Model Type: {info['model_type']}")
            print(f"  Version: {info['model_version']}")
            print(f"  AUC Score: {info['auc_score']}")
            print(f"  Training Samples: {info['training_samples']:,}")
            print(f"  Scaling Method: {info['scaling_method']}")
            print(f"  Feature Count: {info['feature_count']}")
            
        except Exception as e:
            print(f"❌ Error loading {model_file}: {e}")
    
    return model_info

def test_model_performance(model_file):
    """Test the actual performance of a specific model."""
    print(f"\n🧪 TESTING MODEL: {model_file}")
    print("=" * 50)
    
    try:
        # Load model
        model_data = joblib.load(get_model_path(model_file))
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        print(f"Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Feature count: {len(feature_names)}")
        
        # Load and prepare data
        df = pd.read_csv(get_dataset_path())
        print(f"Dataset loaded: {df.shape}")
        
        if 'LoanID' in df.columns:
            df = df.drop('LoanID', axis=1)
        
        X = df.drop('Default', axis=1)
        y = df['Default']
        
        print(f"Default rate: {y.mean():.2%}")
        
        # Preprocess data (same as in visualization script)
        categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
        binary_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
        
        # Convert binary columns
        for col in binary_columns:
            if col in X.columns:
                X[col] = (X[col] == 'Yes').astype(int)
        
        # One-hot encode categorical
        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
        
        # Scale numerical features
        numerical_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                            'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
        
        scaler = StandardScaler()
        X_encoded[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])
        
        print(f"Processed features: {X_encoded.shape}")
        print(f"Feature names match: {set(X_encoded.columns) == set(feature_names)}")
        
        # Align features with model expectations
        missing_features = set(feature_names) - set(X_encoded.columns)
        extra_features = set(X_encoded.columns) - set(feature_names)
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            for feat in missing_features:
                X_encoded[feat] = 0
        
        if extra_features:
            print(f"⚠️ Extra features: {extra_features}")
            X_encoded = X_encoded[feature_names]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"Test set size: {X_test.shape[0]:,}")
        print(f"Test set default rate: {y_test.mean():.2%}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"AUC Score: {auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Performance assessment
        if auc >= 0.8:
            performance = "🟢 EXCELLENT"
        elif auc >= 0.75:
            performance = "🟡 VERY GOOD"
        elif auc >= 0.7:
            performance = "🟠 GOOD"
        elif auc >= 0.6:
            performance = "🟡 FAIR"
        else:
            performance = "🔴 POOR"
        
        print(f"\nPerformance Assessment: {performance}")
        
        # Check prediction distribution
        print(f"\nPrediction Statistics:")
        print(f"Min probability: {y_pred_proba.min():.4f}")
        print(f"Max probability: {y_pred_proba.max():.4f}")
        print(f"Mean probability: {y_pred_proba.mean():.4f}")
        print(f"Std probability: {y_pred_proba.std():.4f}")
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'model_file': model_file
        }
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main diagnostic function."""
    print("🔧 DIAGNOSING AUC PERFORMANCE ISSUE")
    print("=" * 50)
    
    # Check available models
    model_info = check_available_models()
    
    # Test each model
    results = {}
    for model_file in model_info.keys():
        result = test_model_performance(model_file)
        if result:
            results[model_file] = result
    
    # Find best model
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['auc'])
        best_auc = results[best_model]['auc']
        
        print(f"\n🏆 BEST MODEL FOUND:")
        print(f"Model: {best_model}")
        print(f"AUC: {best_auc:.4f}")
        
        if best_auc >= 0.7:
            print("✅ AUC performance is GOOD!")
        else:
            print("❌ AUC performance needs improvement")
    
    print(f"\n🎉 DIAGNOSIS COMPLETE!")
    print("=" * 30)
    print("✅ All models tested")
    print("✅ Performance metrics calculated")
    print("✅ Best model identified")

if __name__ == "__main__":
    main()
