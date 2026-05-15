"""
Generate SHAP Value Distribution - 4 Features in One Figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.preprocessing import StandardScaler
import warnings

from src.paths import (
    CORRECTED_SCALING_MODEL,
    WORKING_MODEL,
    get_dataset_path,
    get_model_path,
    get_output_path,
    setup_project_imports,
)

setup_project_imports()
warnings.filterwarnings('ignore')

# Set professional styling
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

def load_and_prepare_data():
    """Load and preprocess data."""
    df = pd.read_csv(get_dataset_path())
    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)
    
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Preprocessing
    categorical_columns = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
    binary_columns = ['HasMortgage', 'HasDependents', 'HasCoSigner']
    
    for col in binary_columns:
        if col in X.columns:
            X[col] = (X[col] == 'Yes').astype(int)
    
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    numerical_columns = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 
                        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']
    
    scaler = StandardScaler()
    X_encoded[numerical_columns] = scaler.fit_transform(X_encoded[numerical_columns])
    
    return X_encoded, y

# Load model
print("Loading model...")
try:
    model_data = joblib.load(get_model_path(CORRECTED_SCALING_MODEL))
except Exception:
    try:
        model_data = joblib.load(get_model_path(WORKING_MODEL))
    except Exception:
        model_data = joblib.load(get_model_path('loan_default_model.pkl'))

model = model_data['model']
feature_names = model_data['feature_names']

# Load data
print("Loading data...")
X_encoded, y = load_and_prepare_data()

# Sample for SHAP - using smaller sample for faster computation
sample_size = min(200, len(X_encoded))
X_sample = X_encoded.sample(n=sample_size, random_state=42)

# Calculate SHAP values
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Create the plot
print("Creating visualization...")
features_to_plot = ['Income', 'CreditScore', 'DTIRatio', 'LoanAmount']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SHAP Value Distribution: Feature Value Impact on Default Probability', 
             fontsize=15, fontweight='bold', y=0.995)

axes = axes.flatten()

for idx, feature_name in enumerate(features_to_plot):
    try:
        feature_idx = list(feature_names).index(feature_name)
    except ValueError:
        print(f"Feature '{feature_name}' not found, skipping.")
        continue
    
    ax = axes[idx]
    
    # Get feature values and SHAP values
    feature_values = X_sample.iloc[:, feature_idx].values
    feature_shap_values = shap_values[:, feature_idx]
    
    # Create bins
    n_bins = 15
    value_bins = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
    bin_indices = np.digitize(feature_values, value_bins)
    
    # Calculate mean SHAP value for each bin
    bin_means = []
    bin_centers = []
    for i in range(1, len(value_bins)):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means.append(feature_shap_values[mask].mean())
            bin_centers.append((value_bins[i-1] + value_bins[i]) / 2)
    
    # Create bars with gradient color
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bin_centers)))
    bars = ax.barh(range(len(bin_means)), bin_means, color=colors, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ax.set_yticks([0, len(bin_means)-1])
    ax.set_yticklabels(['Low', 'High'], fontsize=10, fontweight='bold')
    ax.set_xlabel('SHAP Value (Impact on Default)', fontweight='bold', fontsize=10)
    ax.set_ylabel(f'{feature_name}', fontweight='bold', fontsize=11)
    ax.set_title(f'{feature_name}', fontweight='bold', fontsize=12, pad=10)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add arrow annotation
    if idx == 0:
        ax.annotate('', xy=(0, len(bin_means)), xytext=(0, -1),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.6))

plt.tight_layout()
shap_output = get_output_path('shap_feature_distributions.png')
plt.savefig(shap_output, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {shap_output}")
plt.show()

print("\n🎉 Visualization complete!")

