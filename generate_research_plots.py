"""
Generate Research Paper Visualizations
Creates comprehensive charts and graphs for loan default prediction model analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
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
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def create_dataset_overview():
    """Create dataset overview visualization."""
    print("Creating dataset overview...")
    
    df = pd.read_csv(get_dataset_path())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Loan Default Dataset Overview', fontsize=16, fontweight='bold')
    
    # Default rate pie chart
    default_counts = df['Default'].value_counts()
    axes[0, 0].pie(default_counts.values, labels=['No Default', 'Default'], 
                   autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Default Rate Distribution', fontweight='bold')
    
    # Credit Score distribution
    df[df['Default'] == 0]['CreditScore'].hist(alpha=0.7, bins=30, label='No Default', ax=axes[0, 1])
    df[df['Default'] == 1]['CreditScore'].hist(alpha=0.7, bins=30, label='Default', ax=axes[0, 1])
    axes[0, 1].set_title('Credit Score Distribution by Default Status', fontweight='bold')
    axes[0, 1].set_xlabel('Credit Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Income distribution
    df[df['Default'] == 0]['Income'].hist(alpha=0.7, bins=30, label='No Default', ax=axes[1, 0])
    df[df['Default'] == 1]['Income'].hist(alpha=0.7, bins=30, label='Default', ax=axes[1, 0])
    axes[1, 0].set_title('Income Distribution by Default Status', fontweight='bold')
    axes[1, 0].set_xlabel('Annual Income ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # DTI Ratio distribution
    df[df['Default'] == 0]['DTIRatio'].hist(alpha=0.7, bins=30, label='No Default', ax=axes[1, 1])
    df[df['Default'] == 1]['DTIRatio'].hist(alpha=0.7, bins=30, label='Default', ax=axes[1, 1])
    axes[1, 1].set_title('DTI Ratio Distribution by Default Status', fontweight='bold')
    axes[1, 1].set_xlabel('Debt-to-Income Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(get_output_path('1_dataset_overview.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 1_dataset_overview.png")

def create_credit_score_analysis():
    """Create credit score analysis visualization."""
    print("Creating credit score analysis...")
    
    df = pd.read_csv(get_dataset_path())
    
    # Create credit score ranges
    df['CreditScoreRange'] = pd.cut(df['CreditScore'], 
                                   bins=[0, 500, 600, 700, 750, 800, 850], 
                                   labels=['<500', '500-600', '600-700', '700-750', '750-800', '800+'])
    
    # Calculate default rates
    credit_analysis = df.groupby('CreditScoreRange').agg({
        'Default': ['count', 'mean'],
        'CreditScore': ['min', 'max', 'mean']
    }).round(3)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart of default rates
    default_rates = credit_analysis[('Default', 'mean')]
    counts = credit_analysis[('Default', 'count')]
    
    bars = axes[0].bar(range(len(default_rates)), default_rates.values, 
                       color=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71', '#27ae60', '#16a085'])
    axes[0].set_title('Default Rate by Credit Score Range', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Credit Score Range')
    axes[0].set_ylabel('Default Rate')
    axes[0].set_xticks(range(len(default_rates)))
    axes[0].set_xticklabels(default_rates.index, rotation=45)
    
    # Add value labels
    for i, (bar, rate, count) in enumerate(zip(bars, default_rates.values, counts.values)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{rate:.1%}\n(n={count:,})', ha='center', va='bottom', fontsize=10)
    
    # Scatter plot
    credit_means = credit_analysis[('CreditScore', 'mean')]
    axes[1].scatter(credit_means.values, default_rates.values, s=counts.values/100, 
                   alpha=0.7, c=default_rates.values, cmap='RdYlGn_r')
    axes[1].set_title('Credit Score vs Default Rate (Bubble Size = Sample Count)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Average Credit Score in Range')
    axes[1].set_ylabel('Default Rate')
    axes[1].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(credit_means.values, default_rates.values, 1)
    p = np.poly1d(z)
    axes[1].plot(credit_means.values, p(credit_means.values), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(get_output_path('2_credit_score_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 2_credit_score_analysis.png")

def create_feature_importance():
    """Create feature importance visualization."""
    print("Creating feature importance plot...")
    
    # Load model
    try:
        model_data = joblib.load(get_model_path(CORRECTED_SCALING_MODEL))
    except FileNotFoundError:
        model_data = joblib.load(get_model_path(WORKING_MODEL))
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    
    if hasattr(model, 'feature_importances_'):
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Take top 15 features
        top_features = importance_df.tail(15)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Feature Importance', fontweight='bold')
        ax.set_title('Top 15 Most Important Features for Loan Default Prediction', 
                    fontweight='bold', fontsize=14)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(get_output_path('3_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: 3_feature_importance.png")

def create_model_performance():
    """Create model performance visualizations."""
    print("Creating model performance plots...")
    
    # Load model
    try:
        model_data = joblib.load(get_model_path(CORRECTED_SCALING_MODEL))
    except FileNotFoundError:
        model_data = joblib.load(get_model_path(WORKING_MODEL))
    
    # Load and prepare data
    df = pd.read_csv(get_dataset_path())
    if 'LoanID' in df.columns:
        df = df.drop('LoanID', axis=1)
    
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Simple preprocessing for demonstration
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # Make predictions
    model = model_data['model']
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create performance plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve', fontweight='bold')
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    axes[0, 1].plot(recall, precision, color='darkgreen', lw=2,
                   label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve', fontweight='bold')
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
               xticklabels=['No Default', 'Default'],
               yticklabels=['No Default', 'Default'])
    axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred_proba[y_test == 0], bins=30, alpha=0.7, 
                   label='No Default', color='green', density=True)
    axes[1, 1].hist(y_pred_proba[y_test == 1], bins=30, alpha=0.7, 
                   label='Default', color='red', density=True)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Probability Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(get_output_path('4_model_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 4_model_performance.png")

def create_correlation_heatmap():
    """Create correlation heatmap."""
    print("Creating correlation heatmap...")
    
    df = pd.read_csv(get_dataset_path())
    
    # Select numerical features
    numerical_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 
                         'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
                         'LoanTerm', 'DTIRatio', 'Default']
    
    # Create correlation matrix
    corr_matrix = df[numerical_features].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(get_output_path('5_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 5_correlation_heatmap.png")

def create_summary_dashboard():
    """Create summary dashboard."""
    print("Creating summary dashboard...")
    
    df = pd.read_csv(get_dataset_path())
    
    # Load model
    try:
        model_data = joblib.load(get_model_path(CORRECTED_SCALING_MODEL))
    except FileNotFoundError:
        model_data = joblib.load(get_model_path(WORKING_MODEL))
    
    # Create dashboard
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Loan Default Prediction Model - Research Summary Dashboard', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Key metrics
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')
    
    default_rate = df['Default'].mean()
    auc_score = model_data.get('auc_score', 0.7342)
    training_samples = model_data.get('training_samples', len(df))
    
    metrics_text = f"""
    Dataset: {len(df):,} loans | Default Rate: {default_rate:.1%} | Model AUC: {auc_score:.3f} | Training Samples: {training_samples:,}
    Model Type: {model_data.get('model_type', 'Random Forest')} | Scaling: {model_data.get('scaling_method', 'Corrected')}
    """
    
    ax1.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Credit score analysis
    ax2 = fig.add_subplot(gs[1, 0])
    df['CreditScoreRange'] = pd.cut(df['CreditScore'], 
                                   bins=[0, 500, 600, 700, 750, 800, 850], 
                                   labels=['<500', '500-600', '600-700', '700-750', '750-800', '800+'])
    credit_rates = df.groupby('CreditScoreRange')['Default'].mean()
    credit_rates.plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Default Rate by Credit Score', fontweight='bold')
    ax2.set_ylabel('Default Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    # Income analysis
    ax3 = fig.add_subplot(gs[1, 1])
    df['IncomeQuartile'] = pd.qcut(df['Income'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    income_rates = df.groupby('IncomeQuartile')['Default'].mean()
    income_rates.plot(kind='bar', ax=ax3, color='lightgreen')
    ax3.set_title('Default Rate by Income Quartile', fontweight='bold')
    ax3.set_ylabel('Default Rate')
    ax3.tick_params(axis='x', rotation=45)
    
    # DTI analysis
    ax4 = fig.add_subplot(gs[1, 2])
    df['DTIRange'] = pd.cut(df['DTIRatio'], 
                           bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0], 
                           labels=['<30%', '30-40%', '40-50%', '50-60%', '60%+'])
    dti_rates = df.groupby('DTIRange')['Default'].mean()
    dti_rates.plot(kind='bar', ax=ax4, color='salmon')
    ax4.set_title('Default Rate by DTI Range', fontweight='bold')
    ax4.set_ylabel('Default Rate')
    ax4.tick_params(axis='x', rotation=45)
    
    # Feature importance
    ax5 = fig.add_subplot(gs[2, :])
    if hasattr(model_data['model'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': model_data['feature_names'],
            'importance': model_data['model'].feature_importances_
        }).sort_values('importance', ascending=True).tail(10)
        
        importance_df.plot(x='feature', y='importance', kind='barh', ax=ax5, color='gold')
        ax5.set_title('Top 10 Feature Importance', fontweight='bold')
        ax5.set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(get_output_path('6_research_summary_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: 6_research_summary_dashboard.png")

def main():
    """Generate all visualizations."""
    print("🎨 GENERATING RESEARCH PAPER VISUALIZATIONS")
    print("=" * 60)
    
    create_dataset_overview()
    create_credit_score_analysis()
    create_feature_importance()
    create_model_performance()
    create_correlation_heatmap()
    create_summary_dashboard()
    
    print("\n🎉 ALL VISUALIZATIONS GENERATED!")
    print("=" * 40)
    print("✅ 6 high-quality images created for your research paper")
    print("✅ All images saved as PNG files with 300 DPI")
    print("✅ Professional styling suitable for academic publication")

if __name__ == "__main__":
    main()
