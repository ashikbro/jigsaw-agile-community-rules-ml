# Missing Visualizations for EDA Notebook
# This script contains the confusion matrices, ROC curves, and other visual analysis
# that should be added to the comprehensive_eda_analysis.ipynb

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score

def add_confusion_matrices_section():
    """
    Code to add confusion matrices for top 3 models
    """
    code = '''
# Detailed Model Evaluation with Confusion Matrices and ROC Curves
if 'results_df' in locals() and not results_df.empty:
    print('=' * 80)
    print('DETAILED MODEL EVALUATION WITH CONFUSION MATRICES')
    print('=' * 80)
    
    # Get top 3 models for detailed evaluation
    top_3_models = results_df_sorted.head(3)['Model'].tolist()
    
    # Re-train top models for detailed evaluation
    detailed_results = {}
    
    for model_name in top_3_models:
        print(f'\nDetailed evaluation for {model_name}...')
        
        # Get the model
        model = models[model_name]
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Store for visualization
        detailed_results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    # Create confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, model_name in enumerate(top_3_models):
        y_pred = detailed_results[model_name]['y_pred']
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
        # Add accuracy on the plot
        accuracy = accuracy_score(y_test, y_pred)
        axes[i].text(0.5, -0.1, f'Accuracy: {accuracy:.4f}', 
                    transform=axes[i].transAxes, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Confusion Matrices for Top 3 Models', fontsize=16, fontweight='bold', y=1.05)
    plt.show()
    
    # Classification reports
    print('\n' + '=' * 80)
    print('DETAILED CLASSIFICATION REPORTS')
    print('=' * 80)
    
    for model_name in top_3_models:
        y_pred = detailed_results[model_name]['y_pred']
        print(f'\n📊 {model_name.upper()} - CLASSIFICATION REPORT:')
        print('-' * 50)
        print(classification_report(y_test, y_pred))
        print('-' * 50)
    
    # ROC Curves (for binary classification)
    if len(np.unique(y_test)) == 2:
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green']
        
        for i, model_name in enumerate(top_3_models):
            y_pred_proba = detailed_results[model_name]['y_pred_proba']
            
            if y_pred_proba is not None:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Top 3 Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    print('\n✅ Detailed model evaluation with confusion matrices completed!')
else:
    print('❌ No model results available for detailed evaluation')
'''
    return code

def add_feature_importance_section():
    """
    Code to add feature importance analysis
    """
    code = '''
# Feature Importance Analysis
if 'detailed_results' in locals() and detailed_results:
    print('=' * 80)
    print('FEATURE IMPORTANCE ANALYSIS')
    print('=' * 80)
    
    # Analyze feature importance for tree-based models
    tree_based_models = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
    
    importance_data = {}
    
    for model_name in top_3_models:
        if model_name in tree_based_models and model_name in detailed_results:
            model = detailed_results[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names (assuming X_final has column names)
                if hasattr(X_final, 'columns'):
                    feature_names = X_final.columns
                else:
                    feature_names = [f'Feature_{i}' for i in range(X_final.shape[1])]
                
                # Get importance scores
                importances = model.feature_importances_
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                importance_data[model_name] = importance_df
    
    # Plot feature importance
    if importance_data:
        n_models = len(importance_data)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(importance_data.items()):
            # Plot top 15 features
            top_features = importance_df.head(15)
            
            axes[i].barh(range(len(top_features)), top_features['Importance'], color='skyblue')
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['Feature'])
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'{model_name}\nTop 15 Features', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Invert y-axis to show most important at top
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.show()
        
        # Print top 10 features for each model
        print('\nTOP 10 MOST IMPORTANT FEATURES:')
        print('-' * 50)
        
        for model_name, importance_df in importance_data.items():
            print(f'\n🏆 {model_name.upper()}:')
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f'{i:2d}. {row["Feature"]:<25} {row["Importance"]:.4f}')
    
    print('\n✅ Feature importance analysis completed!')
else:
    print('❌ No detailed model results available for feature importance analysis')
'''
    return code

def add_additional_visualizations():
    """
    Code to add additional missing visualizations
    """
    code = '''
# Additional Essential Visualizations
print('=' * 80)
print('ADDITIONAL ESSENTIAL VISUALIZATIONS')
print('=' * 80)

# 1. Target Variable Distribution with Enhanced Visualization
if 'label' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    label_counts = df['label'].value_counts()
    axes[0].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
               colors=['lightblue', 'lightcoral'], startangle=90)
    axes[0].set_title('Target Variable Distribution (Pie Chart)', fontweight='bold')
    
    # Bar plot with percentages
    bars = axes[1].bar(label_counts.index, label_counts.values, 
                      color=['lightblue', 'lightcoral'], alpha=0.7)
    axes[1].set_title('Target Variable Distribution (Bar Chart)', fontweight='bold')
    axes[1].set_xlabel('Label')
    axes[1].set_ylabel('Count')
    
    # Add percentage labels on bars
    total = label_counts.sum()
    for bar, count in zip(bars, label_counts.values):
        percentage = (count / total) * 100
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f'\n📊 TARGET VARIABLE STATISTICS:')
    print(f'Total samples: {len(df)}')
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f'{label}: {count} samples ({percentage:.2f}%)')

# 2. Text Length Distribution Analysis
if 'text_length' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram
    axes[0,0].hist(df['text_length'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Text Length Distribution', fontweight='bold')
    axes[0,0].set_xlabel('Text Length (characters)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0,1].boxplot(df['text_length'], vert=True)
    axes[0,1].set_title('Text Length Box Plot', fontweight='bold')
    axes[0,1].set_ylabel('Text Length (characters)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Text length by label (if available)
    if 'label' in df.columns:
        for i, label in enumerate(df['label'].unique()):
            subset = df[df['label'] == label]['text_length']
            axes[1,0].hist(subset, bins=30, alpha=0.6, label=f'Label: {label}')
        axes[1,0].set_title('Text Length Distribution by Label', fontweight='bold')
        axes[1,0].set_xlabel('Text Length (characters)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Violin plot
        df_plot = df[['label', 'text_length']].copy()
        sns.violinplot(data=df_plot, x='label', y='text_length', ax=axes[1,1])
        axes[1,1].set_title('Text Length Distribution by Label (Violin Plot)', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'Label column not available', ha='center', va='center')
        axes[1,1].text(0.5, 0.5, 'Label column not available', ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

# 3. Correlation Heatmap for All Numerical Features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix - All Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print strong correlations
    print('\n🔗 STRONG CORRELATIONS (|r| > 0.7):')
    print('-' * 40)
    strong_corr_found = False
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                print(f'{correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr_val:.3f}')
                strong_corr_found = True
    
    if not strong_corr_found:
        print('No strong correlations found (threshold: |r| > 0.7)')

print('\n✅ Additional essential visualizations completed!')
'''
    return code

if __name__ == "__main__":
    print("This file contains the missing visualization code for the EDA notebook.")
    print("\nSections included:")
    print("1. Confusion Matrices and ROC Curves")
    print("2. Feature Importance Analysis")
    print("3. Additional Essential Visualizations")
    print("\nTo use: Copy the code from each function into your Jupyter notebook.")