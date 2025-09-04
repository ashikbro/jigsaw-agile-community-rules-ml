# Detailed Model Evaluation - Confusion Matrices and ROC Curves
# This code should be added as a new cell in the comprehensive_eda_analysis.ipynb

# Cell 1: Confusion Matrices for Top 3 Models
code_cell_1 = '''
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
        axes[i].set_title(f'{model_name}\\nConfusion Matrix', fontweight='bold')
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
    print('\\n' + '=' * 80)
    print('DETAILED CLASSIFICATION REPORTS')
    print('=' * 80)
    
    for model_name in top_3_models:
        y_pred = detailed_results[model_name]['y_pred']
        print(f'\\n📊 {model_name.upper()} - CLASSIFICATION REPORT:')
        print('-' * 50)
        print(classification_report(y_test, y_pred))
        print('-' * 50)
else:
    print('❌ No target variable or data available for detailed evaluation')
'''

# Cell 2: ROC Curves
code_cell_2 = '''
# ROC Curves for Binary Classification
if 'detailed_results' in locals() and len(np.unique(y_test)) == 2:
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
    plt.title('ROC Curves for Top 3 Models', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print('✅ ROC curves generated successfully!')
else:
    print('⚠️ ROC curves are only available for binary classification problems')
'''

# Cell 3: Feature Importance Analysis
code_cell_3 = '''
# Feature Importance Analysis for Tree-based Models
if 'detailed_results' in locals():
    tree_models = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
    available_tree_models = [model for model in tree_models if model in detailed_results.keys()]
    
    if available_tree_models:
        fig, axes = plt.subplots(1, len(available_tree_models), figsize=(6*len(available_tree_models), 8))
        if len(available_tree_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_tree_models):
            model = detailed_results[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                # Get feature importances
                importances = model.feature_importances_
                
                # Create feature names (assuming we have them)
                if hasattr(X_train, 'columns'):
                    feature_names = X_train.columns
                else:
                    feature_names = [f'Feature_{j}' for j in range(len(importances))]
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Plot top 15 features
                top_n = min(15, len(importances))
                
                axes[i].bar(range(top_n), importances[indices[:top_n]], color='skyblue')
                axes[i].set_title(f'{model_name}\\nTop {top_n} Feature Importances', fontweight='bold')
                axes[i].set_xlabel('Features')
                axes[i].set_ylabel('Importance')
                axes[i].set_xticks(range(top_n))
                axes[i].set_xticklabels([feature_names[i] for i in indices[:top_n]], 
                                       rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        print('✅ Feature importance analysis completed!')
    else:
        print('⚠️ No tree-based models available for feature importance analysis')
else:
    print('❌ No detailed results available for feature importance analysis')
'''

# Cell 4: Target Distribution Analysis
code_cell_4 = '''
# Target Variable Distribution Analysis
if 'y_final' in locals() and y_final is not None:
    plt.figure(figsize=(15, 10))
    
    # Overall target distribution
    plt.subplot(2, 3, 1)
    target_counts = pd.Series(y_final).value_counts().sort_index()
    plt.bar(target_counts.index, target_counts.values, color='lightblue')
    plt.title('Target Variable Distribution', fontweight='bold')
    plt.xlabel('Target Class')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(y_final)
    for i, v in enumerate(target_counts.values):
        plt.text(target_counts.index[i], v + total*0.01, f'{v/total*100:.1f}%', 
                ha='center', fontweight='bold')
    
    # Target distribution in train/test sets
    if 'y_train' in locals() and 'y_test' in locals():
        plt.subplot(2, 3, 2)
        train_counts = pd.Series(y_train).value_counts().sort_index()
        test_counts = pd.Series(y_test).value_counts().sort_index()
        
        x = np.arange(len(train_counts))
        width = 0.35
        
        plt.bar(x - width/2, train_counts.values, width, label='Train', color='lightgreen')
        plt.bar(x + width/2, test_counts.values, width, label='Test', color='lightcoral')
        
        plt.title('Train/Test Target Distribution', fontweight='bold')
        plt.xlabel('Target Class')
        plt.ylabel('Count')
        plt.xticks(x, train_counts.index)
        plt.legend()
    
    # Class imbalance ratio
    plt.subplot(2, 3, 3)
    class_ratios = target_counts / target_counts.sum()
    plt.pie(class_ratios.values, labels=class_ratios.index, autopct='%1.1f%%', 
           colors=['lightblue', 'lightcoral', 'lightgreen', 'gold'][:len(class_ratios)])
    plt.title('Class Distribution (Pie Chart)', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print('📊 Target Distribution Analysis:')
    print(f'Total samples: {len(y_final)}')
    print(f'Number of classes: {len(target_counts)}')
    print(f'Class distribution:')
    for class_label, count in target_counts.items():
        print(f'  Class {class_label}: {count} samples ({count/len(y_final)*100:.2f}%)')
    
    # Calculate imbalance ratio
    max_class = target_counts.max()
    min_class = target_counts.min()
    imbalance_ratio = max_class / min_class
    print(f'\\nImbalance ratio: {imbalance_ratio:.2f}:1')
    
    if imbalance_ratio > 2:
        print('⚠️ Dataset shows class imbalance - consider using balancing techniques')
    else:
        print('✅ Dataset is relatively balanced')
else:
    print('❌ No target variable available for distribution analysis')
'''

print("Detailed model evaluation code cells created successfully!")
print("\nTo integrate into the notebook:")
print("1. Add these as new code cells after the model comparison section")
print("2. Each code block should be in a separate cell")
print("3. Run the cells in sequence after the main model training")