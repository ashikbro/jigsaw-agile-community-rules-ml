# Integration Guide: Adding Missing Visualizations to EDA Notebook

This guide explains how to integrate the missing visualizations and model details into the `comprehensive_eda_analysis.ipynb` notebook.

## 📁 Files Created

1. **`detailed_model_evaluation.py`** - Contains confusion matrices, ROC curves, feature importance, and target distribution
2. **`additional_visualizations.py`** - Contains model predictions analysis, text length analysis, and enhanced correlation analysis
3. **`missing_visualizations.py`** - Original comprehensive visualization functions

## 🔧 Integration Steps

### Step 1: Add Required Imports
Ensure these imports are present in the notebook's import section:

```python
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score
import time
```

### Step 2: Add New Code Cells

Add the following code cells **after the existing model comparison section** (around line 1800 in the notebook):

#### Cell 1: Confusion Matrices and Classification Reports
```python
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
else:
    print('❌ No target variable or data available for detailed evaluation')
```

#### Cell 2: ROC Curves
```python
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
```

#### Cell 3: Feature Importance Analysis
```python
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
                axes[i].set_title(f'{model_name}\nTop {top_n} Feature Importances', fontweight='bold')
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
```

#### Cell 4: Target Distribution Analysis
```python
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
    print(f'\nImbalance ratio: {imbalance_ratio:.2f}:1')
    
    if imbalance_ratio > 2:
        print('⚠️ Dataset shows class imbalance - consider using balancing techniques')
    else:
        print('✅ Dataset is relatively balanced')
else:
    print('❌ No target variable available for distribution analysis')
```

#### Cell 5: Model Predictions Analysis
```python
# Model Predictions Analysis - Correct vs Incorrect Classifications
if 'detailed_results' in locals() and 'X_test' in locals():
    print('=' * 80)
    print('MODEL PREDICTIONS ANALYSIS')
    print('=' * 80)
    
    # Get the best model
    best_model_name = top_3_models[0]
    best_model_results = detailed_results[best_model_name]
    y_pred_best = best_model_results['y_pred']
    
    # Analyze predictions
    correct_predictions = (y_test == y_pred_best)
    incorrect_predictions = ~correct_predictions
    
    print(f'\nPrediction Analysis for {best_model_name}:')
    print(f'Correct predictions: {correct_predictions.sum()} ({correct_predictions.mean()*100:.2f}%)')
    print(f'Incorrect predictions: {incorrect_predictions.sum()} ({incorrect_predictions.mean()*100:.2f}%)')
    
    # Visualize prediction accuracy by class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Prediction accuracy by true class
    axes[0,0].hist([y_test[correct_predictions], y_test[incorrect_predictions]], 
                   bins=np.arange(len(np.unique(y_test))+1)-0.5, 
                   label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    axes[0,0].set_title('Prediction Accuracy by True Class', fontweight='bold')
    axes[0,0].set_xlabel('True Class')
    axes[0,0].set_ylabel('Count')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Prediction accuracy by predicted class
    axes[0,1].hist([y_pred_best[correct_predictions], y_pred_best[incorrect_predictions]], 
                   bins=np.arange(len(np.unique(y_pred_best))+1)-0.5, 
                   label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
    axes[0,1].set_title('Prediction Accuracy by Predicted Class', fontweight='bold')
    axes[0,1].set_xlabel('Predicted Class')
    axes[0,1].set_ylabel('Count')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Confidence distribution for correct vs incorrect predictions
    if best_model_results['y_pred_proba'] is not None:
        y_pred_proba_best = best_model_results['y_pred_proba']
        confidence_correct = np.max(y_pred_proba_best[correct_predictions], axis=1)
        confidence_incorrect = np.max(y_pred_proba_best[incorrect_predictions], axis=1)
        
        axes[1,0].hist([confidence_correct, confidence_incorrect], 
                       bins=20, alpha=0.7, label=['Correct', 'Incorrect'], 
                       color=['green', 'red'])
        axes[1,0].set_title('Prediction Confidence Distribution', fontweight='bold')
        axes[1,0].set_xlabel('Confidence (Max Probability)')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        print(f'\nConfidence Analysis:')
        print(f'Average confidence for correct predictions: {confidence_correct.mean():.4f}')
        print(f'Average confidence for incorrect predictions: {confidence_incorrect.mean():.4f}')
    
    # 4. Confusion matrix heatmap (detailed)
    cm_detailed = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm_detailed, annot=True, fmt='d', cmap='RdYlBu_r', ax=axes[1,1])
    axes[1,1].set_title(f'{best_model_name} - Detailed Confusion Matrix', fontweight='bold')
    axes[1,1].set_xlabel('Predicted')
    axes[1,1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    print('✅ Model predictions analysis completed!')
else:
    print('❌ No detailed results available for predictions analysis')
```

#### Cell 6: Text Length Analysis
```python
# Text Length Analysis by Toxicity Labels
if 'df' in locals() and 'comment_text' in df.columns:
    print('=' * 80)
    print('TEXT LENGTH ANALYSIS BY TOXICITY LABELS')
    print('=' * 80)
    
    # Calculate text lengths
    df['text_length'] = df['comment_text'].str.len()
    df['word_count'] = df['comment_text'].str.split().str.len()
    df['char_count'] = df['comment_text'].str.len()
    
    # Create binary toxicity label for analysis
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_toxicity_cols = [col for col in toxicity_columns if col in df.columns]
    
    if available_toxicity_cols:
        df['is_toxic'] = df[available_toxicity_cols].max(axis=1)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Text length distribution by toxicity
        axes[0,0].hist([df[df['is_toxic']==0]['text_length'], 
                       df[df['is_toxic']==1]['text_length']], 
                      bins=50, alpha=0.7, label=['Non-toxic', 'Toxic'], 
                      color=['blue', 'red'])
        axes[0,0].set_title('Text Length Distribution by Toxicity', fontweight='bold')
        axes[0,0].set_xlabel('Text Length (characters)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Word count distribution by toxicity
        axes[0,1].hist([df[df['is_toxic']==0]['word_count'], 
                       df[df['is_toxic']==1]['word_count']], 
                      bins=50, alpha=0.7, label=['Non-toxic', 'Toxic'], 
                      color=['blue', 'red'])
        axes[0,1].set_title('Word Count Distribution by Toxicity', fontweight='bold')
        axes[0,1].set_xlabel('Word Count')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Box plot of text length by toxicity
        df_plot = df[['text_length', 'is_toxic']].copy()
        df_plot['label'] = df_plot['is_toxic'].map({0: 'Non-toxic', 1: 'Toxic'})
        sns.boxplot(data=df_plot, x='label', y='text_length', ax=axes[0,2])
        axes[0,2].set_title('Text Length Box Plot by Toxicity', fontweight='bold')
        axes[0,2].set_ylabel('Text Length (characters)')
        
        # 4. Violin plot of word count by toxicity
        df_plot['word_count'] = df['word_count']
        sns.violinplot(data=df_plot, x='label', y='word_count', ax=axes[1,0])
        axes[1,0].set_title('Word Count Violin Plot by Toxicity', fontweight='bold')
        axes[1,0].set_ylabel('Word Count')
        
        # 5. Text length statistics by individual toxicity types
        if len(available_toxicity_cols) > 1:
            text_stats = []
            for col in available_toxicity_cols:
                toxic_texts = df[df[col] == 1]['text_length']
                non_toxic_texts = df[df[col] == 0]['text_length']
                
                text_stats.append({
                    'Toxicity_Type': col,
                    'Toxic_Mean_Length': toxic_texts.mean(),
                    'Non_Toxic_Mean_Length': non_toxic_texts.mean(),
                    'Difference': toxic_texts.mean() - non_toxic_texts.mean()
                })
            
            stats_df = pd.DataFrame(text_stats)
            
            axes[1,1].bar(stats_df['Toxicity_Type'], stats_df['Toxic_Mean_Length'], 
                         alpha=0.7, label='Toxic', color='red')
            axes[1,1].bar(stats_df['Toxicity_Type'], stats_df['Non_Toxic_Mean_Length'], 
                         alpha=0.7, label='Non-toxic', color='blue')
            axes[1,1].set_title('Average Text Length by Toxicity Type', fontweight='bold')
            axes[1,1].set_ylabel('Average Text Length')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Character distribution analysis
        df['special_char_count'] = df['comment_text'].str.count(r'[!@#$%^&*(),.?":{}|<>]')
        df['uppercase_ratio'] = df['comment_text'].str.count(r'[A-Z]') / df['text_length']
        
        axes[1,2].scatter(df[df['is_toxic']==0]['special_char_count'], 
                         df[df['is_toxic']==0]['uppercase_ratio'], 
                         alpha=0.5, label='Non-toxic', color='blue', s=10)
        axes[1,2].scatter(df[df['is_toxic']==1]['special_char_count'], 
                         df[df['is_toxic']==1]['uppercase_ratio'], 
                         alpha=0.5, label='Toxic', color='red', s=10)
        axes[1,2].set_title('Special Characters vs Uppercase Ratio', fontweight='bold')
        axes[1,2].set_xlabel('Special Character Count')
        axes[1,2].set_ylabel('Uppercase Ratio')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print('\n📊 Text Length Statistics:')
        print(f'Non-toxic comments - Mean length: {df[df["is_toxic"]==0]["text_length"].mean():.2f} characters')
        print(f'Toxic comments - Mean length: {df[df["is_toxic"]==1]["text_length"].mean():.2f} characters')
        print(f'Non-toxic comments - Mean word count: {df[df["is_toxic"]==0]["word_count"].mean():.2f} words')
        print(f'Toxic comments - Mean word count: {df[df["is_toxic"]==1]["word_count"].mean():.2f} words')
        
        print('\n📈 Character Analysis:')
        print(f'Non-toxic - Average special characters: {df[df["is_toxic"]==0]["special_char_count"].mean():.2f}')
        print(f'Toxic - Average special characters: {df[df["is_toxic"]==1]["special_char_count"].mean():.2f}')
        print(f'Non-toxic - Average uppercase ratio: {df[df["is_toxic"]==0]["uppercase_ratio"].mean():.4f}')
        print(f'Toxic - Average uppercase ratio: {df[df["is_toxic"]==1]["uppercase_ratio"].mean():.4f}')
        
        print('✅ Text length analysis completed!')
    else:
        print('⚠️ No toxicity columns found for text length analysis')
else:
    print('❌ No comment text data available for text length analysis')
```

## 📊 What These Visualizations Add

### 1. **Confusion Matrices**
- Visual representation of model performance
- Shows true positives, false positives, true negatives, false negatives
- Helps identify which classes are being confused

### 2. **ROC Curves**
- Performance comparison across different thresholds
- AUC scores for model comparison
- Useful for binary classification problems

### 3. **Feature Importance**
- Shows which features contribute most to predictions
- Helps with feature selection and model interpretation
- Available for tree-based models

### 4. **Target Distribution Analysis**
- Visualizes class imbalance
- Shows distribution in train/test splits
- Provides statistical insights

### 5. **Model Predictions Analysis**
- Analyzes correct vs incorrect predictions
- Shows confidence distributions
- Helps identify model weaknesses

### 6. **Text Length Analysis**
- Analyzes text characteristics by toxicity
- Shows patterns in toxic vs non-toxic comments
- Includes character-level analysis

## 🚀 Next Steps

1. **Copy and paste each code cell** into the notebook after the model comparison section
2. **Run the cells sequentially** to generate all visualizations
3. **Customize the visualizations** based on your specific dataset characteristics
4. **Add markdown cells** with explanations for each visualization section

## 📝 Notes

- All code is designed to handle missing data gracefully
- Visualizations will adapt to the actual data structure
- Error handling is included for edge cases
- The code follows the existing notebook's style and conventions

---

**✅ After integration, your EDA notebook will have comprehensive visualizations and detailed model analysis!**