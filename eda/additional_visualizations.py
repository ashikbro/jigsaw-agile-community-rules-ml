# Additional Visualizations for EDA Notebook
# Model Predictions Analysis and Text Length Analysis

# Cell 5: Model Predictions Analysis
code_cell_5 = '''
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
    
    print(f'\\nPrediction Analysis for {best_model_name}:')
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
        
        print(f'\\nConfidence Analysis:')
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
'''

# Cell 6: Text Length Analysis
code_cell_6 = '''
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
        print('\\n📊 Text Length Statistics:')
        print(f'Non-toxic comments - Mean length: {df[df["is_toxic"]==0]["text_length"].mean():.2f} characters')
        print(f'Toxic comments - Mean length: {df[df["is_toxic"]==1]["text_length"].mean():.2f} characters')
        print(f'Non-toxic comments - Mean word count: {df[df["is_toxic"]==0]["word_count"].mean():.2f} words')
        print(f'Toxic comments - Mean word count: {df[df["is_toxic"]==1]["word_count"].mean():.2f} words')
        
        print('\\n📈 Character Analysis:')
        print(f'Non-toxic - Average special characters: {df[df["is_toxic"]==0]["special_char_count"].mean():.2f}')
        print(f'Toxic - Average special characters: {df[df["is_toxic"]==1]["special_char_count"].mean():.2f}')
        print(f'Non-toxic - Average uppercase ratio: {df[df["is_toxic"]==0]["uppercase_ratio"].mean():.4f}')
        print(f'Toxic - Average uppercase ratio: {df[df["is_toxic"]==1]["uppercase_ratio"].mean():.4f}')
        
        print('✅ Text length analysis completed!')
    else:
        print('⚠️ No toxicity columns found for text length analysis')
else:
    print('❌ No comment text data available for text length analysis')
'''

# Cell 7: Enhanced Correlation Analysis
code_cell_7 = '''
# Enhanced Correlation Analysis with Toxicity Features
if 'df' in locals():
    print('=' * 80)
    print('ENHANCED CORRELATION ANALYSIS')
    print('=' * 80)
    
    # Create additional text-based features for correlation analysis
    text_features = pd.DataFrame()
    
    if 'comment_text' in df.columns:
        text_features['text_length'] = df['comment_text'].str.len()
        text_features['word_count'] = df['comment_text'].str.split().str.len()
        text_features['special_char_count'] = df['comment_text'].str.count(r'[!@#$%^&*(),.?":{}|<>]')
        text_features['uppercase_ratio'] = df['comment_text'].str.count(r'[A-Z]') / text_features['text_length']
        text_features['exclamation_count'] = df['comment_text'].str.count('!')
        text_features['question_count'] = df['comment_text'].str.count('\\?')
        text_features['caps_words'] = df['comment_text'].str.count(r'\\b[A-Z]{2,}\\b')
    
    # Combine with toxicity labels
    toxicity_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    available_toxicity_cols = [col for col in toxicity_columns if col in df.columns]
    
    if available_toxicity_cols and not text_features.empty:
        # Create correlation matrix
        correlation_data = pd.concat([text_features, df[available_toxicity_cols]], axis=1)
        correlation_matrix = correlation_data.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        
        # Mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.3f')
        
        plt.title('Enhanced Correlation Matrix: Text Features vs Toxicity Labels', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # Find strongest correlations with toxicity
        print('\\n🔍 Strongest Correlations with Toxicity:')
        for tox_col in available_toxicity_cols:
            correlations = correlation_matrix[tox_col].drop(tox_col).abs().sort_values(ascending=False)
            print(f'\\n{tox_col.upper()}:')
            for feature, corr in correlations.head(3).items():
                direction = "positively" if correlation_matrix[tox_col][feature] > 0 else "negatively"
                print(f'  • {feature}: {corr:.3f} ({direction} correlated)')
        
        print('✅ Enhanced correlation analysis completed!')
    else:
        print('⚠️ Insufficient data for enhanced correlation analysis')
else:
    print('❌ No data available for correlation analysis')
'''

print("Additional visualization code cells created successfully!")
print("\nThese cells provide:")
print("1. Model predictions analysis with confidence distributions")
print("2. Text length analysis by toxicity labels")
print("3. Enhanced correlation analysis with text features")
print("\nAdd these as new code cells in the notebook after the detailed model evaluation section.")