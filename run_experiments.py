import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_predict, learning_curve, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None

import warnings
warnings.simplefilter(action='ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('llm_relabeled_data_large.csv') # Reading the new large dataset
    df = df.dropna(subset=['query', 'label'])
    
    # Map label to 0/1, 'small' -> 0, 'large' -> 1
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower().strip() == 'small' else 1)
    
    X = df['query'].values
    y = df['label_id'].values

    print("Data loaded. Class distribution:", {0: sum(y==0), 1: sum(y==1)})

    # Initialize TF-IDF + LR
    tfidf_lr = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    # BERT
    print("Loading SentenceTransformer model...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Encoding text. This may take a minute...")
    X_bert = bert_model.encode(X, show_progress_bar=False)
    clf_bert = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    # PCA Pipelines
    bert_pca_lr = Pipeline([
        ('pca', PCA(n_components=50, random_state=42)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])
    
    tfidf_pca_lr = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('pca', TruncatedSVD(n_components=50, random_state=42)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    latex_output = "\\section*{Experiment Quantitative Results}\\n\\n"

    # --- Experiment 1 ---
    print("\n=== Experiment 1: Baseline ===")
    cv_tfidf = cross_validate(tfidf_lr, X, y, cv=5, scoring=scoring)
    cv_bert = cross_validate(clf_bert, X_bert, y, cv=5, scoring=scoring)
    
    # Store into DataFrame for tracking
    exp1_data = {
        'Model': ['TF-IDF + LR', 'BERT + LR'],
        'Accuracy': [np.mean(cv_tfidf['test_accuracy']), np.mean(cv_bert['test_accuracy'])],
        'Precision': [np.mean(cv_tfidf['test_precision']), np.mean(cv_bert['test_precision'])],
        'Recall': [np.mean(cv_tfidf['test_recall']), np.mean(cv_bert['test_recall'])],
        'F1-Score': [np.mean(cv_tfidf['test_f1']), np.mean(cv_bert['test_f1'])],
        'ROC AUC': [np.mean(cv_tfidf['test_roc_auc']), np.mean(cv_bert['test_roc_auc'])]
    }
    df_exp1 = pd.DataFrame(exp1_data)
    df_exp1.to_csv('experiment_1_metrics.csv', index=False)
    
    # Save LaTeX table for Exp 1
    latex_output += "%% Experiment 1 Base Metrics\n"
    latex_output += df_exp1.to_latex(index=False, float_format="%.4f", caption="Baseline Classification Metrics", label="tab:baseline")
    latex_output += "\n\n"

    y_pred_tfidf = cross_val_predict(tfidf_lr, X, y, cv=5)
    y_pred_bert = cross_val_predict(clf_bert, X_bert, y, cv=5)

    cm_tfidf = confusion_matrix(y, y_pred_tfidf)
    cm_bert = confusion_matrix(y, y_pred_bert)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Small-LLM (0)', 'Large-LLM (1)'], yticklabels=['Small-LLM (0)', 'Large-LLM (1)'])
    axes[0].set_title('TF-IDF + LR (Balanced)')
    
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Small-LLM (0)', 'Large-LLM (1)'], yticklabels=['Small-LLM (0)', 'Large-LLM (1)'])
    axes[1].set_title('BERT + LR (Balanced)')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)

    # --- Experiment 2 ---
    print("\n=== Experiment 2: Dataset Size & PCA (Learning Curves) ===")
    train_sizes_scale = np.linspace(0.1, 1.0, 10)
    
    models_to_test = {
        'BERT': clf_bert,
        'TF-IDF': tfidf_lr,
        'BERT + PCA': bert_pca_lr,
        'TF-IDF + PCA': tfidf_pca_lr
    }
    
    exp2_data = {'Train_Size_Pct': train_sizes_scale}
    plt.figure(figsize=(10, 7))
    colors = {'BERT': 'g', 'TF-IDF': 'b', 'BERT + PCA': 'darkgreen', 'TF-IDF + PCA': 'navy'}
    linestyles = {'BERT': '-', 'TF-IDF': '-', 'BERT + PCA': '--', 'TF-IDF + PCA': '--'}
    
    for name, model in models_to_test.items():
        features = X_bert if 'BERT' in name else X
        t_sizes, _, te_scores = learning_curve(
            model, features, y, cv=5, n_jobs=-1, 
            train_sizes=train_sizes_scale, scoring='f1'
        )
        te_mean = np.mean(te_scores, axis=1)
        # Store for CSV logging
        exp2_data[f"{name}_F1"] = te_mean 
        
        plt.plot(t_sizes, te_mean, marker='o', color=colors[name], linestyle=linestyles[name], label=f"{name} F1-Score")
        
    df_exp2 = pd.DataFrame(exp2_data)
    df_exp2.to_csv('experiment_2_learning_curves.csv', index=False)
    
    latex_output += "%% Experiment 2 Learning Curves\n"
    latex_output += df_exp2.to_latex(index=False, float_format="%.4f", caption="F1-Score over Dataset Size", label="tab:learning_curve")
    latex_output += "\n\n"

    plt.xlabel("Training examples")
    plt.ylabel("Cross-Validated F1 Score")
    plt.legend(loc="lower right")
    plt.title("Learning Curves (Granular Sampling w/ PCA Comparisons)")
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)

    # --- Experiment 3 ---
    print("\n=== Experiment 3: Class Imbalance Learning Curves ===")
    if SMOTE is None:
        print("imbalanced-learn not installed, skipping Exp 3.")
    else:
        idx_small = np.where(y == 0)[0] # 202 samples
        idx_large = np.where(y == 1)[0] # 598 samples
        np.random.seed(42)
        
        # To fix the NaN sampling issue, we scale up the total capacity
        # We use all 598 "large" samples as the majority class.
        # We sample 150 "small" samples as the minority class (~80/20 split).
        idx_small_sub = np.random.choice(idx_small, size=min(150, len(idx_small)), replace=False)
        idx_imbalanced = np.concatenate((idx_large, idx_small_sub))
        np.random.shuffle(idx_imbalanced)
        
        X_imb = X_bert[idx_imbalanced]
        # Invert the original arrays (1->0, 0->1) inside this experiment 
        # so that the 'small' class (the minority) correctly registers as 'Target=1' for sklearn's F1 calc.
        y_imb = 1 - y[idx_imbalanced]
        
        pipeline_unbal = Pipeline([('clf', LogisticRegression(random_state=42, max_iter=1000))])
        pipeline_cw = Pipeline([('clf', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))])
        pipeline_smote = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=3)),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        imb_models = {
            'No Resampling': pipeline_unbal,
            'Class Weights': pipeline_cw,
            'SMOTE': pipeline_smote
        }
        
        train_sizes_imb = np.linspace(0.4, 1.0, 7)
        exp3_data = {'Train_Size_Pct': train_sizes_imb}
        plt.figure(figsize=(9, 6))
        colors_imb = {'No Resampling': 'red', 'Class Weights': 'blue', 'SMOTE': 'green'}
        
        for name, model in imb_models.items():
            t_sizes, _, te_scores = learning_curve(
                model, X_imb, y_imb, cv=5, n_jobs=-1, 
                train_sizes=train_sizes_imb, scoring='f1'
            )
            te_mean = np.mean(te_scores, axis=1)
            te_std = np.std(te_scores, axis=1)
            exp3_data[f"{name}_F1"] = te_mean 
            
            plt.plot(t_sizes, te_mean, marker='o', color=colors_imb[name], label=name)
            plt.fill_between(t_sizes, te_mean - te_std, te_mean + te_std, alpha=0.1, color=colors_imb[name])
            
        df_exp3 = pd.DataFrame(exp3_data)
        df_exp3.to_csv('experiment_3_imbalance.csv', index=False)
        
        latex_output += "%% Experiment 3 Imbalance Techniques\n"
        latex_output += df_exp3.to_latex(index=False, float_format="%.4f", caption="F1-Score of Imbalanced Techniques", label="tab:imbalance")
        latex_output += "\n\n"

        plt.xlabel("Training Examples (Imbalanced Split)")
        plt.ylabel("Cross-Validated F1 Score")
        plt.title('Imbalanced Dataset Performance over Sample Size')
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig('imbalance_results.png', dpi=300)

    # --- Write LaTeX output ---
    print("\nWriting LateX tabular outputs to 'latex_tables.tex'...")
    with open('latex_tables.tex', 'w', encoding='utf-8') as f:
        f.write(latex_output.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline'))

    print("DONE! All experiments completed successfully. Numerical logs saved.")

if __name__ == "__main__":
    main()
