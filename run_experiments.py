import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_predict, learning_curve, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None

import warnings
warnings.simplefilter(action='ignore')

def main():
    print("Loading dataset...")
    df = pd.read_csv('llm_relabeled_data.csv')
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

    # --- Experiment 1 ---
    print("\n=== Experiment 1: Baseline ===")
    cv_tfidf = cross_validate(tfidf_lr, X, y, cv=5, scoring=scoring)
    cv_bert = cross_validate(clf_bert, X_bert, y, cv=5, scoring=scoring)
    
    print(f"TF-IDF | Accuracy: {np.mean(cv_tfidf['test_accuracy']):.4f}, F1: {np.mean(cv_tfidf['test_f1']):.4f}, Precision: {np.mean(cv_tfidf['test_precision']):.4f}, Recall: {np.mean(cv_tfidf['test_recall']):.4f}")
    print(f"BERT   | Accuracy: {np.mean(cv_bert['test_accuracy']):.4f}, F1: {np.mean(cv_bert['test_f1']):.4f}, Precision: {np.mean(cv_bert['test_precision']):.4f}, Recall: {np.mean(cv_bert['test_recall']):.4f}")

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
    print("Saved confusion_matrices.png")

    # --- Experiment 2 ---
    print("\n=== Experiment 2: Dataset Size & PCA (10-point Learning Curves) ===")
    
    plt.figure(figsize=(10, 7))
    train_sizes_scale = np.linspace(0.1, 1.0, 10)
    
    models_to_test = {
        'BERT': clf_bert,
        'TF-IDF': tfidf_lr,
        'BERT + PCA': bert_pca_lr,
        'TF-IDF + PCA': tfidf_pca_lr
    }
    
    colors = {'BERT': 'g', 'TF-IDF': 'b', 'BERT + PCA': 'darkgreen', 'TF-IDF + PCA': 'navy'}
    linestyles = {'BERT': '-', 'TF-IDF': '-', 'BERT + PCA': '--', 'TF-IDF + PCA': '--'}
    
    for name, model in models_to_test.items():
        # X mapping depending on if it's BERT raw features or Text
        features = X_bert if 'BERT' in name else X
        t_sizes, t_scores, te_scores = learning_curve(
            model, features, y, cv=5, n_jobs=-1, 
            train_sizes=train_sizes_scale, scoring='f1'
        )
        te_mean = np.mean(te_scores, axis=1)
        
        plt.plot(t_sizes, te_mean, marker='o', color=colors[name], linestyle=linestyles[name], label=f"{name} F1-Score")
        
    plt.xlabel("Training examples")
    plt.ylabel("Cross-Validated F1 Score")
    plt.legend(loc="lower right")
    plt.title("Learning Curves (Granular Sampling w/ PCA Comparisons)")
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300)
    print("Saved learning_curve.png")

    # --- Experiment 3 ---
    print("\n=== Experiment 3: Class Imbalance Learning Curves ===")
    if SMOTE is None:
        print("Please run `pip install imbalanced-learn` for SMOTE. Terminating experiment 3.")
    else:
        from imblearn.pipeline import Pipeline as ImbPipeline
        idx_small = np.where(y == 0)[0]
        idx_large = np.where(y == 1)[0]
        np.random.seed(42)
        idx_large_sub = np.random.choice(idx_large, size=50, replace=False)
        idx_imbalanced = np.concatenate((idx_small, idx_large_sub))
        
        X_imb = X_bert[idx_imbalanced]
        y_imb = y[idx_imbalanced]
        
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
        
        plt.figure(figsize=(9, 6))
        
        # We start at 30% because SMOTE requires at least k_neighbors samples in minority class
        train_sizes_scale = np.linspace(0.4, 1.0, 7)
        colors = {'No Resampling': 'red', 'Class Weights': 'blue', 'SMOTE': 'green'}
        
        for name, model in imb_models.items():
            t_sizes, t_scores, te_scores = learning_curve(
                model, X_imb, y_imb, cv=5, n_jobs=-1, 
                train_sizes=train_sizes_scale, scoring='f1'
            )
            te_mean = np.mean(te_scores, axis=1)
            te_std = np.std(te_scores, axis=1)
            
            plt.plot(t_sizes, te_mean, marker='o', color=colors[name], label=name)
            plt.fill_between(t_sizes, te_mean - te_std, te_mean + te_std, alpha=0.1, color=colors[name])
            
        plt.xlabel("Training Examples (Imbalanced Split)")
        plt.ylabel("Cross-Validated F1 Score")
        plt.title('Imbalanced Dataset Performance over Sample Size')
        plt.legend(loc="best")
        plt.grid()
        plt.tight_layout()
        plt.savefig('imbalance_results.png', dpi=300)
        print("Saved imbalance_results.png")

    # --- Experiment 4 ---
    print("\n=== Experiment 4: PCA Dimensionality Reduction Strict Comparison ===")
    cv_pca = cross_validate(bert_pca_lr, X_bert, y, cv=5, scoring=scoring)
    
    f1_nopca = np.mean(cv_bert['test_f1'])
    f1_pca = np.mean(cv_pca['test_f1'])
    
    cv_tfidf_pca = cross_validate(tfidf_pca_lr, X, y, cv=5, scoring=scoring)
    
    f1_tf_nopca = np.mean(cv_tfidf['test_f1'])
    f1_tf_pca = np.mean(cv_tfidf_pca['test_f1'])

    labels_pca = ['BERT (384d)', 'BERT+PCA (50d)', 'TF-IDF (1000d)', 'TF-IDF+PCA (50d)']
    f1s_pca = [f1_nopca, f1_pca, f1_tf_nopca, f1_tf_pca]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.bar(labels_pca, f1s_pca, color=['green', 'lightgreen', 'blue', 'lightblue'])
    ax.set_ylabel('F1 Score')
    ax.set_title('Impact of Dimensionality Reduction')
    
    for i, v in enumerate(f1s_pca):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig('pca_results.png', dpi=300)
    print("Saved pca_results.png")
    
    print("DONE! All experiments completed successfully.")

if __name__ == "__main__":
    main()
