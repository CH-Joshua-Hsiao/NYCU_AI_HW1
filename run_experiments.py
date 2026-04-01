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
    df = pd.read_csv('labeled_data.csv')
    df = df.dropna(subset=['query', 'label'])
    
    # Map label to 0/1, 'small' -> 0, 'large' -> 1
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower().strip() == 'small' else 1)
    
    X = df['query'].values
    y = df['label_id'].values

    print("Data loaded. Class distribution:", {0: sum(y==0), 1: sum(y==1)})

    # Initialize TF-IDF + LR
    tfidf_lr = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # BERT
    print("Loading SentenceTransformer model...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Encoding text. This may take a minute...")
    X_bert = bert_model.encode(X, show_progress_bar=False)
    clf_bert = LogisticRegression(random_state=42, max_iter=1000)

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
    axes[0].set_title('TF-IDF + Logistic Regression')
    
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Small-LLM (0)', 'Large-LLM (1)'], yticklabels=['Small-LLM (0)', 'Large-LLM (1)'])
    axes[1].set_title('BERT + Logistic Regression')
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    print("Saved confusion_matrices.png")

    # --- Experiment 2 ---
    print("\n=== Experiment 2: Dataset Size (Learning Curve) ===")
    train_sizes, train_scores, test_scores = learning_curve(
        clf_bert, X_bert, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='f1'
    )
    
    plt.figure(figsize=(8, 6))
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="BERT Cross-validation F1 score")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    train_sizes_tf, train_scores_tf, test_scores_tf = learning_curve(
        tfidf_lr, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.2, 1.0, 5), scoring='f1'
    )
    test_scores_mean_tf = np.mean(test_scores_tf, axis=1)
    test_scores_std_tf = np.std(test_scores_tf, axis=1)
    
    plt.plot(train_sizes_tf, test_scores_mean_tf, 'o-', color="b", label="TF-IDF F1 score")
    plt.fill_between(train_sizes_tf, test_scores_mean_tf - test_scores_std_tf,
                     test_scores_mean_tf + test_scores_std_tf, alpha=0.1, color="b")

    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.title("Learning Curves (BERT vs TF-IDF)")
    plt.grid()
    plt.savefig('learning_curve.png', dpi=300)
    print("Saved learning_curve.png")

    # --- Experiment 3 ---
    print("\n=== Experiment 3: Class Imbalance & Resampling ===")
    if SMOTE is None:
        print("Please run `pip install imbalanced-learn` for SMOTE. Terminating experiment 3.")
    else:
        # Imbalance ratio: 200 small (0), 50 large (1).
        idx_small = np.where(y == 0)[0]
        idx_large = np.where(y == 1)[0]
        np.random.seed(42)
        idx_large_sub = np.random.choice(idx_large, size=50, replace=False)
        idx_imbalanced = np.concatenate((idx_small, idx_large_sub))
        
        X_imb = X[idx_imbalanced]
        y_imb = y[idx_imbalanced]
        X_bert_imb = X_bert[idx_imbalanced]
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_bert_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb)
        
        # 1. Unbalanced Model
        clf_unbal = LogisticRegression(random_state=42, max_iter=1000)
        clf_unbal.fit(X_tr, y_tr)
        y_p_unbal = clf_unbal.predict(X_te)
        f1_unbal = f1_score(y_te, y_p_unbal)
        recall_unbal = recall_score(y_te, y_p_unbal)
        
        # 2. Balanced mapped
        clf_cw = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        clf_cw.fit(X_tr, y_tr)
        y_p_cw = clf_cw.predict(X_te)
        f1_cw = f1_score(y_te, y_p_cw)
        recall_cw = recall_score(y_te, y_p_cw)
        
        # 3. SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_tr, y_tr)
        clf_smote = LogisticRegression(random_state=42, max_iter=1000)
        clf_smote.fit(X_res, y_res)
        y_p_smote = clf_smote.predict(X_te)
        f1_smote = f1_score(y_te, y_p_smote)
        recall_smote = recall_score(y_te, y_p_smote)
        
        labels_bar = ['No Resampling', 'Class Weights', 'SMOTE']
        f1s = [f1_unbal, f1_cw, f1_smote]
        recalls = [recall_unbal, recall_cw, recall_smote]

        x_pos = np.arange(len(labels_bar))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8,6))
        
        ax.bar(x_pos - width/2, f1s, width, label='F1 (Class 1)')
        ax.bar(x_pos + width/2, recalls, width, label='Recall (Class 1)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_bar)
        ax.set_ylabel('Score')
        ax.set_title('Performance on Imbalanced Dataset (20% Class 1)')
        ax.legend()
        plt.tight_layout()
        plt.savefig('imbalance_results.png', dpi=300)
        print("Saved imbalance_results.png")

    # --- Experiment 4 ---
    print("\n=== Experiment 4: PCA Dimensionality Reduction ===")
    pca = PCA(n_components=50, random_state=42)
    X_bert_pca = pca.fit_transform(X_bert)
    cv_pca = cross_validate(clf_bert, X_bert_pca, y, cv=5, scoring=scoring)
    
    f1_nopca = np.mean(cv_bert['test_f1'])
    f1_pca = np.mean(cv_pca['test_f1'])
    
    tfidf_pca_lr = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('pca', TruncatedSVD(n_components=50, random_state=42)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
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
