import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sentence_transformers import SentenceTransformer

# Ignore some user compilation warnings from transformers
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    print("Loading dataset...")
    df = pd.read_csv('dataset.csv')
    X = df['query'].values
    y = df['label_id'].values

    # 1. Define Model 1: TF-IDF + Logistic Regression
    tfidf_lr_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # 2. Define Model 2: BERT Embeddings + Logistic Regression
    print("Downloading/Loading BERT model for sentence embeddings (this may take a moment on first run)...")
    # all-MiniLM-L6-v2 is an excellent, extremely fast BERT-based model for generating text embeddings
    bert_model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    print("Extracting BERT embeddings for all 400 queries...")
    X_bert = bert_model.encode(X, show_progress_bar=True)
    bert_lr_model = LogisticRegression(random_state=42, max_iter=1000)

    # 3. Cross-Validation Configuration
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    print("\n========================================================")
    print("--- Evaluating Model 1: TF-IDF + Logistic Regression ---")
    cv_tfidf = cross_validate(tfidf_lr_pipeline, X, y, cv=5, scoring=scoring)
    print(f"Accuracy:  {np.mean(cv_tfidf['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(cv_tfidf['test_precision']):.4f}")
    print(f"Recall:    {np.mean(cv_tfidf['test_recall']):.4f}")
    print(f"F1 Score:  {np.mean(cv_tfidf['test_f1']):.4f}")
    print(f"ROC AUC:   {np.mean(cv_tfidf['test_roc_auc']):.4f}")

    print("\n========================================================")
    print("--- Evaluating Model 2: BERT + Logistic Regression ---")
    cv_bert = cross_validate(bert_lr_model, X_bert, y, cv=5, scoring=scoring)
    print(f"Accuracy:  {np.mean(cv_bert['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(cv_bert['test_precision']):.4f}")
    print(f"Recall:    {np.mean(cv_bert['test_recall']):.4f}")
    print(f"F1 Score:  {np.mean(cv_bert['test_f1']):.4f}")
    print(f"ROC AUC:   {np.mean(cv_bert['test_roc_auc']):.4f}")
    print("========================================================\n")

    # 4. Generate Confusion Matrices 
    print("Generating cross-validated predictions for confusion matrices...")
    y_pred_tfidf = cross_val_predict(tfidf_lr_pipeline, X, y, cv=5)
    y_pred_bert = cross_val_predict(bert_lr_model, X_bert, y, cv=5)

    cm_tfidf = confusion_matrix(y, y_pred_tfidf)
    cm_bert = confusion_matrix(y, y_pred_bert)

    # 5. Plotting results side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Small-LLM (0)', 'Large-LLM (1)'], 
                yticklabels=['Small-LLM (0)', 'Large-LLM (1)'])
    axes[0].set_title('TF-IDF + Logistic Regression')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Small-LLM (0)', 'Large-LLM (1)'], 
                yticklabels=['Small-LLM (0)', 'Large-LLM (1)'])
    axes[1].set_title('BERT + Logistic Regression')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    print("Success! Evaluation plots saved to 'confusion_matrices.png'.")

if __name__ == '__main__':
    main()
