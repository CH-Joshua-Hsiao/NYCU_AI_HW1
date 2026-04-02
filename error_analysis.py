import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

import warnings
warnings.simplefilter(action='ignore')

def main():
    print("Loading dataset for Error Analysis...")
    df = pd.read_csv('llm_relabeled_data_large.csv')
    df = df.dropna(subset=['query', 'label'])
    
    df['label_id'] = df['label'].apply(lambda x: 0 if str(x).lower().strip() == 'small' else 1)
    
    X = df['query'].values
    y = df['label_id'].values

    print("Loading SentenceTransformer model...")
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    X_bert = bert_model.encode(X, show_progress_bar=False)
    
    clf_bert = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    
    # Get out-of-fold predictions
    print("Predicting with BERT...")
    y_pred = cross_val_predict(clf_bert, X_bert, y, cv=5)
    
    # Analyze errors
    df['predicted'] = y_pred
    
    false_positives = df[(df['label_id'] == 0) & (df['predicted'] == 1)] # True: Small, Predicted: Large
    false_negatives = df[(df['label_id'] == 1) & (df['predicted'] == 0)] # True: Large, Predicted: Small
    
    print("\n" + "="*50)
    print(f"FALSE POSITIVES (True: Small, Predicted: Large) - Count: {len(false_positives)}")
    print("="*50)
    for idx, row in false_positives.head(5).iterrows():
        print(f"Query: {row['query']}")
        print("-" * 50)
        
    print("\n" + "="*50)
    print(f"FALSE NEGATIVES (True: Large, Predicted: Small) - Count: {len(false_negatives)}")
    print("="*50)
    for idx, row in false_negatives.head(5).iterrows():
        print(f"Query: {row['query']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
