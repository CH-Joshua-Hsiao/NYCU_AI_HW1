import csv
import random
import os
from datasets import load_dataset

def fetch_real_data(num_new_samples=400):
    existing_file = 'unlabeled_dataset.csv'
    output_file = 'unlabeled_dataset_large.csv'
    
    existing_queries = set()
    combined_data = []
    
    # 1. Load existing data to avoid duplicates and preserve labels
    if os.path.exists(existing_file):
        print(f"Loading existing dataset from {existing_file}...")
        with open(existing_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_queries.add(row['query'])
                combined_data.append(row)
        print(f"Loaded {len(combined_data)} existing queries.")
    else:
        print(f"File {existing_file} not found. Starting fresh.")
        
    # 2. Fetch new queries
    print("\nLoading SQuAD dataset (Simple Queries)...")
    squad = load_dataset("squad", split="train")
    
    new_simple_queries = set()
    squad_len = len(squad)
    
    print("Sampling unique SQuAD questions...")
    target_simple = num_new_samples // 2
    while len(new_simple_queries) < target_simple:
        idx = random.randint(0, squad_len - 1)
        q = squad[idx]['question']
        # Ensure it's not in the existing dataset AND not already picked
        if q not in existing_queries and q not in new_simple_queries:
            new_simple_queries.add(q)
            
    print("\nLoading HotpotQA dataset (Complex Queries)...")
    hotpot = load_dataset("hotpot_qa", "distractor", split="train")
    
    new_complex_queries = set()
    hotpot_len = len(hotpot)
    
    print("Sampling unique HotpotQA questions...")
    target_complex = num_new_samples // 2
    while len(new_complex_queries) < target_complex:
        idx = random.randint(0, hotpot_len - 1)
        q = hotpot[idx]['question']
        # Ensure it's not in the existing dataset AND not already picked
        if q not in existing_queries and q not in new_complex_queries:
            new_complex_queries.add(q)
            
    # Combine new queries
    new_data = []
    for q in new_simple_queries:
        new_data.append({"query": q, "label": "", "label_id": ""})
    for q in new_complex_queries:
        new_data.append({"query": q, "label": "", "label_id": ""})
        
    random.shuffle(new_data)
    combined_data.extend(new_data)
    
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        # We assume the standard fields
        writer = csv.DictWriter(f, fieldnames=['query', 'label', 'label_id'])
        writer.writeheader()
        writer.writerows(combined_data)
        
    print(f"Successfully concatenated datasets.")
    print(f"Total rows in {output_file}: {len(combined_data)} (of which {len(new_data)} are new & unlabeled).")

if __name__ == '__main__':
    fetch_real_data()
