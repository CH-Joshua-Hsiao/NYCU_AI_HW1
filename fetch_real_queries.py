import csv
import random
from datasets import load_dataset

def fetch_real_data(num_samples=400):
    print("Loading SQuAD dataset (Simple Queres)...")
    # SQuAD questions are typically short and factual
    squad = load_dataset("squad", split="train")
    
    # We use a set to ensure unique queries are recorded
    simple_queries = set()
    squad_len = len(squad)
    
    print("Sampling unique SQuAD questions...")
    while len(simple_queries) < (num_samples // 2):
        idx = random.randint(0, squad_len - 1)
        q = squad[idx]['question']
        simple_queries.add(q)
        
    print("Loading HotpotQA dataset (Complex Queries)...")
    # HotpotQA questions require multi-hop reasoning over multiple documents
    hotpot = load_dataset("hotpot_qa", "distractor", split="train")
    
    complex_queries = set()
    hotpot_len = len(hotpot)
    
    print("Sampling unique HotpotQA questions...")
    while len(complex_queries) < (num_samples // 2):
        idx = random.randint(0, hotpot_len - 1)
        q = hotpot[idx]['question']
        complex_queries.add(q)
        
    # Combine the datasets
    data = []
    
    for q in simple_queries:
        # Leaving labels empty as requested by the user
        data.append({"query": q, "label": "", "label_id": ""})
        
    for q in complex_queries:
        # Leaving labels empty as requested by the user
        data.append({"query": q, "label": "", "label_id": ""})
        
    # Shuffle so the labels aren't strictly divided top/bottom
    random.shuffle(data)
    
    # Write to an unlabeled CSV for the user to manually grade
    print("Writing to unlabeled_dataset.csv...")
    with open('unlabeled_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['query', 'label', 'label_id'])
        writer.writeheader()
        writer.writerows(data)
        
    print(f"Successfully scraped {len(data)} authentic, unique queries from HuggingFace.")
    print("The file 'unlabeled_dataset.csv' is ready for your manual labeling!")

if __name__ == '__main__':
    fetch_real_data()
