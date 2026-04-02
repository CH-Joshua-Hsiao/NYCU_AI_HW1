import pandas as pd
import os
import time

try:
    from openai import OpenAI
except ImportError:
    print("Please install the OpenAI SDK: pip install openai")
    exit(1)

# Initialize OpenAI Client
# Make sure to set your API key as an environment variable:
# Windows (CMD): set OPENAI_API_KEY=your_api_key_here
# Windows (PowerShell): $env:OPENAI_API_KEY="your_api_key_here"
client = OpenAI(api_key="")

def label_query_with_llm(query, retries=3):
    """
    Sends a query to the LLM and asks it to classify it into 'small' or 'large'.
    - 'small' -> simple, short, factual.
    - 'large' -> complex, multi-hop reasoning, domain-specific.
    """
    prompt = f"""
    You are an AI router system classifying incoming queries into two strict categories: 'small' or 'large'.
    
    CRITERIA FOR 'small' (Simple/Direct):
    - Direct factual questions involving a single distinct entity (e.g., "What is the capital of France?", "When was Barack Obama born?").
    - Simple definitions, translations, or short conversational greetings.
    - Queries that require zero intermediate reasoning or analytical comparisons to answer.
    
    CRITERIA FOR 'large' (Complex/Reasoning):
    - Multi-hop reasoning questions where the AI must connect two or more disparate entities/facts.
    - Intersecting or comparative questions (e.g., "Who starred in Movie A that was also directed by the director of Movie B?" or "Compare X and Y's policies").
    - Detailed coding problems, logic puzzles, domain-specific analytics, or highly convoluted sentence structures.
    - Any question where you logically need to figure out an 'intermediate fact' before getting the final answer is 'large'.

    Query: "{query}"

    Respond ONLY with the exactly the single word 'small' or 'large'. Do not include punctuation or explanations.
    """
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # You can upgrade to gpt-4o for better reasoning reliability
                messages=[
                    {"role": "system", "content": "You are a precise query routing classification agent."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0, # Keep temperature at 0 for deterministic outputs
                max_tokens=5
            )
            
            # Clean up the response
            label = response.choices[0].message.content.strip().lower()
            
            # Ensure it falls strictly into one of our two categories
            if 'small' in label:
                return 'small'
            elif 'large' in label:
                return 'large'
            else:
                return label # Handles edge cases where LLM ignored instructions
                
        except Exception as e:
            print(f"Error calling LLM (attempt {attempt+1}): {e}. Retrying in 2 seconds...")
            time.sleep(2)
            
    return 'unknown' # Fallback if API fails repeatedly

def main():
    input_file = 'unlabeled_dataset_large.csv' # Reading your current dataset
    output_file = 'llm_relabeled_data_large.csv'
    
    print(f"Loading queries from {input_file}...")
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['query']) # Ensure we don't process empty strings
    
    print(f"Total queries to label: {len(df)}")
    
    new_labels = []
    
    for idx, row in df.iterrows():
        query = row['query']
        safe_query = query.encode('ascii', 'ignore').decode('ascii')
        print(f"[{idx+1}/{len(df)}] Analyzing: {safe_query}...")
        
        # 1. Ask the LLM to classify it
        label = label_query_with_llm(query)
        new_labels.append(label)
        
        # Add a brief delay to avoid hitting basic API rate limits
        time.sleep(0.5) 
        
    # 2. Assign back to CSV DataFrame
    df['label'] = new_labels
    
    # 3. Save to a new file so you don't ruin the original
    df.to_csv(output_file, index=False)
    
    # 4. Show distribution
    class_counts = df['label'].value_counts().to_dict()
    print(f"\nSuccessfully relabeled dataset!")
    print(f"Saved to: {output_file}")
    print(f"Final Class Distribution: {class_counts}")

if __name__ == '__main__':
    main()
