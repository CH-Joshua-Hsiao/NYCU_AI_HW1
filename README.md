# AI Capstone Project #1: LLM Router Dataset 

**Author:** 蕭至恆 (413551034)
**Repository:** [https://github.com/CH-Joshua-Hsiao/NYCU_AI_HW1](https://github.com/CH-Joshua-Hsiao/NYCU_AI_HW1)

## Overview
As modern applications increasingly integrate Large Language Models (LLMs), routing user queries to the appropriate model has become a critical challenge. Invoking a massive, highly-capable LLM for every request is computationally expensive and incurs unnecessary latency.

This repository features the dataset and classification pipelines designed to train an efficient **"LLM Router."** It is a lightweight, supervised classifier capable of distinguishing between simple queries (low-reasoning, factual, brief) and complex queries (multi-step, reasoning-heavy).

By successfully classifying incoming query complexity natively, inference costs can be drastically optimized.

---

## Project Structure
The repository heavily leverages autonomous scripting. The system systematically samples native internet texts, classifies them strictly via zero-shot prompts leveraging OpenAI's GPT-4o, and ultimately tests embeddings inside Scikit-Learn.

- `fetch_real_queries.py`: Samples 800 unique queries directly from the Hugging Face `SQuAD` and `HotpotQA` databases.
- `llm_labeling.py`: Leverages `openai` constraints to programmatically classify the 800 inputs as `small` (trivia/simple fact) or `large` (multi-hop relational logic).
- `run_experiments.py`: Generates machine-learning performance evaluations utilizing Scikit-Learn pipelines tracking BERT Embeddings (`sentence-transformers`) and TF-IDF matrix overlaps.
- `error_analysis.py`: (Optional) Extracts out-of-fold predictions to evaluate False Positive / False Negative distributions.
- `Report_Draft.tex / .pdf`: Comprehensive final documentation containing learning curves, imbalance validations, and PCA boundaries.

---

## Setup & Installation

**Prerequisites:** Python 3.11+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CH-Joshua-Hsiao/NYCU_AI_HW1.git
   cd NYCU_AI_HW1
   ```

2. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn sentence-transformers datasets openai imbalanced-learn
   ```

3. **Set OpenAI API Key (Required for Labeling Script):**
   ```bash
   # Windows (CMD)
   set OPENAI_API_KEY=your_api_key_here
   
   # Windows (PowerShell)
   $env:OPENAI_API_KEY="your_api_key_here"
   
   # Linux / macOS
   export OPENAI_API_KEY="your_api_key_here"
   ```

---

## Execution Guide

### 1. Generating the Dataset
To pull fresh, randomized samples from SQuAD and HotpotQA:
```bash
python fetch_real_queries.py
```
> *Outputs:* `unlabeled_dataset.csv`

### 2. LLM Labeling
To pass the collected data through GPT-4o for strict multi-hop evaluations:
```bash
python llm_labeling.py
```
> *Outputs:* `llm_relabeled_data_large.csv`

### 3. Running Classifiers & Experiments
Executes a heavy cross-validation testing pipeline exploring Logistic Regressions over Classical NLP (TF-IDF) vs Deep-Learning Transforms (BERT / `all-MiniLM-L6-v2`). Runs all sub-experiments (Sizing impact, SMOTE tracking, PCA limitations):
```bash
python run_experiments.py
```
> *Outputs:* 
> - Numerical CSVs: (`experiment_1_metrics.csv`, `experiment_2_learning_curves.csv`, `experiment_3_imbalance.csv`)
> - Graphic Analysis: (`confusion_matrices.png`, `learning_curve.png`, `imbalance_results.png`, `pca_results.png`)
> - LaTeX Formats: `latex_tables.tex`

---

## Results Snapshot
The real-world constraints showcased that true human natural language is profoundly difficult to isolate by raw word counts alone. While TF-IDF marginally won F1-Scores mathematically, the dense BERT mapping demonstrated higher ROC-AUC margin separability thresholds across logical sentence structures! 

*(For a full deep-dive into semantic vector compressions, dataset imbalances, and synthetic SMOTE recoveries, please compile the included `Report_Draft.tex`)*.