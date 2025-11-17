🛡️ Multi-Agent AI System for Phishing Email Detection and Response Automation

This project is a full-stack phishing email detection system that integrates:

Classical ML (TF-IDF + Logistic Regression)

Rule-based Heuristics

Threat Intelligence (OpenPhish + URLHaus)

Local LLM Agents using Ollama (Qwen 2.5, Llama 3.1)

A Multi-Agent LangGraph Pipeline

FastAPI Backend

Streamlit Frontend Dashboard

Robustness Tools (Adversarial mutations, Cross-model testing)

Everything is optimized for local execution, complete reproducibility, high explainability, and enterprise-style forensic outputs.

📚 Table of Contents

Features

System Architecture

Project Structure

Datasets

Machine Learning Pipeline

LangGraph Multi-Agent Pipeline

Threat Intelligence Integration

Local LLM Integration (Ollama)

Frontend Dashboard

Backend API Endpoints

Full Reproducibility Guide

Testing

Robustness Utilities

⭐ Features
1. Multi-agent detection pipeline

Ingestion + pre-processing

ML classifier (TF-IDF → Logistic Regression)

Heuristic scoring

Threat intelligence lookup

Explainability via LLM

Safe reply generation

SOC recommendations

Forensic report generation

2. Hybrid ML + LLM

ML handles classification

LLM handles reasoning, guidance, response generation

Avoids hallucinations by separating concerns

3. Full Threat Intelligence Support

Integrates OpenPhish URL feeds

Integrates URLHaus malicious host lists

Cached locally & auto-refreshed daily

4. Dataset Evaluation Dashboard

Label distribution

TF-IDF token importance

Word count & URL count graphs

Length statistics

Source dataset distribution

5. Performance Analytics

Threshold sweep for optimal phishing threshold

ROC & PR curves

Confusion matrices

Full classification report

6. Robustness Tools

Adversarial mutations (typos, URL obfuscation, noise sentences)

Cross-model comparison across Ollama LLMs

🧠 System Architecture

                   ┌──────────────────────────┐
                   │     Streamlit Frontend   │
                   │   (UI for Analysis/Eval) │
                   └────────────┬─────────────┘
                                │ REST
                                ▼
                    ┌────────────────────────┐
                    │     FastAPI Backend    │
                    │   /analyze_email etc   │
                    └───────────┬────────────┘
                                │
                           LangGraph Pipeline
                                │
        ┌───────────────┬───────────────┬──────────────┬
        ▼               ▼               ▼              ▼
   ML Classifier   Heuristic Engine   Threat Intel    Local LLM Agents


📂 Project Structure

PROJECT/
│
├── backend/                                  # Backend logic (FastAPI + Core Engine)
│   ├── api/                                   # FastAPI entrypoint
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── core/                                  # Core ML, heuristics, TI, LLM utilities
│   │   ├── __init__.py
│   │   ├── config.py                          # Global configuration
│   │   ├── dataset_utils.py                   # Dataset processing helpers
│   │   ├── eval_utils.py                      # Evaluation metrics & utilities
│   │   ├── email_utils.py                     # Feature extraction, URL parsing, etc.
│   │   ├── llm_manager.py                     # Ollama-based LLM interface
│   │   ├── model_loader.py                    # TF-IDF + LR model loader
│   │   ├── robustness_utils.py                # Adversarial mutations + cross-model tests
│   │   └── ti_manager.py                      # Threat Intelligence (OpenPhish + URLHaus)
│   │
│   ├── graph/                                 # Multi-Agent LangGraph pipeline
│   │   ├── __init__.py
│   │   ├── graph_builder.py                   # Pipeline assembly
│   │   ├── nodes.py                           # All agent nodes (ingest → forensic)
│   │   └── state.py                           # EmailAnalysisState TypedDict
│   │
│   └── models/
│       └── email_classifier.joblib            # Saved TF-IDF vectorizer + LR model
│
├── data/                                      # All datasets
│   ├── manual_tests/
│   │   └── email_test_dataset.txt             # Handwritten evaluation samples
│   │
│   ├── processed/
│   │   └── combined.jsonl                     # Unified training dataset (post-cleaning)
│   │
│   └── raw/
│       ├── CEAS_08.csv                        # Raw CEAS dataset
│       ├── Enron.csv                          # Raw Enron corporate dataset
│       └── Nazario.csv                        # Raw phishing dataset
│
├── frontend/                                  # Streamlit visualization dashboard
│   └── app.py
│
├── training/                                  # Training & preprocessing scripts
│   ├── prepare_dataset.py                     # Normalization pipeline → combined.jsonl
│   └── train_classifier.py                    # Train TF-IDF + Logistic Regression
│
├── quick_test.py                              # Quick check: ML + LLM + TI outputs
├── requirements.txt                           # Required dependencies
├── test_graph.py                              # Full LangGraph pipeline tester
└── README.md                                  # Project documentation

📊 Datasets
Included datasets (local only, not pushed to GitHub):

| Dataset       | Type     | Notes                        |
| ------------- | -------- | ---------------------------- |
| **Enron**     | Benign   | Corporate emails             |
| **CEAS 2008** | Mixed    | Anti-spam conference dataset |
| **Nazario**   | Phishing | Malware researcher corpus    |

These are processed by:

training/prepare_dataset.py


Output file:

data/processed/combined.jsonl   (ignored in Git)

🔬 Machine Learning Pipeline

Training code lives in:

training/train_classifier.py

Steps:

1. Load combined.jsonl

2. Build text = subject + body

3. TF-IDF vectorizer (60k features, 1–2 n-grams)

4. Logistic Regression (balanced class weights)

5. Save model to:

backend/models/email_classifier.joblib

Prediction API:

from backend.core.model_loader import predict_proba
predict_proba("email text")

🧩 LangGraph Multi-Agent Pipeline

Nodes executed in order:

ingest → filter → threat_intel → explain → response → soc → forensics


State contains:

ML probabilities

heuristic scores

threat intel hits

risk scores

final decisions

LLM explanation

user guidance

SOC actions

forensic summary

🌐 Threat Intelligence Integration

TI manager handles:

OpenPhish (URLs)

URLHaus (malicious hosts)

Disk caching + in-memory caching

Daily refresh

Fast membership checks

🤖 Local LLM Integration (Ollama)

Supported tested models:

qwen2.5:3b

qwen2.5:7b

llama3.1:8b

Used for:

explainability

safe reply generation

Not used for classification — improves reliability.

🖥️ Frontend Dashboard (Streamlit)

Tabs:

1. Email Analyzer

Decision, risk, ML probabilities

URL extractions

TI hits

LLM explanation

Safe reply suggestion

SOC actions

Forensic notes

Raw JSON

2. Evaluation & Performance

Dataset summary

TF-IDF feature insights

Word count & URL count graphs

Threshold sweep

ROC curve

Precision–Recall curve

Confusion matrices

Robustness tools

🔗 Backend API Endpoints (FastAPI)

| Endpoint                       | Purpose                   |
| ------------------------------ | ------------------------- |
| `POST /analyze_email`          | Full multi-agent pipeline |
| `POST /eval_summary`           | Offline evaluation        |
| `POST /threshold_sweep`        | Precision/recall tuning   |
| `POST /roc_curve`              | ROC data                  |
| `POST /pr_curve`               | PR curve                  |
| `POST /confusion_at_threshold` | Confusion matrix          |
| `POST /adversarial_mutations`  | Mutated variants          |
| `POST /cross_model_compare`    | Multi-LLM comparison      |


🔁 Full Reproducibility Guide

Below are exact commands to rebuild everything from scratch.

1️⃣ Clone the repository
git clone https://github.com/sudarshan-sridhar/Multi-Agent-AI-System-for-Phishing-Email-Detection-and-Response-Automation.git
cd Multi-Agent-AI-System-for-Phishing-Email-Detection-and-Response-Automation

2️⃣ Create a virtual environment
Windows:
python -m venv env
env\Scripts\activate

Mac/Linux:
python3 -m venv env
source env/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Install Ollama + models

Install Ollama:

https://ollama.ai

Then pull the models:

ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b

5️⃣ Prepare dataset

Place raw datasets inside:

data/raw/


Then run:

cd training
python prepare_dataset.py


Output:

data/processed/combined.jsonl

6️⃣ Train classifier
python train_classifier.py


Output:

backend/models/email_classifier.joblib

7️⃣ Start backend
cd ../backend/api
uvicorn main:app --reload --port 8000


Backend runs at:

http://127.0.0.1:8000

8️⃣ Start frontend
cd ../../frontend
streamlit run app.py


Frontend opens at:

http://localhost:8501

🧪 Testing
Quick ML + TI + LLM sanity test:
python quick_test.py

Full graph run:
python test_graph.py

🛡 Robustness Utilities

Adversarial Mutations
1. Random typos

2. URL obfuscation

3. Extra noise sentences

Cross-Model Comparison

1. Run pipeline across multiple LLMs

2. Detect instability

Both tools are included in the Streamlit UI.

🚀 Future Work

Potential enhancements:

Fine-tuned LLM classifier

BERT/RoBERTa phishing classifier

RNN/LSTM hybrid models

Automated retraining pipelines

UI authentication + multi-user support

👤 Author

Developed by:

Sudarshan Sridhar