# 🛡️ Multi-Agent AI System for Phishing Email Detection and Response Automation
### **Developed by: _Sudarshan Sridhar_**

A complete end-to-end phishing email detection and response automation system that combines:

- **Classical Machine Learning** (TF-IDF + Logistic Regression)  
- **Rule-based heuristics**  
- **Threat Intelligence Integration** (OpenPhish + URLHaus)  
- **Local LLM Agents via Ollama** (Qwen 2.5, Llama 3.1)  
- **A Multi-Agent LangGraph Pipeline**  
- **FastAPI backend**  
- **Streamlit analytics dashboard**  
- **Robustness evaluation tools** (adversarial mutations, cross-model comparison)

This system is optimized for **local execution, high accuracy, high explainability, and reproducible workflows** — suitable for research, enterprise simulations, SOC analysis, and LLM-augmented email security.

---

# 📚 Table of Contents

1. Features  
2. System Architecture  
3. Project Structure  
4. Datasets  
5. Machine Learning Pipeline  
6. LangGraph Multi-Agent Pipeline  
7. Threat Intelligence  
8. Local LLM Integration (Ollama)  
9. Frontend Dashboard  
10. Backend API Endpoints  
11. Reproducibility Guide  
12. Testing  
13. Robustness Utilities  
14. Author  

---

# ⭐ Features

### **1. Multi-Agent Detection Pipeline**
- Ingestion & preprocessing  
- ML classifier  
- Rule-based heuristics  
- Threat intelligence lookup  
- LLM explainability  
- LLM safe reply generation  
- SOC recommendations  
- Forensic summary generation  

### **2. Hybrid ML + LLM**
| Task | Handled By |
|------|------------|
| Classification | ML model |
| Reasoning | LLM |
| Guided responses | LLM |
| Forensics | LLM |

### **3. Threat Intelligence**
- OpenPhish URLs  
- URLHaus malicious hosts  
- Daily auto-refresh  
- Cached locally  

### **4. Evaluation Dashboard**
- Label distribution  
- TF-IDF token importance  
- URL statistics  
- Text length statistics  

### **5. Performance Tools**
- ROC & PR curves  
- Threshold tuning  
- Confusion matrices  
- Classification report  

### **6. Robustness Tools**
- Adversarial mutations  
- Cross-model LLM comparison  

---

# 🧠 System Architecture

```
               ┌──────────────────────────┐
               │     Streamlit Frontend   │
               └──────────────┬───────────┘
                              │ REST API
                              ▼
               ┌──────────────────────┐
               │     FastAPI Backend  │
               └──────────┬───────────┘
                          │
                  LangGraph Pipeline
                          │
   ┌──────────────┬───────────────┬───────────────┬
   ▼              ▼               ▼               ▼
ML Classifier   Heuristics   Threat Intel     Local LLM Agents
```

---

# 📂 Project Structure

```
PHISH-LC/
│
├── backend/
│   ├── api/
│   ├── core/
│   ├── graph/
│   └── models/
│
├── data/
│   ├── manual_tests/
│   ├── processed/
│   └── raw/
│
├── frontend/
├── training/
├── quick_test.py
├── test_graph.py
├── requirements.txt
└── README.md
```

---

# 📊 Datasets

| Dataset | Type |
|--------|------|
| Enron | Benign |
| CEAS 2008 | Mixed |
| Nazario | Phishing |

Processed using:
```
training/prepare_dataset.py
```

Output:
```
data/processed/combined.jsonl
```

---

# 🔬 Machine Learning Pipeline

- TF-IDF vectorizer (60k features, bigrams)  
- Logistic Regression  
- Balanced class weights  
- Stored at:

```
backend/models/email_classifier.joblib
```

Predict using:

```python
from backend.core.model_loader import predict_proba
predict_proba("email text")
```

---

# 🧩 LangGraph Multi-Agent Pipeline

Order:

```
ingest → filter → threat_intel → explain → response → soc → forensics
```

Each step enriches the shared state.

---

# 🌐 Threat Intelligence Integration

- OpenPhish URL feed  
- URLHaus malicious hosts  
- Cached + refreshed daily  
- Integrated into risk scoring  

---

# 🤖 Local LLM Integration (Ollama)

Models used:

```
qwen2.5:3b
qwen2.5:7b
llama3.1:8b
```

Used for reasoning, explanation, and reply generation.

---

# 🖥️ Frontend Dashboard (Streamlit)

Tabs:

### **Email Analyzer**
- Decision + risk score  
- ML + heuristics + TI  
- LLM explanation  
- Safe reply  
- SOC actions  
- Forensic summary  

### **Evaluation & Performance**
- Dataset insights  
- Token importance  
- ROC & PR curves  
- Confusion matrices  
- Robustness tools  

---

# 🔗 Backend API Endpoints

| Endpoint | Description |
|----------|-------------|
| POST /analyze_email | Full pipeline |
| POST /eval_summary | Offline evaluation |
| POST /threshold_sweep | PR/recall tuning |
| POST /roc_curve | ROC points |
| POST /pr_curve | PR points |
| POST /confusion_at_threshold | Confusion matrix |
| POST /adversarial_mutations | Mutated variants |
| POST /cross_model_compare | Compare LLMs |

---

# 🔁 Full Reproducibility Guide

### **1. Clone**

```bash
git clone https://github.com/sudarshan-sridhar/Multi-Agent-AI-System-for-Phishing-Email-Detection-and-Response-Automation.git
cd Multi-Agent-AI-System-for-Phishing-Email-Detection-and-Response-Automation
```

### **2. Virtual Environment**

Windows:
```bash
python -m venv env
env\Scriptsctivate
```

Mac/Linux:
```bash
python3 -m venv env
source env/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Install Ollama + Models**

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
```

### **5. Prepare Dataset**

```bash
cd training
python prepare_dataset.py
```

### **6. Train Classifier**

```bash
python train_classifier.py
```

### **7. Start Backend**

```bash
cd ../backend/api
uvicorn main:app --reload --port 8000
```

### **8. Start Frontend**

```bash
cd ../../frontend
streamlit run app.py
```

---

# 🧪 Testing

### Quick test:

```bash
python quick_test.py
```

### Full pipeline test:

```bash
python test_graph.py
```

---

# 👤 Author

**Sudarshan Sridhar**

