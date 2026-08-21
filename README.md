# 🏥 MediSense AI

### AI-Powered Healthcare Diagnosis Assistant

**CDAC BDA Major Project | Feb 2026 Batch | SIIT Pune**
**Uttam Suthar & Tushar Bhute**

---

## 🔗 Live Demo

**App:** https://medisense-aichat.streamlit.app  
**Tableau Dashboard:** https://public.tableau.com/app/profile/uttam.suthar1718/viz/ML-project_17846536081640/Dashboard1

---

## 📌 Project Overview

MediSense AI is an end-to-end AI-powered healthcare diagnosis assistant that accepts patient symptom descriptions in plain English and returns a differential diagnosis across 677 diseases, grounded in WHO medical literature.

---

## 🏗️ System Architecture

User Input (Streamlit)
↓
Security Layer (Sanitization + Topic Guard)
↓
NLP Symptom Extractor (Fuzzy Matching + Synonym Dictionary)
↓
XGBoost Classifier (677 diseases | 73.79% accuracy)
↓
RAG Retrieval (1,174 WHO document chunks)
↓
Llama 3.1 via Groq API
↓
Patient Report (English / Hindi / Marathi)

---

## 📊 Dataset

| Attribute        | Value                                             |
| ---------------- | ------------------------------------------------- |
| Source           | Kaggle — dhivyeshrk/diseases-and-symptoms-dataset |
| Total Records    | 246,945                                           |
| Diseases         | 773 (677 after filtering)                         |
| Symptom Features | 377 (146 after VarianceThreshold)                 |
| Missing Values   | 0                                                 |

---

## 🤖 Model Performance

| Model                  | Test Accuracy | F1 Score   | Model Size  |
| ---------------------- | ------------- | ---------- | ----------- |
| Random Forest          | 64.27%        | 64.27%     | ~1.1 GB     |
| **XGBoost (selected)** | **73.79%**    | **73.79%** | **39.9 MB** |

---

## 🛠️ Tech Stack

| Layer          | Technology                                            |
| -------------- | ----------------------------------------------------- |
| ML Model       | XGBoost (n=50, depth=5, hist method)                  |
| NLP            | FuzzyWuzzy + Custom Synonym Dictionary                |
| GenAI          | Llama 3.1 8B via Groq API                             |
| RAG            | Sentence Transformers + Cosine Similarity             |
| Knowledge Base | 9 WHO PDFs → 1,174 chunks                             |
| Security       | Regex sanitization + Topic guard + Prompt constraints |
| Frontend       | Streamlit                                             |
| Deployment     | Streamlit Cloud                                       |
| Analytics      | Tableau Public                                        |

---

## 📁 Repository Structure

medisense-ai/
├── app.py # Complete Streamlit application
├── xgb_model_lite.json # XGBoost model (39.9 MB, 677 classes)
├── medisense_deploy.pkl # Label encoder + feature names
├── rag_knowledge.pkl # 1,174 WHO document embeddings
├── nlp_artifacts.pkl # Symptom lookup + synonym dictionary
├── requirements.txt # Pinned dependencies
├── Notebooks/ # Jupyter notebooks (EDA, training)
└── README.md

---

## ⚙️ Local Setup

```bash
git clone https://github.com/UttamSuthar09/medisense-ai.git
cd medisense-ai
pip install -r requirements.txt

# Set Groq API key
export GROQ_API_KEY=""

# Run app
streamlit run app.py
```

---

## 🔒 Security Features

- **Layer 1:** Regex-based prompt injection sanitization (15+ patterns)
- **Layer 2:** Topic guard — blocks non-medical queries before LLM call
- **Layer 3:** System prompt constraints — LLM instructed to refuse off-topic requests

---

## 📈 Key Numbers

- **246,945** training records
- **677** disease classes
- **146** symptom features (after variance filtering)
- **73.79%** test accuracy
- **1,174** WHO document chunks in RAG
- **3-layer** security architecture
- **3** languages supported (English, Hindi, Marathi)

---

## ⚠️ Disclaimer

MediSense AI is a screening tool only. It is not a substitute for professional medical advice. Always consult a qualified healthcare professional.
