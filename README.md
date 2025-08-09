# 🛍 RetailVista: AI-Powered Retail Intelligence & Forecasting

**RetailVista** is a **full-stack AI retail analytics platform** that unifies **demand forecasting, customer segmentation, promo uplift modeling, multilingual sentiment analysis, and GPT-powered marketing campaigns** — all in a single, production-ready system.

Designed for **retail analysts, growth teams, and marketing strategists**, RetailVista transforms **POS data, loyalty metrics, and customer feedback** into **actionable intelligence**, backed by **explainable AI** and **interactive dashboards**.

---

## 🔍 Why RetailVista?

> Most retail tools tell you *what happened*.  
> **RetailVista tells you what will happen next — and what to do about it.**

By combining **time-series forecasting**, **customer segmentation**, **causal impact modeling**, and **LLM-generated marketing briefs**, RetailVista empowers teams to:

✅ Predict SKU-level demand weeks ahead  
✅ Segment customers dynamically based on behavior & loyalty  
✅ Quantify marketing impact using causal inference  
✅ Extract sentiment & sarcasm from multilingual reviews  
✅ Generate data-backed GPT marketing briefs for targeted campaigns  

---

## 🛠 Core Features

| Category | Highlights |
|----------|------------|
| **📈 Demand Forecasting** | Prophet + NeuralProphet ensemble models for SKU-level forecasting |
| **👥 Customer Segmentation** | HDBSCAN clustering with RFM & behavioral features |
| **🎯 Promo Uplift Modeling** | EconML + DoWhy for causal campaign impact estimation |
| **🗣 Arabic NLP Intelligence** | Sentiment + sarcasm detection using LABR & ArSarcasm datasets |
| **💬 Multilingual Sentiment** | Arabic + English classification via QARiB BERT |
| **💡 GPT Campaign Summaries** | Mistral v0.3 LLM-generated, data-driven marketing briefs |
| **⚖️ Explainable Fairness Audit** | LIME-based explainability + bias audits by gender/dialect |
| **⚡ Unified API Layer** | FastAPI endpoints for forecasting, clustering, sentiment, GPT |
| **📊 Interactive Dashboard** | Streamlit Analyst Lab for forecasts, segmentation, and insights |

---

## 🧬 Tech Stack

- **Data Science & ML**: Pandas, NumPy, Scikit-learn, Prophet, NeuralProphet, HDBSCAN  
- **NLP & LLMs**: Transformers, QARiB BERT, Mistral v0.3  
- **Causal Inference**: EconML, DoWhy  
- **Explainability**: LIME-based token-level attribution (dual-language)  
- **Backend**: FastAPI, Uvicorn  
- **Frontend**: Streamlit  
- **Deployment**: Streamlit Cloud  
- **Data Sources**: LABR, ArSarcasm v1/v2  

---

## 📂 Key Artifacts

| Path | Content |
|------|---------|
| `data/forecasts/` | Prophet, NeuralProphet, and ensemble outputs |
| `data/processed/` | RFM features, clustering labels, sentiment scores, GPT briefs |
| `outputs/lime_sentiment/` | Dual-language LIME sentiment visualizations |
| `streamlit_lab/` | Analyst simulation & scenario testing app |
| `app/main.py` | Unified FastAPI backend with all inference endpoints |

---

## 🚀 Example API Workflow

| Step | Endpoint | Description |
|------|----------|-------------|
| **1️⃣ Forecast SKU Demand** | `/forecast/ensemble` | Get ensemble time-series predictions |
| **2️⃣ Segment Customers** | `/cluster/infer` | Assign customer clusters |
| **3️⃣ Estimate Promo Impact** | `/uplift` *(future)* | Quantify campaign uplift |
| **4️⃣ Analyze Reviews** | `/sentiment` | Get sentiment & sarcasm classification |
| **5️⃣ Generate Campaign Brief** | `/campaign_summary` | Produce GPT-based marketing content |

---

## 🌐 Live & Interactive

- **Streamlit Analyst Lab** → [https://retailvista.streamlit.app/](https://retailvista.streamlit.app/) *(interactive dashboard)*  
- **Swagger API Docs** → `http://localhost:8000/docs` *(local FastAPI documentation)*  

---

## ⚙️ Next Extensions

- Real-time retail event alerts  
- Automated campaign A/B testing  
- Predictive churn scoring  
- Multi-language GPT brief generation  

---

## 🤝 Contributing

Contributions are welcome!

- Additional LLM fine-tuning for retail use cases  
- Expanded dataset integration  
- Dashboard visualization enhancements  

---

## 📜 License
MIT License © 2025 Akash Gutti