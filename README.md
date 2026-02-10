# AI System for Structuring and Querying Daily Drilling Reports (DDRs) 🚀

**Multi-modal AI pipeline** that turns messy DDR PDFs into structured, queryable data + natural language analytics

<p align="center">
  <img src="https://img.shields.io/badge/Status-Research%20%26%20Demo-blue?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Tech-Streamlit%20·%20Gemini%20·%20PaddleOCR%20·%20YOLO-green?style=for-the-badge" alt="Tech">
  <img src="https://img.shields.io/badge/Use%20Case-Oil%20%26%20Gas%20Drilling-orange?style=for-the-badge" alt="Domain">
</p>

## 🎯 Overview

Daily Drilling Reports (DDRs) are semi-structured technical documents containing:

- Operational logs  
- Tables (fluids, survey, lithology, gas, etc.)  
- Engineering plots (pressure–time, pressure–depth)

**This system automatically:**

- 📄 Parses DDR PDFs using Vision + OCR  
- 🗃️ Extracts structured data into SQL database  
- 📊 Understands engineering plots (pressure-time, depth plots)  
- 🤖 Performs LLM-based analytics & summarization  
- 💬 Enables natural-language querying via chatbot

Deployed as an **interactive Streamlit web application** for research & demonstration.

## 🏗 High-Level Architecture
**PDF / DOCX / Images**  
↓  
**Layout Detection + OCR**  
↓  
Structured Text + Tables + Plots  
↓  
Domain Parsers + VLM  
↓  
Clean Entities + Time-series + Plot Data  
↓  
SQL Database  

- Analytics + Summaries → PDF Reports  
- TAG-style Chatbot → Natural Language Answers


## 📂 Repository Structure
.  
├── app.py — 🎨 Streamlit UI & workflow orchestration  
├── actions/ — 🔄 High-level ingestion & mutation transactions  
├── AI/ — 🧠 LLM & Vision-Language model integrations  
├── domain/ — 🏭 Domain models & shared abstractions  
├── fine-tuning/ — 🛠 Fine-tuning experiments & assets  
├── model_code/ — 👁️ Vision & document understanding models  
├── reports/ — 📑 Analytics PDF report generation  
├── services/ — ⚙️ Core reusable services (OCR, DB, routing…)  
├── tables/ — 📋 Table OCR, parsers & DB writers  
├── requirements.txt  
└── .gitignore  


## ✨ Main Features (Streamlit Workflows)
1. **DDR PDF Report Ingestion**  
   Upload PDF/DOCX → Layout + OCR → Metadata + Tables → SQL

2. **Engineering Plot Ingestion**  
   Upload PNG/JPG → Auto-detect type (P vs t / P vs depth) → Extract data & interpretation → Store

3. **Analytics & Natural Language Querying**  
   - LLM-powered summaries & insights  
   - Generate formatted PDF analytics reports  
   - TAG-style chatbot (Text → Action → Graph)

## 🔥 Key Technical Highlights

| Area                  | Technology / Approach                          | Purpose                              |
|-----------------------|------------------------------------------------|--------------------------------------|
| Layout Detection      | YOLO-based                                     | Page structure & region detection    |
| OCR & Table Extraction| PaddleOCR                                      | High-accuracy table reading          |
| Plot Understanding    | Gemini Vision + structured JSON output         | Pressure-time & pressure-depth data  |
| LLM Usage             | Gemini (pluggable)                             | Analytics, interpretation, Text-to-SQL |
| Database              | SQLite / PostgreSQL (configurable)             | Structured DDR entities & time-series|
| Chat Interface        | TAG pipeline (Text–Action–Graph)               | Reliable NL → SQL → Answer           |
| Output                | Markdown + PDF analytics reports               | Human-readable insights              |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional but recommended)
# Use GPU for faster OCR & vision if available

# 3. Launch the app
streamlit run app.py

💬 Example Natural Language Queries

Summarize drilling activities for this reporting period
Which operations caused the most downtime?
Were there any abnormal gas readings?
Show and explain the pressure–time trend across offset wells
Compare ROP performance between these two sections
What was the total mud loss last week?

🔮 Research Directions & Future Work

🔄 Replace hosted LLMs with local / open-weight models (Llama 3.1, Qwen-VL, etc.)
🤝 Multi-agent analytics orchestration
📈 Advanced VLM-based plot digitization & interpretation
🌐 Cross-well / cross-field comparative analysis
👩‍🏫 Active learning for layout detection & table parsing
📊 Time-series anomaly detection on drilling parameters

⚠️ Important Notes

🔬 This repository is for research and demonstration purposes only
🚫 No proprietary or confidential data is included
🛠 Some components may require API keys (currently Gemini)
