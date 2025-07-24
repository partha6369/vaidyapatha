---
title: Dr. Partha’s Research Centre
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.38.1
app_file: VaidyaPatha.py
pinned: false
license: mit
short_description: Semantically similar clinical trial finder.
---
# 🧠 VaidyaPatha (वैद्यपथ) — Semantic Clinical Trial Explorer

**VaidyaPatha** is a Sanskrit-inspired name meaning *“The Path of the Healer.”* This Gradio-based application allows researchers, clinicians, and patients to semantically explore clinical trial data using natural language queries. 

## 🔍 What does it do?

- Accepts a free-text query like “brain tumour clinical trials.”
- Uses a Sentence Transformer model to embed the query.
- Searches a FAISS index of pre-embedded clinical trial summaries.
- Returns the **Top 3 most relevant clinical trials**.
- For each matched trial:
  - Generates a **Word Cloud** from the full text.
  - Displays the **raw combined clinical text**.
  - Shows the **Clinical Trial ID**.

## 🚀 Quickstart

### 1. Clone this repository and install dependencies

```bash
pip install -r requirements.txt
```

## 📜 License

This project is released under the MIT License.