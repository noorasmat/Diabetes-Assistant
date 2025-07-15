# 🧠 Diabetes Support Assistant — Fine-Tuned TinyLlama with QLoRA

This project builds and deploys a lightweight, domain-specific chatbot for answering **diabetes-related questions** using the **TinyLlama-1.1B-Chat** model, fine-tuned with **QLoRA (Quantized Low-Rank Adaptation)**.

## 🚀 Overview

- Fine-tunes `TinyLlama` on a custom dataset of diabetes instructions using **QLoRA + PEFT**
- Merges adapter weights into the base model for efficient deployment
- Deploys an interactive chatbot using **Streamlit**
- Supports both instruction-only and instruction+input-style prompts

---

## 🏥 Example Use Cases

- **What is insulin resistance?**
- **What should a 60-year-old with Type 2 diabetes eat?**
- **What are common symptoms of high blood sugar?**

---

## 🧱 Tech Stack

- `TinyLlama-1.1B-Chat`
- `QLoRA` via `PEFT` (4-bit training)
- `Hugging Face Transformers & Datasets`
- `LangChain`
- `Streamlit`

---

## 🗂️ Project Structure

```bash
.
├── diabetes_dataset.csv         # Alpaca-style fine-tuning data
├── train.py                     # Training script using QLoRA
├── app.py                       # Streamlit-based chatbot UI
├── /tinyllama_diabetes_qlora   # Fine-tuned & merged model directory
└── README.md
