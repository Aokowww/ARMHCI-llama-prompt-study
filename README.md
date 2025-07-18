# LLaMA Prompt Engineering Study

A controlled 4×4 factorial investigation of how **prompt format** and **in‑context shot count** affect **accuracy** and **reproducibility** in instruction‑tuned LLaMA models on a sentiment classification task.

---

## Project Overview

- **Models:**  
  - **Model 1:** `meta-llama-3-8b-instruct-Q4_K_S` (8 B parameters, 4‑bit quantized)  
  - **Model 2:** `llama-3.2-3b-instruct-f16` (3 B parameters, 16‑bit)

- **Prompt Formats:** Plain Text, Markdown, YAML, JSON  
- **Shot Counts:** 0, 1, 3, 5 examples (“shots”)  

We ran each of 16 prompt conditions × 10 repeat trials per input, measured:
- **Accuracy** against human‑annotated sentiment labels  
- **Reproducibility** (run‑to‑run exact‑match rate)  

Statistical analysis was performed via two‑way ANOVAs (Format × Shots) and Tukey HSD post‑hoc tests, per model.

---

## Getting Started

### Prerequisites

- **Git** (to clone this repo)  
- **Python 3.11**  
  - `pip install -r requirements.txt`  
    (includes `transformers`, `pandas`, `numpy`, `statsmodels`, `matplotlib`)  

