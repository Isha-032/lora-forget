# LoRA-Forget: Lightweight Machine Unlearning for PII Removal in Language Models

A pipeline that makes a language model **selectively forget specific training data on demand** — without retraining from scratch.

---

## The Problem

When a model is trained on data containing Personally Identifiable Information (PII) — names, emails, phone numbers — that data gets encoded into the model weights. Simply deleting a database row is not enough. The model still "remembers" it.

This is a real legal problem under GDPR and Australia's Privacy Act — users have the **right to be forgotten**, but current ML systems have no clean way to comply.

---

## The Solution

This project implements **Gradient Ascent Unlearning** — a technique that surgically removes specific PII from a fine-tuned DistilBERT model while preserving its performance on everything else.

```
Input:  "John Smith can be reached at john@gmail.com or 555-1234"

Before Unlearning → ⚠️ CONTAINS PII  (confidence: 98.9%)
After Unlearning  → ✅ NO PII         (confidence: 99.8%)
```

---

## Key Features

1. **Lightweight** — runs entirely on CPU, no GPU required
2. **Joint optimization** — maximizes loss on forget set (gradient ascent) + minimizes loss on retain set (gradient descent) simultaneously
3. **3-method comparison** — benchmarks against Fine-tune Only and Random Labels baselines
4. **Unlearning Efficacy Score (UES)** — a composite metric combining forget quality, retain quality, and MIA resistance into one score
5. **Live demo** — interactive Gradio web interface

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run the Pipeline

```bash
# Step 1: Generate synthetic PII dataset
python src/1_prepare_data.py

# Step 2: Fine-tune DistilBERT on dataset (~15-20 min on CPU)
python src/2_finetune.py

# Step 3: Apply unlearning and compare 3 methods (~20 min on CPU)
python src/3_unlearn.py

# Step 4: Evaluate results
python src/4_evaluate.py

# Step 5: Generate charts
python src/5_visualize_results.py
```

### Quick sanity test
```bash
python test_quick.py
```

### Live demo
```bash
pip install gradio
python demo.py
# Opens at http://localhost:7860
```

---

## The Algorithm

```
L_total = -α × L_forget + β × L_retain

L_forget = loss on forget set  →  MAXIMIZED (gradient ascent)
L_retain = loss on retain set  →  MINIMIZED (gradient descent)
α = 1.0  |  β = 0.5
```

---

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| Forget Accuracy | ↓ Lower = better | Model should misclassify forgotten PII |
| Retain Accuracy | ↑ Higher = better | Model should stay accurate on clean data |
| MIA Loss | ↑ Higher = better | Harder for attacker to extract PII |
| UES Score | ↑ Higher = better | Overall unlearning quality (composite) |

---

## Project Structure

```
lora-forget/
├── src/
│   ├── 1_prepare_data.py       # Synthetic PII dataset generation
│   ├── 2_finetune.py           # Fine-tune DistilBERT
│   ├── 3_unlearn.py            # Core unlearning — 3 methods compared
│   ├── 4_evaluate.py           # UES + MIA evaluation
│   └── 5_visualize_results.py  # Generate figures
├── data/
│   ├── forget_set.csv          # 100 PII records to forget
│   ├── retain_set.csv          # 400 records to preserve
│   ├── full_dataset.csv
│   └── test_set.csv
├── results/figures/            # Generated charts
├── demo.py                     # Gradio web demo
├── test_quick.py               # Quick test script
├── ablation_study.py           # Hyperparameter analysis
└── requirements.txt
```

---

## Tech Stack

- Python 3.10+
- PyTorch 2.6+
- HuggingFace Transformers (DistilBERT)
- Gradio
- scikit-learn, matplotlib, seaborn

---

## License

MIT
