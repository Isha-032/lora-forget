# LoRA-Forget: Lightweight Machine Unlearning for PII Removal in Language Models

---

## What This Project Does

This project implements **Machine Unlearning** — the ability to make a language model
selectively forget specific training data (PII) on demand, without retraining from scratch.

**The Problem:** When a user requests deletion of their data (GDPR "right to be forgotten"),
a model that was trained on that data cannot simply "delete a row." The knowledge is baked
into the model weights.

**Our Solution:** A Gradient Ascent Unlearning pipeline that surgically removes PII from a
fine-tuned DistilBERT model while preserving its general performance.

**Our Novel Contribution:**
1. Lightweight implementation — runs on CPU, no expensive hardware
2. Joint forget + retain optimization with tunable α/β weights
3. New composite metric: **Unlearning Efficacy Score (UES)**
4. Evaluation via simulated Membership Inference Attack (MIA)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (in order)
```bash
# Step 1: Generate synthetic PII dataset
python src/1_prepare_data.py

# Step 2: Fine-tune DistilBERT on dataset (~10-20 min on CPU)
python src/2_finetune.py

# Step 3: Apply machine unlearning (~10-15 min on CPU)
python src/3_unlearn.py

# Step 4: Evaluate both models
python src/4_evaluate.py

# Step 5: Generate paper-ready figures
python src/5_visualize_results.py
```

---

## Project Structure
```
lora-forget/
├── data/
│   ├── full_dataset.csv      # All 500 samples
│   ├── forget_set.csv        # 100 PII records to forget
│   ├── retain_set.csv        # 400 records to keep
│   ├── test_set.csv          # 50 evaluation samples
│   └── metadata.json
├── models/
│   ├── finetuned/            # Model after training on PII
│   └── unlearned/            # Model after unlearning
├── results/
│   ├── evaluation_report.json
│   └── figures/              # 5 paper-ready charts
├── src/
│   ├── 1_prepare_data.py
│   ├── 2_finetune.py
│   ├── 3_unlearn.py          ← CORE CONTRIBUTION
│   ├── 4_evaluate.py
│   └── 5_visualize_results.py
└── requirements.txt
```

---

## The Unlearning Algorithm

```
L_total = -α × L_forget + β × L_retain

Where:
  L_forget = cross-entropy loss on forget set  (MAXIMIZED via gradient ascent)
  L_retain = cross-entropy loss on retain set  (MINIMIZED via gradient descent)
  α = 1.0  (forget weight)
  β = 0.5  (retain weight)
```

---

## Evaluation Metrics

| Metric | Desired Direction | Description |
|--------|------------------|-------------|
| Forget Accuracy | ↓ Lower = better | Model should misclassify forgotten PII |
| Retain Accuracy | ↑ Higher = better | Model should stay accurate on clean data |
| MIA Score (loss) | ↑ Higher = better | Harder for attacker to extract PII |
| **UES** | ↑ Higher = better | Our composite unlearning quality metric |

---
