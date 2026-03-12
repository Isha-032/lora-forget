"""
STEP 4: COMPREHENSIVE EVALUATION
==================================
What this script does:
- Loads BOTH the fine-tuned model and the unlearned model
- Compares them across 4 evaluation metrics:
    1. Forget Quality  — did the model truly forget the PII?
    2. Retain Quality  — is the model still useful on clean data?
    3. MIA Score       — Membership Inference Attack resistance
    4. Unlearning Efficacy Score (UES) — our composite metric (novel!)
- Saves all results to /results/evaluation_report.json
  (these numbers go directly into your paper's Results table)

The Unlearning Efficacy Score (UES) — your novel metric:
---------------------------------------------------------
UES = (1 - forget_acc) * retain_acc * mia_normalized

Where:
    (1 - forget_acc) = how well it forgot (high = good)
    retain_acc       = how well it retained (high = good)
    mia_normalized   = how resistant to MIA (high = good)

A perfect unlearner would have UES = 1.0
A model that forgot nothing would have UES ≈ 0
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "finetuned_dir":  "models/finetuned",
    "unlearned_dir":  "models/unlearned",
    "data_dir":       "data",
    "results_dir":    "results",
    "max_length":     128,
    "batch_size":     16,
}
os.makedirs(CONFIG["results_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class PIIDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts  = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("📂 Loading data...")
forget_df = pd.read_csv(f"{CONFIG['data_dir']}/forget_set.csv")
retain_df = pd.read_csv(f"{CONFIG['data_dir']}/retain_set.csv")
test_df   = pd.read_csv(f"{CONFIG['data_dir']}/test_set.csv")


# ─────────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────────

def full_evaluate(model, tokenizer, df, label=""):
    dataset = PIIDataset(df, tokenizer, CONFIG["max_length"])
    loader  = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model.eval()
    all_preds, all_labels, all_losses = [], [], []

    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            all_losses.append(out.loss.item())

            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc      = accuracy_score(all_labels, all_preds)
    avg_loss = np.mean(all_losses)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()

    return {
        "accuracy":  round(float(acc), 4),
        "loss":      round(float(avg_loss), 4),
        "precision": round(float(p), 4),
        "recall":    round(float(r), 4),
        "f1":        round(float(f1), 4),
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────
# EVALUATE BOTH MODELS
# ─────────────────────────────────────────────

report = {}

for phase, model_dir in [("finetuned", CONFIG["finetuned_dir"]),
                          ("unlearned", CONFIG["unlearned_dir"])]:

    print(f"\n🤖 Evaluating [{phase}] model from {model_dir}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    report[phase] = {
        "forget_set": full_evaluate(model, tokenizer, forget_df, "Forget"),
        "retain_set": full_evaluate(model, tokenizer, retain_df, "Retain"),
        "test_set":   full_evaluate(model, tokenizer, test_df,   "Test"),
    }

    print(f"   Forget  → acc={report[phase]['forget_set']['accuracy']} | f1={report[phase]['forget_set']['f1']}")
    print(f"   Retain  → acc={report[phase]['retain_set']['accuracy']} | f1={report[phase]['retain_set']['f1']}")
    print(f"   Test    → acc={report[phase]['test_set']['accuracy']}   | f1={report[phase]['test_set']['f1']}")


# ─────────────────────────────────────────────
# COMPUTE UES — Unlearning Efficacy Score
# YOUR NOVEL COMPOSITE METRIC
# ─────────────────────────────────────────────

def compute_ues(forget_acc, retain_acc, pre_mia_loss, post_mia_loss):
    """
    Unlearning Efficacy Score (UES):
    Measures the overall quality of unlearning in one number.

    Components:
        forget_quality = 1 - forget_acc          (lower forget accuracy = better)
        retain_quality = retain_acc               (higher retain accuracy = better)
        mia_resistance = post_mia / pre_mia       (higher post-MIA loss = model forgot = better)

    UES = forget_quality * retain_quality * mia_resistance (capped at 1.0)
    """
    forget_quality = 1 - forget_acc
    retain_quality = retain_acc
    mia_resistance = min(post_mia_loss / (pre_mia_loss + 1e-8), 1.0)

    ues = forget_quality * retain_quality * mia_resistance
    return round(float(ues), 4), {
        "forget_quality": round(float(forget_quality), 4),
        "retain_quality": round(float(retain_quality), 4),
        "mia_resistance": round(float(mia_resistance), 4),
    }

pre_mia  = report["finetuned"]["forget_set"]["loss"]
post_mia = report["unlearned"]["forget_set"]["loss"]

ues_score, ues_components = compute_ues(
    forget_acc = report["unlearned"]["forget_set"]["accuracy"],
    retain_acc = report["unlearned"]["retain_set"]["accuracy"],
    pre_mia_loss  = pre_mia,
    post_mia_loss = post_mia,
)


# ─────────────────────────────────────────────
# DELTA ANALYSIS
# Changes before vs after unlearning
# ─────────────────────────────────────────────

deltas = {
    "forget_acc_delta": round(
        report["unlearned"]["forget_set"]["accuracy"] -
        report["finetuned"]["forget_set"]["accuracy"], 4),
    "retain_acc_delta": round(
        report["unlearned"]["retain_set"]["accuracy"] -
        report["finetuned"]["retain_set"]["accuracy"], 4),
    "test_acc_delta": round(
        report["unlearned"]["test_set"]["accuracy"] -
        report["finetuned"]["test_set"]["accuracy"], 4),
    "mia_loss_delta": round(post_mia - pre_mia, 4),
}


# ─────────────────────────────────────────────
# FINAL SUMMARY TABLE (for your paper)
# ─────────────────────────────────────────────

print("\n" + "="*65)
print("  RESULTS TABLE (copy this into your paper)")
print("="*65)
print(f"{'Metric':<30} {'Before':>10} {'After':>10} {'Delta':>10}")
print("-"*65)
print(f"{'Forget Set Accuracy':<30} {report['finetuned']['forget_set']['accuracy']:>10} {report['unlearned']['forget_set']['accuracy']:>10} {deltas['forget_acc_delta']:>10}")
print(f"{'Retain Set Accuracy':<30} {report['finetuned']['retain_set']['accuracy']:>10} {report['unlearned']['retain_set']['accuracy']:>10} {deltas['retain_acc_delta']:>10}")
print(f"{'Test Set Accuracy':<30} {report['finetuned']['test_set']['accuracy']:>10} {report['unlearned']['test_set']['accuracy']:>10} {deltas['test_acc_delta']:>10}")
print(f"{'MIA Loss (↑ = better)':<30} {pre_mia:>10.4f} {post_mia:>10.4f} {deltas['mia_loss_delta']:>10.4f}")
print("-"*65)
print(f"{'Unlearning Efficacy Score (UES)':<30} {'—':>10} {ues_score:>10} {'':>10}")
print("="*65)
print(f"\nUES Components: {ues_components}")


# ─────────────────────────────────────────────
# SAVE FULL REPORT
# ─────────────────────────────────────────────

full_report = {
    "model_results": report,
    "deltas":        deltas,
    "ues_score":     ues_score,
    "ues_components": ues_components,
    "interpretation": {
        "forget_acc_drop": "Negative delta = model forgot the PII (desired outcome)",
        "retain_acc_stable": "Near-zero delta = model kept its general knowledge (desired)",
        "mia_loss_increase": "Positive delta = PII harder to extract via MIA (desired)",
        "ues": "Closer to 1.0 = better overall unlearning quality",
    }
}

with open(f"{CONFIG['results_dir']}/evaluation_report.json", "w") as f:
    json.dump(full_report, f, indent=2)

print(f"\n✅ Full evaluation report saved to {CONFIG['results_dir']}/evaluation_report.json")
print("\n👉 Next step: Run src/5_visualize_results.py")
