"""
STEP 3: MACHINE UNLEARNING — 3 METHODS COMPARED
==================================================
This is your CORE RESEARCH CONTRIBUTION.

We implement and compare 3 unlearning methods:

METHOD A — Fine-tune Only (Baseline 1)
    Standard approach: just retrain on retain set only.
    Model "forgets" by catastrophic forgetting.
    Simple but no targeted forgetting.

METHOD B — Random Labels (Baseline 2)
    Replace forget-set labels with random labels, then train.
    Confuses the model about forgotten data.
    Used in NeurIPS Machine Unlearning Competition as a baseline.

METHOD C — Gradient Ascent + Retain (OURS)
    Simultaneously maximize loss on forget set (gradient ascent)
    and minimize loss on retain set (gradient descent).
    Our proposed method with tunable alpha/beta weights.

Reference baselines from:
    Graves et al. (2021), Golatkar et al. (2020),
    NeurIPS Machine Unlearning Challenge (2023),
    TOFU Benchmark: Maini et al. (2024) arXiv:2401.06121
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
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

CONFIG = {
    "finetuned_dir":  "models/finetuned",
    "data_dir":       "data",
    "max_length":     128,
    "batch_size":     16,
    "epochs":         2,
    "learning_rate":  1e-5,
    "alpha":          1.0,
    "beta":           0.5,
    "seed":           42,
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
for d in ["models/method_a", "models/method_b", "models/method_c", "results"]:
    os.makedirs(d, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")


class PIIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, random_labels=False):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        if random_labels:
            self.labels = [np.random.randint(0, 2) for _ in self.labels]
        self.tok = tokenizer
        self.max = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(self.labels[idx], dtype=torch.long)}


def evaluate(model, loader, label=""):
    model.eval()
    preds, labels, losses = [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            losses.append(out.loss.item())
            preds.extend(torch.argmax(out.logits, 1).cpu().numpy())
            labels.extend(b["labels"].numpy())
    acc = accuracy_score(labels, preds)
    if label:
        print(f"   {label:<12}: acc={acc:.4f} | loss={np.mean(losses):.4f}")
    return acc, float(np.mean(losses))


def compute_ues(forget_acc, retain_acc, pre_mia, post_mia):
    fq  = 1 - forget_acc
    rq  = retain_acc
    mia = min(post_mia / (pre_mia + 1e-8), 1.0)
    return round(fq * rq * mia, 4)


# ── Load data ────────────────────────────────────────────
print("Loading data...")
tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG["finetuned_dir"])
forget_df = pd.read_csv(f"{CONFIG['data_dir']}/forget_set.csv")
retain_df = pd.read_csv(f"{CONFIG['data_dir']}/retain_set.csv")
test_df   = pd.read_csv(f"{CONFIG['data_dir']}/test_set.csv")

forget_loader    = DataLoader(PIIDataset(forget_df, tokenizer),                    batch_size=CONFIG["batch_size"], shuffle=True)
retain_loader    = DataLoader(PIIDataset(retain_df, tokenizer),                    batch_size=CONFIG["batch_size"], shuffle=True)
test_loader      = DataLoader(PIIDataset(test_df,   tokenizer),                    batch_size=CONFIG["batch_size"])
forget_rl_loader = DataLoader(PIIDataset(forget_df, tokenizer, random_labels=True),batch_size=CONFIG["batch_size"], shuffle=True)

# Pre-unlearning baseline
base = DistilBertForSequenceClassification.from_pretrained(CONFIG["finetuned_dir"]).to(device)
print("\nBaseline (original fine-tuned model):")
pre_f_acc, pre_mia = evaluate(base, forget_loader, "Forget")
pre_r_acc, _       = evaluate(base, retain_loader, "Retain")
pre_t_acc, _       = evaluate(base, test_loader,   "Test")
del base

# ═══ METHOD A ════════════════════════════════════════════
print("\n" + "="*50)
print("METHOD A: Fine-tune Only (Baseline 1)")
print("="*50)
model_a = DistilBertForSequenceClassification.from_pretrained(CONFIG["finetuned_dir"]).to(device)
opt_a   = AdamW(model_a.parameters(), lr=CONFIG["learning_rate"])
for ep in range(CONFIG["epochs"]):
    model_a.train()
    for b in tqdm(retain_loader, desc=f"  A ep{ep+1}"):
        opt_a.zero_grad()
        model_a(input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)).loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        opt_a.step()
a_f_acc, a_mia = evaluate(model_a, forget_loader, "Forget")
a_r_acc, _     = evaluate(model_a, retain_loader, "Retain")
a_t_acc, _     = evaluate(model_a, test_loader,   "Test")
a_ues           = compute_ues(a_f_acc, a_r_acc, pre_mia, a_mia)
model_a.save_pretrained("models/method_a"); tokenizer.save_pretrained("models/method_a")
print(f"  UES={a_ues}")

# ═══ METHOD B ════════════════════════════════════════════
print("\n" + "="*50)
print("METHOD B: Random Labels (Baseline 2)")
print("="*50)
model_b = DistilBertForSequenceClassification.from_pretrained(CONFIG["finetuned_dir"]).to(device)
opt_b   = AdamW(model_b.parameters(), lr=CONFIG["learning_rate"])
for ep in range(CONFIG["epochs"]):
    model_b.train()
    for b in tqdm(forget_rl_loader, desc=f"  B forget(random) ep{ep+1}"):
        opt_b.zero_grad()
        model_b(input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)).loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()
    for b in tqdm(retain_loader, desc=f"  B retain ep{ep+1}"):
        opt_b.zero_grad()
        model_b(input_ids=b["input_ids"].to(device),
                attention_mask=b["attention_mask"].to(device),
                labels=b["labels"].to(device)).loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()
b_f_acc, b_mia = evaluate(model_b, forget_loader, "Forget")
b_r_acc, _     = evaluate(model_b, retain_loader, "Retain")
b_t_acc, _     = evaluate(model_b, test_loader,   "Test")
b_ues           = compute_ues(b_f_acc, b_r_acc, pre_mia, b_mia)
model_b.save_pretrained("models/method_b"); tokenizer.save_pretrained("models/method_b")
print(f"  UES={b_ues}")

# ═══ METHOD C (OURS) ══════════════════════════════════════
print("\n" + "="*50)
print(f"METHOD C: GA+Retain (OURS) alpha={CONFIG['alpha']} beta={CONFIG['beta']}")
print("="*50)
model_c     = DistilBertForSequenceClassification.from_pretrained(CONFIG["finetuned_dir"]).to(device)
opt_c       = AdamW(model_c.parameters(), lr=CONFIG["learning_rate"])
retain_iter = iter(retain_loader)
for ep in range(CONFIG["epochs"]):
    model_c.train()
    for fb in tqdm(forget_loader, desc=f"  C ep{ep+1}"):
        opt_c.zero_grad()
        f_out = model_c(input_ids=fb["input_ids"].to(device),
                        attention_mask=fb["attention_mask"].to(device),
                        labels=fb["labels"].to(device))
        try:    rb = next(retain_iter)
        except: retain_iter = iter(retain_loader); rb = next(retain_iter)
        r_out = model_c(input_ids=rb["input_ids"].to(device),
                        attention_mask=rb["attention_mask"].to(device),
                        labels=rb["labels"].to(device))
        loss = -CONFIG["alpha"] * f_out.loss + CONFIG["beta"] * r_out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
        opt_c.step()
c_f_acc, c_mia = evaluate(model_c, forget_loader, "Forget")
c_r_acc, _     = evaluate(model_c, retain_loader, "Retain")
c_t_acc, _     = evaluate(model_c, test_loader,   "Test")
c_ues           = compute_ues(c_f_acc, c_r_acc, pre_mia, c_mia)
model_c.save_pretrained("models/method_c"); tokenizer.save_pretrained("models/method_c")
print(f"  UES={c_ues}")

# ═══ COMPARISON TABLE ════════════════════════════════════
print("\n" + "="*75)
print("  COMPARISON TABLE — Table 1 in your paper")
print("="*75)
print(f"{'Method':<35} {'Forget Acc':>10} {'Retain Acc':>10} {'Test Acc':>9} {'MIA Loss':>9} {'UES':>7}")
print("-"*75)
rows = [
    ("Original (no unlearning)",      pre_f_acc, pre_r_acc, pre_t_acc, pre_mia, 0.0),
    ("Method A: Fine-tune Only",      a_f_acc,   a_r_acc,   a_t_acc,   a_mia,   a_ues),
    ("Method B: Random Labels",       b_f_acc,   b_r_acc,   b_t_acc,   b_mia,   b_ues),
    ("Method C: GA+Retain (Ours) *",  c_f_acc,   c_r_acc,   c_t_acc,   c_mia,   c_ues),
]
for name, fa, ra, ta, mia, ues in rows:
    print(f"{name:<35} {fa:>10.4f} {ra:>10.4f} {ta:>9.4f} {mia:>9.4f} {ues:>7.4f}")
print("="*75)
print("* = proposed method  |  Forget Acc: lower is better  |  others: higher is better")

results = {
    "original": {"forget_acc": pre_f_acc, "retain_acc": pre_r_acc, "test_acc": pre_t_acc, "mia": pre_mia, "ues": 0.0},
    "method_a": {"forget_acc": a_f_acc,   "retain_acc": a_r_acc,   "test_acc": a_t_acc,   "mia": a_mia,   "ues": a_ues},
    "method_b": {"forget_acc": b_f_acc,   "retain_acc": b_r_acc,   "test_acc": b_t_acc,   "mia": b_mia,   "ues": b_ues},
    "method_c": {"forget_acc": c_f_acc,   "retain_acc": c_r_acc,   "test_acc": c_t_acc,   "mia": c_mia,   "ues": c_ues},
    "config":   CONFIG,
}
with open("results/comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nAll 3 methods saved.")
print("Results -> results/comparison_results.json")
print("\nNext: Run src/5_visualize_results.py")
