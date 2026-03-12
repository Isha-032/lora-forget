"""
ABLATION STUDY
===============
Tests different α/β combinations and compares UES scores.
Results go into Table 2 of your paper.

Run: python ablation_study.py
Takes ~30-40 mins on CPU (runs unlearning 4 times)
"""

import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from tqdm import tqdm

os.makedirs("results/ablation", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ──────────────────────────────────────────────
class PIIDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max    = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max,
                       padding="max_length", truncation=True, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         torch.tensor(self.labels[idx], dtype=torch.long)}

# ── Evaluate ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    preds, labels, losses = [], [], []
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            losses.append(out.loss.item())
            preds.extend(torch.argmax(out.logits,1).cpu().numpy())
            labels.extend(b["labels"].numpy())
    return accuracy_score(labels, preds), np.mean(losses)

# ── UES ───────────────────────────────────────────────────
def compute_ues(forget_acc, retain_acc, pre_mia, post_mia):
    fq  = 1 - forget_acc
    rq  = retain_acc
    mia = min(post_mia / (pre_mia + 1e-8), 1.0)
    return round(fq * rq * mia, 4)

# ── Single unlearning run ─────────────────────────────────
def run_unlearning(alpha, beta, forget_loader, retain_loader,
                   test_loader, tokenizer, pre_mia, epochs=5):
    model = DistilBertForSequenceClassification.from_pretrained("models/finetuned")
    model.to(device)
    opt   = AdamW(model.parameters(), lr=1e-5)
    retain_iter = iter(retain_loader)

    for epoch in range(epochs):
        model.train()
        for fb in tqdm(forget_loader, desc=f"  α={alpha} β={beta} epoch {epoch+1}/{epochs}", leave=False):
            opt.zero_grad()
            f_out = model(input_ids=fb["input_ids"].to(device),
                          attention_mask=fb["attention_mask"].to(device),
                          labels=fb["labels"].to(device))
            try:    rb = next(retain_iter)
            except: retain_iter = iter(retain_loader); rb = next(retain_iter)
            r_out = model(input_ids=rb["input_ids"].to(device),
                          attention_mask=rb["attention_mask"].to(device),
                          labels=rb["labels"].to(device))
            loss = -alpha * f_out.loss + beta * r_out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    f_acc, f_loss = evaluate(model, forget_loader)
    r_acc, _      = evaluate(model, retain_loader)
    t_acc, _      = evaluate(model, test_loader)
    ues           = compute_ues(f_acc, r_acc, pre_mia, f_loss)
    return {"alpha": alpha, "beta": beta,
            "forget_acc": round(f_acc,4), "retain_acc": round(r_acc,4),
            "test_acc": round(t_acc,4), "mia_loss": round(f_loss,4), "ues": ues}

# ── Load data ────────────────────────────────────────────
print("📂 Loading data and tokenizer...")
tokenizer    = DistilBertTokenizerFast.from_pretrained("models/finetuned")
forget_df    = pd.read_csv("data/forget_set.csv")
retain_df    = pd.read_csv("data/retain_set.csv")
test_df      = pd.read_csv("data/test_set.csv")
forget_loader = DataLoader(PIIDataset(forget_df, tokenizer), batch_size=16, shuffle=True)
retain_loader = DataLoader(PIIDataset(retain_df, tokenizer), batch_size=16, shuffle=True)
test_loader   = DataLoader(PIIDataset(test_df,   tokenizer), batch_size=16, shuffle=False)

# Pre-unlearning MIA baseline
base_model = DistilBertForSequenceClassification.from_pretrained("models/finetuned").to(device)
_, pre_mia = evaluate(base_model, forget_loader)
del base_model
print(f"   Pre-unlearning MIA loss: {pre_mia:.4f}")

# ── Ablation configurations ───────────────────────────────
configs = [
    (1.0, 0.5),   # original baseline
    (1.0, 0.8),   # more retention
    (1.0, 1.0),   # equal weight
    (2.0, 0.5),   # aggressive forgetting
    (2.0, 1.0),   # aggressive forget + strong retain
]

# ── Run all configs ───────────────────────────────────────
results = []
for alpha, beta in configs:
    print(f"\n🔥 Running: α={alpha}, β={beta}")
    r = run_unlearning(alpha, beta, forget_loader, retain_loader,
                       test_loader, tokenizer, pre_mia)
    results.append(r)
    print(f"   forget_acc={r['forget_acc']} | retain_acc={r['retain_acc']} | UES={r['ues']}")

# ── Save results ─────────────────────────────────────────
with open("results/ablation/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ── Print table ──────────────────────────────────────────
print("\n" + "="*75)
print("  ABLATION TABLE (copy into your paper as Table 2)")
print("="*75)
print(f"{'α':>5} {'β':>5} {'Forget Acc':>12} {'Retain Acc':>12} {'Test Acc':>10} {'MIA Loss':>10} {'UES':>8}")
print("-"*75)
for r in results:
    marker = " ← baseline" if r["alpha"]==1.0 and r["beta"]==0.5 else ""
    print(f"{r['alpha']:>5} {r['beta']:>5} {r['forget_acc']:>12} {r['retain_acc']:>12} {r['test_acc']:>10} {r['mia_loss']:>10} {r['ues']:>8}{marker}")
print("="*75)

# ── Plot ──────────────────────────────────────────────────
BG = "#F8FAFC"
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Ablation Study: Effect of α and β on Unlearning Quality", fontsize=14, fontweight="bold")

labels     = [f"α={r['alpha']}\nβ={r['beta']}" for r in results]
forget_acc = [r["forget_acc"] for r in results]
retain_acc = [r["retain_acc"] for r in results]
ues_scores = [r["ues"]        for r in results]
x = range(len(results))

colors = ["#DC2626","#1A56A0","#16A34A","#D97706","#7C3AED"]

for ax in axes:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].bar(x, forget_acc, color=colors, alpha=0.85)
axes[0].set_title("Forget Accuracy\n(↓ lower = better)", fontweight="bold")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_ylim(0, 1.2)
for i, v in enumerate(forget_acc):
    axes[0].text(i, v+0.02, str(v), ha="center", fontsize=9, fontweight="bold")

axes[1].bar(x, retain_acc, color=colors, alpha=0.85)
axes[1].set_title("Retain Accuracy\n(↑ higher = better)", fontweight="bold")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8)
axes[1].set_ylim(0, 1.2)
for i, v in enumerate(retain_acc):
    axes[1].text(i, v+0.02, str(v), ha="center", fontsize=9, fontweight="bold")

axes[2].bar(x, ues_scores, color=colors, alpha=0.85)
axes[2].set_title("UES Score\n(↑ higher = better overall)", fontweight="bold")
axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8)
axes[2].set_ylim(0, 1.2)
for i, v in enumerate(ues_scores):
    axes[2].text(i, v+0.02, str(v), ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("results/ablation/ablation_chart.png", dpi=180, bbox_inches="tight")
plt.close()

print("\n✅ Ablation study complete!")
print("   Table  → results/ablation/ablation_results.json")
print("   Chart  → results/ablation/ablation_chart.png")
print("\n   Add the chart as Figure 6 in your paper.")
print("   Add the table as Table 2 in Section 5.2 (Ablation Study).")
