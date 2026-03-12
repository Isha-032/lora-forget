"""
STEP 2: FINE-TUNE DistilBERT
==============================
What this script does:
- Loads DistilBERT (a small, fast version of BERT — perfect for CPU)
- Fine-tunes it on our PII dataset so it "learns" to recognize
  and recall personal information from the forget set
- Saves the fine-tuned model checkpoint

After this step, the model "knows" the PII.
Step 3 will make it forget specific records.

Why DistilBERT?
- 40% smaller than BERT, 60% faster, 97% of BERT's accuracy
- Runs on CPU in reasonable time (~10-20 mins for this dataset)
- Widely cited in research — reviewers respect it
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
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "model_name":    "distilbert-base-uncased",
    "max_length":    128,
    "batch_size":    16,
    "epochs":        3,
    "learning_rate": 2e-5,
    "num_labels":    2,
    "output_dir":    "models/finetuned",
    "data_dir":      "data",
    "seed":          42,
}

torch.manual_seed(CONFIG["seed"])
os.makedirs(CONFIG["output_dir"], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")


# ─────────────────────────────────────────────
# DATASET CLASS
# Converts our CSV rows into tensors for PyTorch
# ─────────────────────────────────────────────

class PIIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts  = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("\n📂 Loading dataset...")
full_df = pd.read_csv(f"{CONFIG['data_dir']}/full_dataset.csv")
test_df = pd.read_csv(f"{CONFIG['data_dir']}/test_set.csv")

# Training uses the full dataset (forget + retain combined)
# This simulates: model was originally trained on ALL user data
train_df = full_df.copy()

print(f"   Train samples : {len(train_df)}")
print(f"   Test samples  : {len(test_df)}")
print(f"   Label distribution:\n{train_df['label'].value_counts().to_string()}")


# ─────────────────────────────────────────────
# TOKENIZER + MODEL
# ─────────────────────────────────────────────

print(f"\n🤖 Loading {CONFIG['model_name']}...")
tokenizer = DistilBertTokenizerFast.from_pretrained(CONFIG["model_name"])
model     = DistilBertForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=CONFIG["num_labels"],
)
model.to(device)
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ─────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────

train_dataset = PIIDataset(train_df, tokenizer, CONFIG["max_length"])
test_dataset  = PIIDataset(test_df,  tokenizer, CONFIG["max_length"])

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=CONFIG["batch_size"], shuffle=False)


# ─────────────────────────────────────────────
# OPTIMIZER + SCHEDULER
# ─────────────────────────────────────────────

optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
total_steps = len(train_loader) * CONFIG["epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader)
    return avg_loss, acc, all_preds, all_labels


print(f"\n🚀 Starting fine-tuning for {CONFIG['epochs']} epochs...")
history = []

for epoch in range(CONFIG["epochs"]):
    model.train()
    total_train_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

    for batch in progress:
        optimizer.zero_grad()

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / len(train_loader)
    val_loss, val_acc, _, _ = evaluate(model, test_loader)

    epoch_stats = {
        "epoch":      epoch + 1,
        "train_loss": round(avg_train_loss, 4),
        "val_loss":   round(val_loss, 4),
        "val_acc":    round(val_acc, 4),
    }
    history.append(epoch_stats)
    print(f"\n   Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")


# ─────────────────────────────────────────────
# FINAL EVALUATION
# ─────────────────────────────────────────────

print("\n📊 Final evaluation on test set:")
_, final_acc, preds, labels = evaluate(model, test_loader)
print(classification_report(labels, preds, target_names=["No PII", "Contains PII"]))


# ─────────────────────────────────────────────
# SAVE MODEL + RESULTS
# ─────────────────────────────────────────────

model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

results = {
    "phase":          "finetuned",
    "final_accuracy": round(final_acc, 4),
    "training_history": history,
    "config":         CONFIG,
}
with open(f"{CONFIG['output_dir']}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Model saved to {CONFIG['output_dir']}/")
print(f"   Final accuracy: {final_acc:.4f}")
print("\n👉 Next step: Run src/3_unlearn.py")
