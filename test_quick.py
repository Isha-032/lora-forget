from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

tokenizer     = DistilBertTokenizerFast.from_pretrained("models/finetuned")
model_before  = DistilBertForSequenceClassification.from_pretrained("models/finetuned")
model_after   = DistilBertForSequenceClassification.from_pretrained("models/unlearned")

def predict(model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    with torch.no_grad():
        logits = model(**inputs).logits
    prob  = torch.softmax(logits, dim=1)
    pred  = torch.argmax(logits).item()
    label = "CONTAINS PII ⚠️" if pred == 1 else "NO PII ✅"
    return label, prob[0][pred].item()

test_texts = [
    "The user John Smith can be reached at john@gmail.com or by calling 555-1234.",
    "A new machine learning paper was published about neural networks.",
    "Customer profile: Alice Johnson, email: alice@yahoo.com, phone: 555-9876.",
    "Researchers developed a new algorithm for image classification.",
    "Please contact David Lee via david@outlook.com. His phone number is 555-4321.",
    "The annual AI conference will be held in Melbourne this year.",
]

print(f"\n{'Text':<55} {'BEFORE':>18} {'AFTER':>18}")
print("-" * 95)
for text in test_texts:
    b_label, b_conf = predict(model_before, text)
    a_label, a_conf = predict(model_after,  text)
    short = text[:52] + "..."
    print(f"{short:<55} {b_label} ({b_conf:.2f})  →  {a_label} ({a_conf:.2f})")
