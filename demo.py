"""
DEMO — Gradio Web App
Run: python demo.py
Opens at http://localhost:7860
Screenshot this for your paper!
"""
import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer    = DistilBertTokenizerFast.from_pretrained("models/finetuned")
model_before = DistilBertForSequenceClassification.from_pretrained("models/finetuned")
model_after  = DistilBertForSequenceClassification.from_pretrained("models/unlearned")

def analyze(text):
    if not text.strip():
        return "Please enter some text.", "Please enter some text."
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding="max_length")
    with torch.no_grad():
        b_logits = model_before(**inputs).logits
        a_logits = model_after(**inputs).logits

    b_pred = torch.argmax(b_logits).item()
    a_pred = torch.argmax(a_logits).item()
    b_conf = torch.softmax(b_logits, dim=1)[0][b_pred].item()
    a_conf = torch.softmax(a_logits, dim=1)[0][a_pred].item()

    b_out = f"{'⚠️ CONTAINS PII' if b_pred==1 else '✅ NO PII'}  (confidence: {b_conf:.1%})"
    a_out = f"{'⚠️ CONTAINS PII' if a_pred==1 else '✅ NO PII'}  (confidence: {a_conf:.1%})"
    return b_out, a_out

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=3, placeholder="Type any sentence with a name, email, phone number..."),
    outputs=[
        gr.Textbox(label="🔴 Before Unlearning — Model still remembers PII"),
        gr.Textbox(label="🟢 After Unlearning  — Model has forgotten PII"),
    ],
    title="LoRA-Forget: Machine Unlearning Live Demo",
    description="Type a sentence and see how the model's PII detection changes after unlearning. "
                "PII sentences should flip from CONTAINS PII → NO PII after unlearning.",
    examples=[
        ["The user John Smith can be reached at john@gmail.com or 555-1234."],
        ["Researchers published a new paper on neural network optimization."],
        ["Account holder Alice Wong, email: alice@company.com, phone: 555-9999."],
        ["The conference on machine learning will be held in Sydney next year."],
    ],
    theme=gr.themes.Soft()
)

demo.launch()
