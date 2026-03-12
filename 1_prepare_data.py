"""
STEP 1: PREPARE DATA
=====================
What this script does:
- Generates a synthetic dataset of text samples containing PII
  (Personally Identifiable Information: names, emails, phone numbers)
- Labels each sample as either "forget" (data we want the model to unlearn)
  or "retain" (data the model should keep knowing)
- Saves the dataset to the /data folder

Why synthetic data?
- We can't use real people's private data for research
- Synthetic data lets us control exactly what the model learns and forgets
- Using the Faker library to generate realistic-looking but fake PII

This is YOUR dataset — your paper's novel contribution includes
how you constructed and structured this unlearning benchmark.
"""

import pandas as pd
import random
import json
import os
from faker import Faker

# Initialize Faker for generating realistic fake PII
fake = Faker()
Faker.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
NUM_FORGET_SAMPLES = 100   # samples we want the model to FORGET
NUM_RETAIN_SAMPLES = 400   # samples the model should RETAIN (keep knowing)
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# TEMPLATE FUNCTIONS
# Each function generates a realistic sentence
# containing PII in different contexts
# ─────────────────────────────────────────────

def generate_profile_sentence(name, email, phone, city):
    """Generate a sentence that contains multiple PII fields."""
    templates = [
        f"The user {name} can be reached at {email} or by calling {phone}.",
        f"Customer profile: {name}, located in {city}, email: {email}, phone: {phone}.",
        f"Please contact {name} via {email}. Their phone number is {phone}.",
        f"{name} from {city} registered with the email {email} and phone {phone}.",
        f"Account holder {name} provided contact details: {email}, {phone}, {city}.",
        f"User record — Name: {name}, City: {city}, Contact: {email} / {phone}.",
        f"Support ticket raised by {name} ({email}). Callback number: {phone}.",
        f"Subscription confirmed for {name}. Login: {email}. Support line: {phone}.",
    ]
    return random.choice(templates)


def generate_neutral_sentence():
    """Generate a sentence with NO PII — just normal text the model retains."""
    templates = [
        f"The weather in {fake.city()} is expected to be sunny this weekend.",
        f"A new machine learning paper was published on {fake.date()} about neural networks.",
        f"The company reported strong quarterly earnings in the {fake.bs()} sector.",
        f"Researchers at the university developed a new algorithm for image classification.",
        f"The annual conference on artificial intelligence will be held in {fake.city()}.",
        f"Students are encouraged to participate in the upcoming coding hackathon.",
        f"The {fake.color()} model achieved state-of-the-art performance on the benchmark.",
        f"Data privacy regulations continue to evolve across different jurisdictions.",
        f"The team successfully deployed the new recommendation system last week.",
        f"Open-source contributions to the project increased by 40% this year.",
    ]
    return random.choice(templates)


# ─────────────────────────────────────────────
# GENERATE FORGET SET
# These are the records the model must UNLEARN
# In a real scenario, these would be users who
# submitted a "right to be forgotten" GDPR request
# ─────────────────────────────────────────────

print("Generating FORGET set (PII records to be erased)...")
forget_records = []

for i in range(NUM_FORGET_SAMPLES):
    name  = fake.name()
    email = fake.email()
    phone = fake.phone_number()
    city  = fake.city()

    record = {
        "id":        f"forget_{i:04d}",
        "text":      generate_profile_sentence(name, email, phone, city),
        "label":     1,            # 1 = PII-containing text
        "split":     "forget",     # this record must be unlearned
        "name":      name,
        "email":     email,
        "phone":     phone,
        "city":      city,
    }
    forget_records.append(record)

forget_df = pd.DataFrame(forget_records)
print(f"  ✓ Generated {len(forget_df)} forget samples")


# ─────────────────────────────────────────────
# GENERATE RETAIN SET
# Mix of: PII records (different people — keep them)
#         + neutral sentences (no PII)
# The model should keep knowing these after unlearning
# ─────────────────────────────────────────────

print("Generating RETAIN set (records model must keep)...")
retain_records = []

# Half retain = other PII records (different people)
for i in range(NUM_RETAIN_SAMPLES // 2):
    name  = fake.name()
    email = fake.email()
    phone = fake.phone_number()
    city  = fake.city()

    record = {
        "id":    f"retain_pii_{i:04d}",
        "text":  generate_profile_sentence(name, email, phone, city),
        "label": 1,
        "split": "retain",
        "name":  name, "email": email, "phone": phone, "city": city,
    }
    retain_records.append(record)

# Half retain = neutral sentences (no PII)
for i in range(NUM_RETAIN_SAMPLES // 2):
    record = {
        "id":    f"retain_neutral_{i:04d}",
        "text":  generate_neutral_sentence(),
        "label": 0,        # 0 = no PII
        "split": "retain",
        "name":  "", "email": "", "phone": "", "city": "",
    }
    retain_records.append(record)

retain_df = pd.DataFrame(retain_records)
print(f"  ✓ Generated {len(retain_df)} retain samples")


# ─────────────────────────────────────────────
# COMBINE AND SAVE
# ─────────────────────────────────────────────

full_df = pd.concat([forget_df, retain_df], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Save full dataset
full_df.to_csv(f"{OUTPUT_DIR}/full_dataset.csv", index=False)

# Save splits separately (used in fine-tuning and evaluation)
forget_df.to_csv(f"{OUTPUT_DIR}/forget_set.csv",  index=False)
retain_df.to_csv(f"{OUTPUT_DIR}/retain_set.csv",  index=False)

# Save a small test set from retain (used to check model quality after unlearning)
test_df = retain_df.sample(n=50, random_state=42)
test_df.to_csv(f"{OUTPUT_DIR}/test_set.csv", index=False)

# Save metadata
meta = {
    "total_samples":   len(full_df),
    "forget_samples":  len(forget_df),
    "retain_samples":  len(retain_df),
    "test_samples":    len(test_df),
    "label_map":       {"0": "no_PII", "1": "contains_PII"},
    "splits":          {"forget": "records to be unlearned", "retain": "records to be kept"},
    "model_target":    "distilbert-base-uncased",
    "task":            "binary classification (PII detection)",
}
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

# ─────────────────────────────────────────────
# PREVIEW
# ─────────────────────────────────────────────
print("\n✅ Dataset created successfully!")
print(f"   Total samples : {len(full_df)}")
print(f"   Forget set    : {len(forget_df)}")
print(f"   Retain set    : {len(retain_df)}")
print(f"   Test set      : {len(test_df)}")
print(f"\n📂 Files saved to /{OUTPUT_DIR}/")
print("\n🔍 Sample FORGET record:")
print(forget_df[["id","text","split"]].iloc[0].to_string())
print("\n🔍 Sample RETAIN (neutral) record:")
print(retain_df[retain_df["label"]==0][["id","text","split"]].iloc[0].to_string())
