# 08_transformer_domain_only.py
"""
08_transformer_domain_only.py

Purpose:
Fine-tune the base BERT model using only the target financial-domain dataset.

Method:
- Base model: google-bert/bert-base-uncased
- Train on target_train
- Validate on target_validation
- Evaluate on target_test

Research role:
Represents the domain-only Transformer experiment.

Outputs:
- Evaluation metrics
- Target test predictions
- Model training history
- Saved checkpoint if required

Owner:
BERT / Transfer Learning Member
"""


# ============================================================
# IMPORT LIBRARIES
# ============================================================

# Provides tools for working with file and folder paths.
import os

# Used for numerical operations and arrays.
import numpy as np

# Used to load and work with CSV datasets.
import pandas as pd

# PyTorch is the deep learning framework used to run BERT.
import torch

# Converts pandas DataFrames into Hugging Face Dataset objects.
from datasets import Dataset

# Used to calculate model evaluation metrics.
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Hugging Face tools for tokenization, BERT classification,
# model training, and training configuration.
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Pretrained BERT model that will be fine-tuned for
# financial-domain sentiment/emotion classification.
MODEL_NAME = "google-bert/bert-base-uncased"


# ============================================================
# DATA PATHS
# ============================================================

# Folder containing the cleaned and prepared datasets.
PROCESSED_DATA_DIR = "data/processed"

# Target-domain training dataset used to fine-tune BERT.
TARGET_TRAIN_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_train.csv"
)

# Target-domain validation dataset used during model development.
TARGET_VALIDATION_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_validation.csv"
)

# Target-domain test dataset reserved for final evaluation.
TARGET_TEST_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_test.csv"
)


# ============================================================
# LOAD TARGET-DOMAIN DATASETS
# ============================================================

# Load the financial-domain training dataset.
train_df = pd.read_csv(TARGET_TRAIN_PATH)

# Load the financial-domain validation dataset.
validation_df = pd.read_csv(TARGET_VALIDATION_PATH)

# Load the financial-domain test dataset.
test_df = pd.read_csv(TARGET_TEST_PATH)


# ============================================================
# CHECK LOADED DATA
# ============================================================

# Display the number of rows and columns in each dataset.
print("Train shape:", train_df.shape)
print("Validation shape:", validation_df.shape)
print("Test shape:", test_df.shape)

# Display column names to identify the text and label columns.
print("\nColumns:")
print(train_df.columns)

# Display the first five training examples.
print("\nTraining data sample:")
print(train_df.head())

# ============================================================
# CREATE LABEL MAPPING
# ============================================================

# Get all unique labels from the target training dataset.
label_names = sorted(train_df["label"].unique())

# Convert label names to integer IDs.
label_to_id = {
    label: idx
    for idx, label in enumerate(label_names)
}

# Convert integer IDs back to readable label names.
id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}

print("\nLabel mapping:")
print(label_to_id)

# ============================================================
# CONVERT LABELS TO INTEGER IDS
# ============================================================

# Convert text labels to integer IDs for BERT training.
train_df["label"] = train_df["label"].map(label_to_id)
validation_df["label"] = validation_df["label"].map(label_to_id)
test_df["label"] = test_df["label"].map(label_to_id)

# Display a sample to verify the conversion.
print("\nTraining data after label conversion:")
print(train_df.head())

# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

# Convert pandas DataFrames into Hugging Face Dataset objects.
train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)

# Check the converted datasets.
print("\nHugging Face datasets:")
print(train_dataset)
print(validation_dataset)
print(test_dataset)

# ============================================================
# LOAD TOKENIZER
# ============================================================

# Load the tokenizer that matches the pretrained BERT model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTokenizer loaded successfully.")

# ============================================================
# TOKENIZE DATASETS
# ============================================================

# Tokenize a batch of text for BERT.
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Test the tokenizer on one example from the training dataset.
sample_text = train_dataset[0]["text"]

sample_tokens = tokenizer(
    sample_text,
    truncation=True,
    max_length=128
)

print("\nOriginal text:")
print(sample_text)

print("\nToken IDs:")
print(sample_tokens["input_ids"])

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(sample_tokens["input_ids"]))

# Apply tokenization to all datasets.
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenization completed.")
print(tokenized_train_dataset)

# ============================================================
# LOAD BERT MODEL
# ============================================================

# Load pretrained BERT for 7-class sentiment classification.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id_to_label,
    label2id=label_to_id
)

print("\nBERT model loaded successfully.")
print("Number of sentiment classes:", len(label_names))

# ============================================================
# DEFINE EVALUATION METRICS
# ============================================================

# Calculate evaluation metrics from model predictions.
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================
# DEFINE TRAINING ARGUMENTS
# ============================================================

# Configure how BERT will be fine-tuned.
training_args = TrainingArguments(
    output_dir="results/models/bert_domain_only",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

print("\nTraining arguments created successfully.")

# ============================================================
# CREATE TRAINER
# ============================================================

# Connect the model, training settings, datasets, and metrics.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    compute_metrics=compute_metrics
)

print("\nTrainer created successfully.")

# ============================================================
# TRAIN MODEL
# ============================================================

# Training has already been completed.
# The saved best checkpoint will be used for final evaluation.

# print("\nStarting BERT fine-tuning...")
# train_result = trainer.train()
# print("\nTraining completed successfully.")

# ============================================================
# LOAD BEST TRAINED CHECKPOINT
# ============================================================

print("\n" + "=" * 70)
print("LOAD BEST TRAINED CHECKPOINT")
print("=" * 70)

# Load the best model saved during training.
BEST_CHECKPOINT = "results/models/bert_domain_only/checkpoint-1314"

trainer.model = AutoModelForSequenceClassification.from_pretrained(
    BEST_CHECKPOINT
)

print("Best checkpoint loaded successfully.")
print("Checkpoint:", BEST_CHECKPOINT)

# ============================================================
# FINAL EVALUATION ON TARGET TEST SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON TARGET TEST SET")
print("=" * 70)

# Evaluate the best trained model on the target test dataset.
test_metrics = trainer.evaluate(
    eval_dataset=tokenized_test_dataset
)

print("\nFinal target test metrics:")
print(test_metrics)

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATE TEST PREDICTIONS")
print("=" * 70)

# Generate predictions for the target test dataset.
prediction_output = trainer.predict(tokenized_test_dataset)

# Select the class with the highest prediction score.
predicted_ids = np.argmax(
    prediction_output.predictions,
    axis=-1
)

# Convert numeric predictions back to readable labels.
predicted_labels = [
    id_to_label[prediction_id]
    for prediction_id in predicted_ids
]

# Create a table containing the text, true label, and prediction.
predictions_df = pd.DataFrame({
    "text": test_df["text"],
    "true_label": [
        id_to_label[label_id]
        for label_id in test_df["label"]
    ],
    "predicted_label": predicted_labels
})

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

# Create the predictions folder if it does not already exist.
os.makedirs("results/predictions", exist_ok=True)

# Save test predictions for later comparison.
predictions_df.to_csv(
    "results/predictions/bert_domain_only_predictions.csv",
    index=False
)

print("Test predictions saved successfully.")

# ============================================================
# SAVE EVALUATION METRICS
# ============================================================

print("\n" + "=" * 70)
print("SAVE EVALUATION METRICS")
print("=" * 70)

# Create the results table folder if it does not already exist.
os.makedirs("results/tables", exist_ok=True)

# Convert final test metrics into a one-row DataFrame.
metrics_df = pd.DataFrame([test_metrics])

# Save final test metrics for later model comparison.
metrics_df.to_csv(
    "results/tables/bert_domain_only_test_metrics.csv",
    index=False
)

print("Evaluation metrics saved successfully.")