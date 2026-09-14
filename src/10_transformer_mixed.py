# 10_transformer_mixed.py
"""
Purpose:
Fine-tune BERT using a mixed source + target training dataset.

Method:
- Start with the same pretrained BERT model
- Train using mixed_train
- Validate using target_validation
- Evaluate using target_test

Research role:
Tests whether combining general and domain-specific training data improves
financial-domain performance.

Outputs:
- Evaluation metrics
- Predictions
- Training history

Owner:
BERT / Transfer Learning Member
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Use the same pretrained BERT model as the other experiments.
MODEL_NAME = "google-bert/bert-base-uncased"

# Keep full training disabled while developing the script.
RUN_FULL_TRAINING = False

# Use a small dataset sample for quick testing.
TEST_MODE = True

# Use one epoch in test mode and three epochs for full training.
NUM_EPOCHS = 1 if TEST_MODE else 3

# File paths for mixed training and target-domain evaluation.
MIXED_TRAIN_PATH = "data/processed/mixed_train.csv"
TARGET_VALIDATION_PATH = "data/processed/target_validation.csv"
TARGET_TEST_PATH = "data/processed/target_test.csv"

# ============================================================
# LOAD DATASETS
# ============================================================

# Load the mixed source + target training dataset.
mixed_train_df = pd.read_csv(MIXED_TRAIN_PATH)

# Load target-domain validation and test datasets.
target_validation_df = pd.read_csv(TARGET_VALIDATION_PATH)
target_test_df = pd.read_csv(TARGET_TEST_PATH)

# ============================================================
# TEST MODE
# ============================================================

# Use small samples to quickly test the pipeline during development.
if TEST_MODE:
    _, mixed_train_df = train_test_split(
        mixed_train_df,
        test_size=100,
        stratify=mixed_train_df["label"],
        random_state=42
    )

    _, target_validation_df = train_test_split(
        target_validation_df,
        test_size=50,
        stratify=target_validation_df["label"],
        random_state=42
    )

    _, target_test_df = train_test_split(
        target_test_df,
        test_size=50,
        stratify=target_test_df["label"],
        random_state=42
    )

    print("\nTEST MODE enabled: using stratified dataset samples.")

# Display dataset sizes after applying test mode.
print("\nDataset shapes:")
print("Mixed train:", mixed_train_df.shape)
print("Target validation:", target_validation_df.shape)
print("Target test:", target_test_df.shape)

# ============================================================
# CREATE LABEL MAPPING
# ============================================================

# Use the fixed 7-class target-domain label set for all experiments.
label_names = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

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
mixed_train_df["label"] = mixed_train_df["label"].map(label_to_id)
target_validation_df["label"] = target_validation_df["label"].map(label_to_id)
target_test_df["label"] = target_test_df["label"].map(label_to_id)

# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

# Convert pandas DataFrames into Hugging Face Dataset objects.
mixed_train_dataset = Dataset.from_pandas(
    mixed_train_df[["text", "label"]],
    preserve_index=False
)

target_validation_dataset = Dataset.from_pandas(
    target_validation_df[["text", "label"]],
    preserve_index=False
)

target_test_dataset = Dataset.from_pandas(
    target_test_df[["text", "label"]],
    preserve_index=False
)

# ============================================================
# LOAD TOKENIZER
# ============================================================

# Load the tokenizer that matches the pretrained BERT model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ============================================================
# TOKENIZE DATASETS
# ============================================================

# Tokenize text and prepare fixed-length inputs for BERT.
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# ============================================================
# APPLY TOKENIZATION
# ============================================================

# Tokenize the mixed training dataset.
tokenized_mixed_train = mixed_train_dataset.map(
    tokenize_function,
    batched=True
)

# Tokenize the target validation dataset.
tokenized_target_validation = target_validation_dataset.map(
    tokenize_function,
    batched=True
)

# Tokenize the target test dataset.
tokenized_target_test = target_test_dataset.map(
    tokenize_function,
    batched=True
)

# ============================================================
# LOAD BERT MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOAD BERT MODEL")
print("=" * 70)

# Load pretrained BERT for 7-class sentiment classification.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_names),
    id2label=id_to_label,
    label2id=label_to_id
)

print("BERT model loaded successfully.")
print("Number of sentiment classes:", len(label_names))

# ============================================================
# DEFINE EVALUATION METRICS
# ============================================================

# Calculate evaluation metrics from model predictions.
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Select the class with the highest prediction score.
    predictions = np.argmax(logits, axis=-1)

    # Calculate weighted precision, recall, and F1-score.
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    # Calculate overall accuracy.
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

# Configure BERT fine-tuning on the mixed training dataset.
training_args = TrainingArguments(
    output_dir="results/models/bert_mixed",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no" if TEST_MODE else "epoch",
    load_best_model_at_end=not TEST_MODE,
    logging_strategy="epoch",
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

# ============================================================
# CREATE TRAINER
# ============================================================

# Connect the model, training settings, mixed training data,
# target validation data, and evaluation metrics.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mixed_train,
    eval_dataset=tokenized_target_validation,
    compute_metrics=compute_metrics
)

# ============================================================
# TRAIN MODEL
# ============================================================

# Run mixed-domain training in full training mode or test mode.
if RUN_FULL_TRAINING or TEST_MODE:
    print("\n" + "=" * 70)
    print("MIXED-DOMAIN BERT TRAINING")
    print("=" * 70)

    print("\nStarting mixed-domain fine-tuning...")
    train_result = trainer.train()

    print("\nMixed-domain fine-tuning completed.")

# ============================================================
# FINAL EVALUATION ON TARGET TEST SET
# ============================================================

# Evaluate the trained mixed-domain model on the target test set.
if RUN_FULL_TRAINING or TEST_MODE:
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TARGET TEST SET")
    print("=" * 70)

    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_target_test
    )

    print("\nFinal target test metrics:")
    print(test_metrics)

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

if RUN_FULL_TRAINING or TEST_MODE:
    print("\n" + "=" * 70)
    print("SAVE TEST PREDICTIONS")
    print("=" * 70)

    # Generate predictions for the target test dataset.
    prediction_output = trainer.predict(tokenized_target_test)

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
        "text": target_test_df["text"],
        "true_label": [
            id_to_label[label_id]
            for label_id in target_test_df["label"]
        ],
        "predicted_label": predicted_labels
    })

    # Create the output directory if it does not already exist.
    os.makedirs("results/predictions", exist_ok=True)

    # Save test predictions for later comparison.
    predictions_df.to_csv(
        "results/predictions/bert_mixed_predictions.csv",
        index=False
    )

    print("Test predictions saved successfully.")
    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVE TRAINING HISTORY")
    print("=" * 70)
    # Create the output directory if it does not already exist.
    os.makedirs("results/tables", exist_ok=True)

    # Convert the Trainer log history into a DataFrame.
    history_df = pd.DataFrame(
        trainer.state.log_history
    )

    # Save training history for later analysis.
    history_df.to_csv(
        "results/tables/bert_mixed_training_history.csv",
        index=False
    )

    print("Training history saved successfully.")


    # ========================================================
    # SAVE EVALUATION METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print("SAVE EVALUATION METRICS")
    print("=" * 70)

    # Convert final test metrics into a one-row DataFrame.
    metrics_df = pd.DataFrame([test_metrics])

    # Save final evaluation metrics for model comparison.
    metrics_df.to_csv(
        "results/tables/bert_mixed_test_metrics.csv",
        index=False
    )

    print("Evaluation metrics saved successfully.")