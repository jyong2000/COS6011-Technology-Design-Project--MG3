# 09_transformer_sequential.py
"""
Purpose:
Perform sequential transfer learning from general-domain emotion data to the
financial-domain dataset.

Method:
1. Start with pretrained BERT
2. Fine-tune using source/general data
3. Continue fine-tuning using target financial data
4. Evaluate on the same target test set

Research role:
Primary transfer-learning experiment.

Outputs:
- Evaluation metrics
- Predictions
- Training history
- Model checkpoint if required

Owner:
BERT / Transfer Learning Member
"""

import os
import numpy as np
import pandas as pd
import torch
import time

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
RUN_FULL_TRAINING = True
TEST_MODE = False
# Use one epoch in test mode and three epochs for full training.
NUM_EPOCHS = 1 if TEST_MODE else 3
# File paths for sequential transfer learning.
SOURCE_TRAIN_PATH = "data/processed/source_train.csv"
SOURCE_VALIDATION_PATH = "data/processed/source_validation.csv"

TARGET_TRAIN_PATH = "data/processed/target_train.csv"
TARGET_VALIDATION_PATH = "data/processed/target_validation.csv"
TARGET_TEST_PATH = "data/processed/target_test.csv"

# ============================================================
# LOAD DATASETS
# ============================================================

# Load source-domain datasets for the first training stage.
source_train_df = pd.read_csv(SOURCE_TRAIN_PATH)
source_validation_df = pd.read_csv(SOURCE_VALIDATION_PATH)

# Load target financial-domain datasets for the second training stage.
target_train_df = pd.read_csv(TARGET_TRAIN_PATH)
target_validation_df = pd.read_csv(TARGET_VALIDATION_PATH)

# Target test data is reserved for final evaluation only.
target_test_df = pd.read_csv(TARGET_TEST_PATH)



# ============================================================
# TEST MODE
# ============================================================

# Use small samples to quickly test the pipeline during development.
if TEST_MODE:
    _, source_train_df = train_test_split(
        source_train_df,
        test_size=100,
        stratify=source_train_df["label"],
        random_state=42
    )

    _, source_validation_df = train_test_split(
        source_validation_df,
        test_size=50,
        stratify=source_validation_df["label"],
        random_state=42
    )

    _, target_train_df = train_test_split(
        target_train_df,
        test_size=100,
        stratify=target_train_df["label"],
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

    print("\nDataset shapes:")
print("Source train:", source_train_df.shape)
print("Source validation:", source_validation_df.shape)
print("Target train:", target_train_df.shape)
print("Target validation:", target_validation_df.shape)
print("Target test:", target_test_df.shape)

# ============================================================
# CREATE LABEL MAPPING
# ============================================================

# Create label mapping using the target-domain label set.
label_names = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]
label_to_id = {
    label: idx
    for idx, label in enumerate(label_names)
}

id_to_label = {
    idx: label
    for label, idx in label_to_id.items()
}

print("\nLabel mapping:")
print(label_to_id)

# ============================================================
# CONVERT LABELS TO INTEGER IDS
# ============================================================

# Convert text labels to integer IDs for all datasets.
source_train_df["label"] = source_train_df["label"].map(label_to_id)
source_validation_df["label"] = source_validation_df["label"].map(label_to_id)

target_train_df["label"] = target_train_df["label"].map(label_to_id)
target_validation_df["label"] = target_validation_df["label"].map(label_to_id)
target_test_df["label"] = target_test_df["label"].map(label_to_id)

# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

# Convert source-domain DataFrames.
source_train_dataset = Dataset.from_pandas(
    source_train_df[["text", "label"]],
    preserve_index=False
)

source_validation_dataset = Dataset.from_pandas(
    source_validation_df[["text", "label"]],
    preserve_index=False
)

# Convert target-domain DataFrames.
target_train_dataset = Dataset.from_pandas(
    target_train_df[["text", "label"]],
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

# Tokenize source-domain datasets.
tokenized_source_train = source_train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_source_validation = source_validation_dataset.map(
    tokenize_function,
    batched=True
)

# Tokenize target-domain datasets.
tokenized_target_train = target_train_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_target_validation = target_validation_dataset.map(
    tokenize_function,
    batched=True
)

tokenized_target_test = target_test_dataset.map(
    tokenize_function,
    batched=True
)

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
# STAGE 1: SOURCE TRAINING ARGUMENTS
# ============================================================

# Configure the first fine-tuning stage on the general-domain data.
source_training_args = TrainingArguments(
    output_dir="results/models/bert_sequential_source",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no" if TEST_MODE else "epoch",
    logging_strategy="epoch",
    load_best_model_at_end=not TEST_MODE,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

# ============================================================
# STAGE 1: CREATE SOURCE TRAINER
# ============================================================

# Create the Trainer for general-domain fine-tuning.
source_trainer = Trainer(
    model=model,
    args=source_training_args,
    train_dataset=tokenized_source_train,
    eval_dataset=tokenized_source_validation,
    compute_metrics=compute_metrics
) 

# ============================================================
# STAGE 1: TRAIN ON SOURCE DATA
# ============================================================

# ============================================================
# STAGE 1: TRAIN ON SOURCE DATA
# ============================================================

# Run source-domain training only when full training is enabled.
if RUN_FULL_TRAINING or TEST_MODE:
    print("\nStarting source-domain fine-tuning...")

    source_training_start_time = time.time()

    source_trainer.train()

    source_training_end_time = time.time()
    source_training_time_seconds = (
        source_training_end_time - source_training_start_time
    )
    

    print("\nSource-domain fine-tuning completed.")


# ============================================================
# STAGE 2: TARGET TRAINING ARGUMENTS
# ============================================================

# Configure the second fine-tuning stage on the financial-domain data.
target_training_args = TrainingArguments(
    output_dir="results/models/bert_sequential_target",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no" if TEST_MODE else "epoch",
    logging_strategy="epoch",
    load_best_model_at_end=not TEST_MODE,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

# ============================================================
# STAGE 2: CREATE TARGET TRAINER
# ============================================================

# Continue fine-tuning the same model on the financial-domain data.
target_trainer = Trainer(
    model=model,
    args=target_training_args,
    train_dataset=tokenized_target_train,
    eval_dataset=tokenized_target_validation,
    compute_metrics=compute_metrics
)

# ============================================================
# STAGE 2: TRAIN ON TARGET DATA
# ============================================================

# Continue training on the target domain only after source training.
if RUN_FULL_TRAINING or TEST_MODE:
    print("\nStarting target-domain fine-tuning...")

    target_training_start_time = time.time()

    target_trainer.train()

    target_training_end_time = time.time()
    target_training_time_seconds = (
        target_training_end_time - target_training_start_time
    )

    print("\nTarget-domain fine-tuning completed.")

# Calculate the total sequential training time.
total_training_time_seconds = (
    source_training_time_seconds + target_training_time_seconds
)


# ============================================================
# FINAL EVALUATION ON TARGET TEST SET
# ============================================================

# Evaluate the final sequential-transfer model on the target test set.
if RUN_FULL_TRAINING or TEST_MODE:
    print("\nEvaluating final model on target test set...")

    test_metrics = target_trainer.evaluate(
        eval_dataset=tokenized_target_test
    )

    print("\nFinal target test metrics:")
    print(test_metrics)

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions on target test set...")

# Generate predictions for the target test dataset.
prediction_output = target_trainer.predict(
    tokenized_target_test
)

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

true_labels = [
    id_to_label[label_id]
    for label_id in target_test_df["label"]
]

predictions_df = pd.DataFrame({
    "text": target_test_df["text"].values,
    "true_label": true_labels,
    "predicted_label": predicted_labels
})

# ============================================================
# SAVE PREDICTIONS
# ============================================================

# Create the output directory if it does not already exist.
os.makedirs("results/predictions", exist_ok=True)

# Save test predictions for later error analysis.
predictions_df.to_csv(
    "results/predictions/bert_sequential_predictions.csv",
    index=False
)

print("\nPredictions saved successfully.")

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================
# Create the output directory if it does not already exist.
os.makedirs("results/tables", exist_ok=True)

# Save the training history from the source-domain fine-tuning stage.
source_history_df = pd.DataFrame(
    source_trainer.state.log_history
)

source_history_df.to_csv(
    "results/tables/bert_sequential_source_training_history.csv",
    index=False
)
# Save the training history from the target-domain fine-tuning stage.
target_history_df = pd.DataFrame(
    target_trainer.state.log_history
)

target_history_df.to_csv(
    "results/tables/bert_sequential_target_training_history.csv",
    index=False
)

print("\nTraining history saved successfully.")

# ============================================================
# SAVE TEST METRICS
# ============================================================
# Add training times to the final metrics.
test_metrics["source_training_time_seconds"] = source_training_time_seconds
test_metrics["target_training_time_seconds"] = target_training_time_seconds
test_metrics["total_training_time_seconds"] = total_training_time_seconds

# Convert test metrics to a DataFrame.
test_metrics_df = pd.DataFrame([test_metrics])

# Create the output directory if it does not already exist.
os.makedirs("results/tables", exist_ok=True)

# Save the final target test metrics.
test_metrics_df.to_csv(
    "results/tables/bert_sequential_test_metrics.csv",
    index=False
)

print("\nTest metrics saved successfully.")