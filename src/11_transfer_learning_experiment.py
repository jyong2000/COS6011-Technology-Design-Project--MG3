# 11_transfer_learning_experiment.py
"""
Purpose:
Implement and evaluate dataset weighting as the team's second
transfer-learning technique.

Method:
- Use the mixed source + target training dataset
- Give target-domain financial samples a higher training weight
- Fine-tune the same pretrained BERT model
- Validate using target_validation
- Evaluate using the same target_test set

Research role:
Second transfer-learning experiment.

Comparison:
Compare dataset weighting against:
- Domain-only BERT
- Sequential Transfer Learning
- Mixed Training

Outputs:
- Evaluation metrics
- Predictions
- Training history
- Experiment notes

Owner:
BERT / Transfer Learning Membe
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

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

MODEL_NAME = "google-bert/bert-base-uncased"

RUN_FULL_TRAINING = True
TEST_MODE = False

NUM_EPOCHS = 1 if TEST_MODE else 3

MIXED_TRAIN_PATH = "data/processed/mixed_train.csv"
TARGET_VALIDATION_PATH = "data/processed/target_validation.csv"
TARGET_TEST_PATH = "data/processed/target_test.csv"

SOURCE_WEIGHT = 1.0
TARGET_WEIGHT = 2.0

# ============================================================
# LOAD DATASETS
# ============================================================

# Load the mixed source + target training dataset.
mixed_train_df = pd.read_csv(MIXED_TRAIN_PATH)

# Load target-domain validation and test datasets.
target_validation_df = pd.read_csv(TARGET_VALIDATION_PATH)
target_test_df = pd.read_csv(TARGET_TEST_PATH)

print("\nOriginal dataset shapes:")
print("Mixed train:", mixed_train_df.shape)
print("Target validation:", target_validation_df.shape)
print("Target test:", target_test_df.shape)
print("Mixed train columns:", mixed_train_df.columns.tolist())

# ============================================================
# ASSIGN DATASET WEIGHTS
# ============================================================

# Give target-domain samples a higher training weight.
mixed_train_df["sample_weight"] = mixed_train_df["domain"].map({
    "source": SOURCE_WEIGHT,
    "target": TARGET_WEIGHT
})

print("\nSample weight distribution:")
print(mixed_train_df["sample_weight"].value_counts())

# ============================================================
# TEST MODE
# ============================================================

# Use small stratified samples to quickly test the pipeline.
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

# ============================================================
# CREATE LABEL MAPPING
# ============================================================

# Use the same fixed 7-class label set as the other BERT experiments.
label_names = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
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

# Convert text labels to integer IDs for BERT training.
mixed_train_df["label"] = mixed_train_df["label"].map(label_to_id)
target_validation_df["label"] = target_validation_df["label"].map(label_to_id)
target_test_df["label"] = target_test_df["label"].map(label_to_id)

# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

# Keep sample_weight in the mixed training dataset because it will
# be used later to calculate the weighted training loss.
mixed_train_dataset = Dataset.from_pandas(
    mixed_train_df[["text", "label", "sample_weight"]],
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

tokenized_mixed_train = mixed_train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_target_validation = target_validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_target_test = target_test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
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
# DEFINE WEIGHTED TRAINER
# ============================================================

# Custom Trainer that applies a different loss weight to each sample.
class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        # Remove sample weights before passing inputs to BERT.
        sample_weights = inputs.pop("sample_weight", None)

        # Get the true class labels.
        labels = inputs["labels"]

        # Run the BERT model.
        outputs = model(**inputs)

        # During validation/test, sample weights are not used.
        if sample_weights is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Calculate loss separately for each sample.
        loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        losses = loss_function(outputs.logits, labels)

        sample_weights = sample_weights.to(
            losses.device,
            dtype=losses.dtype
        )

        # Apply source/target weights and calculate the average loss.
        weighted_loss = (
            (losses * sample_weights).sum()
            / sample_weights.sum()
        )

        return (weighted_loss, outputs) if return_outputs else weighted_loss



# ============================================================
# DEFINE TRAINING ARGUMENTS
# ============================================================

# Configure BERT fine-tuning with dataset weighting.
training_args = TrainingArguments(
    output_dir="results/models/bert_dataset_weighting",
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
    report_to="none",
    remove_unused_columns=False
)

# ============================================================
# CREATE WEIGHTED TRAINER
# ============================================================

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mixed_train,
    eval_dataset=tokenized_target_validation,
    compute_metrics=compute_metrics
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("START DATASET WEIGHTING TRAINING")
print("=" * 70)

training_start_time = time.time()

train_result = trainer.train()

training_end_time = time.time()
training_time_seconds = training_end_time - training_start_time

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

os.makedirs("results/tables", exist_ok=True)

training_history_df = pd.DataFrame(
    trainer.state.log_history
)

training_history_df.to_csv(
    "results/tables/bert_dataset_weighting_training_history.csv",
    index=False
)

# ============================================================
# EVALUATE ON TARGET TEST SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION ON TARGET TEST SET")
print("=" * 70)

test_metrics = trainer.evaluate(
    eval_dataset=tokenized_target_test
)

print("\nFinal target test metrics:")
print(test_metrics)

# ============================================================
# SAVE TEST METRICS
# ============================================================

# Add training time to the final metrics.
test_metrics["training_time_seconds"] = training_time_seconds

# Convert final test metrics into a one-row DataFrame.
metrics_df = pd.DataFrame([test_metrics])

metrics_df.to_csv(
    "results/tables/bert_dataset_weighting_test_metrics.csv",
    index=False
)

# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

prediction_output = trainer.predict(tokenized_target_test)

predicted_ids = np.argmax(
    prediction_output.predictions,
    axis=-1
)

predicted_labels = [
    id_to_label[pred_id]
    for pred_id in predicted_ids
]

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

os.makedirs("results/predictions", exist_ok=True)

prediction_df = pd.DataFrame({
    "text": target_test_df["text"],
    "true_label": [
        id_to_label[label_id]
        for label_id in target_test_df["label"]
    ],
    "predicted_label": predicted_labels
})

prediction_df.to_csv(
    "results/predictions/bert_dataset_weighting_predictions.csv",
    index=False
)

# ============================================================
# SAVE EXPERIMENT NOTES
# ============================================================

experiment_notes = {
    "model": MODEL_NAME,
    "method": "Dataset Weighting",
    "source_weight": SOURCE_WEIGHT,
    "target_weight": TARGET_WEIGHT,
    "epochs": NUM_EPOCHS,
    "test_mode": TEST_MODE
}

experiment_notes_df = pd.DataFrame([experiment_notes])

experiment_notes_df.to_csv(
    "results/tables/bert_dataset_weighting_experiment_notes.csv",
    index=False
)