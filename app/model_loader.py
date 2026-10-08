from pathlib import Path
import re

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Same tokenizer used during BERT training
TOKENIZER_NAME = "google-bert/bert-base-uncased"

# Best checkpoints selected using validation weighted F1
MODEL_PATHS = {
    "Sequential Transfer Learning": (
        "results/models/bert_sequential_target/checkpoint-1314"
    ),
    "Domain Only": ("results/models/bert_domain_only/checkpoint-1314"),
    "Mixed Training": ("results/models/bert_mixed/checkpoint-9448"),
    "Dataset Weighting": ("results/models/bert_dataset_weighting/checkpoint-14172"),
}

EXPECTED_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@st.cache_resource(show_spinner="Loading BERT model...")
def load_model(model_name):
    """Load and cache a trained BERT checkpoint and its tokenizer."""

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model: {model_name}")

    checkpoint_path = PROJECT_ROOT / MODEL_PATHS[model_name]

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_path))

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    if labels != EXPECTED_LABELS:
        raise ValueError(f"Unexpected label mapping for {model_name}: {labels}")

    # Use GPU when available; otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_emotion(text, model_name):
    """Predict the emotion and seven class probabilities."""

    if not text or not text.strip():
        raise ValueError("Please enter financial text.")

    tokenizer, model, device = load_model(model_name)

    # Match the conservative text normalization used in preprocessing
    cleaned_text = re.sub(r"\s+", " ", text).strip()

    # Match training tokenization settings
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Inference only: no training or gradient calculation
    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())

    predicted_label = model.config.id2label[predicted_id]
    confidence = float(probabilities[predicted_id].item())

    class_probabilities = {
        model.config.id2label[i]: float(probabilities[i].item())
        for i in range(model.config.num_labels)
    }

    return {
        "emotion": predicted_label,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }
