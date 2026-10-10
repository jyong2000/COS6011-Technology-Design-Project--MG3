
from pathlib import Path
import os
import re

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Original tokenizer used during training
TOKENIZER_NAME = "google-bert/bert-base-uncased"

# Selected model for deployment
MODEL_NAME = "Dataset Weighting"
LOCAL_CHECKPOINT = (
    PROJECT_ROOT
    / "results/models/bert_dataset_weighting/checkpoint-14172"
)

HF_MODEL_REPO = "jyong2000/financial-emotion-dataset-weighting"

EXPECTED_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


def get_hf_token():
    """Retrieve the private Hugging Face token securely."""
    token = os.environ.get("HF_TOKEN")

    if token:
        return token

    try:
        return st.secrets.get("HF_TOKEN")
    except (FileNotFoundError, KeyError):
        return None


@st.cache_resource(show_spinner="Loading Dataset Weighting BERT model...")
def load_model(model_name=MODEL_NAME):
    """Load the selected model locally or from Hugging Face."""

    if model_name != MODEL_NAME:
        raise ValueError(f"Unsupported model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if LOCAL_CHECKPOINT.is_dir():
        # Use the existing local checkpoint
        model_source = str(LOCAL_CHECKPOINT)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source
        )
    else:
        # Download the private model on Streamlit Cloud
        token = get_hf_token()

        if not token:
            raise RuntimeError(
                "The local checkpoint is unavailable and HF_TOKEN "
                "has not been configured."
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_REPO,
            token=token,
        )

    labels = [
        model.config.id2label[i]
        for i in range(model.config.num_labels)
    ]

    if labels != EXPECTED_LABELS:
        raise ValueError(
            f"Unexpected label mapping: {labels}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict_emotion(text, model_name=MODEL_NAME):
    """Predict the emotion and all seven class probabilities."""

    if not text or not text.strip():
        raise ValueError("Please enter financial text.")

    tokenizer, model, device = load_model(model_name)

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )[0]

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
