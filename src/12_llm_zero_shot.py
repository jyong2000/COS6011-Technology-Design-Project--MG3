# 12_llm_zero_shot.py

"""Classify financial text using zero-shot prompting with FLAN-T5."""

import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-base"

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

TEST_FILE = Path("data/processed/target_test.csv")
OUTPUT_FOLDER = Path("results/predictions")


TEST_LIMIT = None


def create_prompt(text):
    labels = ", ".join(EMOTION_LABELS)

    return (
        "Classify the following financial text into exactly one emotion.\n"
        f"Allowed emotions: {labels}\n"
        "Return only the emotion name. Do not give an explanation.\n\n"
        f"Financial text: {text}\n"
        "Emotion:"
    )


def standardise_label(answer):
    answer = answer.lower().strip()

    matching_labels = [
        label
        for label in EMOTION_LABELS
        if re.search(rf"\b{re.escape(label)}\b", answer)
    ]

    if len(matching_labels) == 1:
        return matching_labels[0]

    return None


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {TEST_FILE}")

    data = pd.read_csv(TEST_FILE)

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    data = data.dropna(subset=["text"])

    if TEST_LIMIT is not None:
        data = data.head(TEST_LIMIT)

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    prediction_file = OUTPUT_FOLDER / "12_llm_zero_shot.jsonl"
    config_file = OUTPUT_FOLDER / "12_llm_zero_shot_config.json"

    configuration = {
        "model_name": MODEL_NAME,
        "prompt_type": "zero-shot",
        "labels": EMOTION_LABELS,
        "test_file": str(TEST_FILE),
        "test_limit": TEST_LIMIT,
        "temperature": 0,
        "examples_used": False,
    }

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)

    with open(prediction_file, "w", encoding="utf-8") as file:
        for number, (_, row) in enumerate(data.iterrows(), start=1):
            financial_text = str(row["text"])
            prompt = create_prompt(financial_text)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            inputs = {
                name: value.to(device)
                for name, value in inputs.items()
            }

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )

            raw_answer = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
            ).strip()

            predicted_label = standardise_label(raw_answer)

            result = {
                "row_number": number,
                "text": financial_text,
                "true_label": str(row["label"]),
                "predicted_label": predicted_label,
                "raw_response": raw_answer,
                "prompt_type": "zero-shot",
                "model_name": MODEL_NAME,
            }

            file.write(json.dumps(result) + "\n")

            print(
                f"{number}/{len(data)} | "
                f"Prediction: {predicted_label} | "
                f"Response: {raw_answer}"
            )

    print("\nPredictions saved to:")
    print(prediction_file)

    print("\nConfiguration saved to:")
    print(config_file)


if __name__ == "__main__":
    main()