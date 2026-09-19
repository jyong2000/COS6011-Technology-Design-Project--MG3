# 14_llm_structured_prompt.py

import json
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "google/flan-t5-base"

LABELS = [
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


def build_prompt(text):
    labels = ", ".join(LABELS)

    return f"""
Classify the emotion in this financial text.

Allowed labels:
{labels}

Financial text:
{text}

Return the answer only in this format:
{{"label": "emotion"}}

Use one of the allowed labels and do not add an explanation.
""".strip()


def get_prediction(response):
    response = response.lower().strip()

    matching_labels = [
        label
        for label in LABELS
        if re.search(rf"\b{re.escape(label)}\b", response)
    ]

    if len(matching_labels) == 1:
        return matching_labels[0]

    return None


def main():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Could not find {TEST_FILE}")

    data = pd.read_csv(TEST_FILE)

    if "text" not in data.columns or "label" not in data.columns:
        raise ValueError("The CSV file must contain text and label columns.")

    data = data.dropna(subset=["text", "label"])

    if TEST_LIMIT is not None:
        data = data.head(TEST_LIMIT)

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    prediction_file = OUTPUT_FOLDER / "14_llm_structured_prompt.jsonl"
    config_file = OUTPUT_FOLDER / "14_llm_structured_prompt_config.json"

    config = {
        "model_name": MODEL_NAME,
        "prompt_type": "structured",
        "labels": LABELS,
        "test_file": str(TEST_FILE),
        "test_limit": TEST_LIMIT,
        "response_format": "{\"label\": \"emotion\"}",
    }

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    with open(prediction_file, "w", encoding="utf-8") as file:
        for number, (_, row) in enumerate(data.iterrows(), start=1):
            text = str(row["text"])
            prompt = build_prompt(text)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                )

            raw_response = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
            ).strip()

            prediction = get_prediction(raw_response)

            result = {
                "row_number": number,
                "text": text,
                "true_label": str(row["label"]),
                "predicted_label": prediction,
                "raw_response": raw_response,
                "prompt_type": "structured",
                "model_name": MODEL_NAME,
            }

            file.write(json.dumps(result) + "\n")

            print(
                f"{number}/{len(data)} | "
                f"Prediction: {prediction} | "
                f"Response: {raw_response}"
            )

    print(f"\nPredictions saved to: {prediction_file}")
    print(f"Configuration saved to: {config_file}")


if __name__ == "__main__":
    main()