# 13_llm_few_shot.py

"""Use a few labelled examples to classify financial text."""

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

TRAIN_FILE = Path("data/processed/target_train.csv")
TEST_FILE = Path("data/processed/target_test.csv")
OUTPUT_FOLDER = Path("results/predictions")


TEST_LIMIT = None


def choose_examples(train_data):
    """Select one training example for each emotion."""
    examples = []

    for label in LABELS:
        matching_rows = train_data[
            train_data["label"].astype(str).str.lower() == label
        ]

        if matching_rows.empty:
            raise ValueError(f"No training example found for label: {label}")

        row = matching_rows.iloc[0]

        examples.append({
            "text": str(row["text"]),
            "label": label,
        })

    return examples


def create_prompt(text, examples):
    """Create the few-shot prompt sent to the model."""
    prompt = [
        "Classify the following financial text into exactly one emotion.",
        "Allowed emotions: " + ", ".join(LABELS),
        "",
        "Here are some labelled examples:",
        "",
    ]

    for example in examples:
        prompt.append(f"Text: {example['text']}")
        prompt.append(f"Emotion: {example['label']}")
        prompt.append("")

    prompt.extend([
        "Classify this new text:",
        f"Text: {text}",
        "Emotion:",
    ])

    return "\n".join(prompt)


def find_label(response):
    """Convert the model's answer into one standard label."""
    response = response.lower().strip()

    found_labels = [
        label
        for label in LABELS
        if re.search(rf"\b{re.escape(label)}\b", response)
    ]

    if len(found_labels) == 1:
        return found_labels[0]

    return None


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_FILE}")

    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing file: {TEST_FILE}")

    train_data = pd.read_csv(TRAIN_FILE)
    test_data = pd.read_csv(TEST_FILE)

    required_columns = {"text", "label"}

    if not required_columns.issubset(train_data.columns):
        raise ValueError("The training file must contain text and label columns.")

    if not required_columns.issubset(test_data.columns):
        raise ValueError("The test file must contain text and label columns.")

    train_data = train_data.dropna(subset=["text", "label"])
    test_data = test_data.dropna(subset=["text", "label"])

    examples = choose_examples(train_data)

    if TEST_LIMIT is not None:
        test_data = test_data.head(TEST_LIMIT)

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    prediction_file = OUTPUT_FOLDER / "13_llm_few_shot.jsonl"
    config_file = OUTPUT_FOLDER / "13_llm_few_shot_config.json"

    configuration = {
        "model_name": MODEL_NAME,
        "prompt_type": "few-shot",
        "labels": LABELS,
        "training_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "test_limit": TEST_LIMIT,
        "examples_source": "target_train.csv",
        "number_of_examples": len(examples),
        "examples": examples,
    }

    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)

    with open(prediction_file, "w", encoding="utf-8") as file:
        for number, (_, row) in enumerate(test_data.iterrows(), start=1):
            text = str(row["text"])
            prompt = create_prompt(text, examples)

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
                    max_new_tokens=10,
                    do_sample=False,
                )

            raw_response = tokenizer.decode(
                output[0],
                skip_special_tokens=True,
            ).strip()

            prediction = find_label(raw_response)

            result = {
                "row_number": number,
                "text": text,
                "true_label": str(row["label"]),
                "predicted_label": prediction,
                "raw_response": raw_response,
                "prompt_type": "few-shot",
                "model_name": MODEL_NAME,
            }

            file.write(json.dumps(result) + "\n")

            print(
                f"{number}/{len(test_data)} | "
                f"Prediction: {prediction} | "
                f"Response: {raw_response}"
            )

    print(f"\nPredictions saved to: {prediction_file}")
    print(f"Configuration saved to: {config_file}")


if __name__ == "__main__":
    main()