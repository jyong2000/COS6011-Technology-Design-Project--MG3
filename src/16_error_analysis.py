"""
16_error_analysis.py

Purpose:
    Perform post-hoc error analysis on predictions produced by Scripts 05-14.

Analyses:
    1. Per-class error rates
    2. Most common confusion pairs
    3. Prediction distributions
    4. Traditional ML domain-shift comparison (A/B/C)
    5. BERT training-strategy comparison
    6. LLM invalid/unparseable responses
    7. Representative classification errors
    8. Cross-model disagreement

No models are trained in this script.

Outputs:
    results/error_analysis/
        class_error_summary.csv
        confusion_pairs.csv
        prediction_distribution.csv
        domain_shift_summary.csv
        bert_error_comparison.csv
        llm_invalid_responses.csv
        representative_errors.csv
        model_disagreement.csv
        error_analysis_summary.json

    results/figures/
        class_recall_comparison.png
        top_confusion_pairs.png
        prediction_distribution.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "results" / "predictions"
ERROR_DIR = ROOT / "results" / "error_analysis"
FIGURE_DIR = ROOT / "results" / "figures"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

INVALID_LABEL = "__invalid__"

EXPECTED_TEST_ROWS = 1500


# ============================================================
# Prediction files
# ============================================================

CSV_MODELS = {
    # Traditional ML
    "Logistic Regression A": "logistic_regression_A_target_test.csv",
    "Logistic Regression B": "logistic_regression_B_target_test.csv",
    "Logistic Regression C": "logistic_regression_C_target_test.csv",

    "Linear SVM A": "linear_svm_A_target_test.csv",
    "Linear SVM B": "linear_svm_B_target_test.csv",
    "Linear SVM C": "linear_svm_C_target_test.csv",

    "Naive Bayes A": "naive_bayes_A_target_test.csv",
    "Naive Bayes B": "naive_bayes_B_target_test.csv",
    "Naive Bayes C": "naive_bayes_C_target_test.csv",

    # BERT
    "BERT Domain-only": "bert_domain_only_predictions.csv",
    "BERT Sequential": "bert_sequential_predictions.csv",
    "BERT Mixed": "bert_mixed_predictions.csv",
    "BERT Dataset Weighting": "bert_dataset_weighting_predictions.csv",
}

JSONL_MODELS = {
    "FLAN-T5 Zero-shot": "12_llm_zero_shot.jsonl",
    "FLAN-T5 Few-shot": "13_llm_few_shot.jsonl",
    "FLAN-T5 Structured": "14_llm_structured_prompt.jsonl",
}


# ============================================================
# Model metadata
# ============================================================

MODEL_INFO = {
    "Logistic Regression A": ("Traditional ML", "Source-only"),
    "Logistic Regression B": ("Traditional ML", "Target-only"),
    "Logistic Regression C": ("Traditional ML", "Mixed"),

    "Linear SVM A": ("Traditional ML", "Source-only"),
    "Linear SVM B": ("Traditional ML", "Target-only"),
    "Linear SVM C": ("Traditional ML", "Mixed"),

    "Naive Bayes A": ("Traditional ML", "Source-only"),
    "Naive Bayes B": ("Traditional ML", "Target-only"),
    "Naive Bayes C": ("Traditional ML", "Mixed"),

    "BERT Domain-only": ("BERT", "Target-only"),
    "BERT Sequential": ("BERT", "Sequential transfer"),
    "BERT Mixed": ("BERT", "Mixed"),
    "BERT Dataset Weighting": ("BERT", "Dataset weighting"),

    "FLAN-T5 Zero-shot": ("LLM", "Zero-shot"),
    "FLAN-T5 Few-shot": ("LLM", "Few-shot"),
    "FLAN-T5 Structured": ("LLM", "Structured prompt"),
}


# ============================================================
# Label conversion
# ============================================================

LABEL_ID_TO_NAME = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}


# ============================================================
# Loading
# ============================================================

def standardise_dataframe(df, model_name):
    """
    Convert prediction outputs from different scripts into a common format.
    """

    # Some BERT outputs use "label" instead of "true_label".
    if "true_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "true_label"})

    required = {"true_label", "predicted_label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{model_name}: prediction file must contain "
            f"true_label and predicted_label. "
            f"Found: {list(df.columns)}"
        )

    # Convert numeric BERT labels back into emotion names.
    if pd.api.types.is_numeric_dtype(df["true_label"]):
        df["true_label"] = df["true_label"].map(LABEL_ID_TO_NAME)

    if pd.api.types.is_numeric_dtype(df["predicted_label"]):
        df["predicted_label"] = df["predicted_label"].map(
            LABEL_ID_TO_NAME
        )

    # Make sure a text column exists.
    if "text" not in df.columns:
        df["text"] = ""

    # Preserve the original prediction before invalid values are replaced.
    df["original_predicted_label"] = df["predicted_label"]

    # Identify invalid / missing predictions.
    invalid_mask = (
        df["predicted_label"].isna()
        | ~df["predicted_label"].isin(LABELS)
    )

    df["invalid_prediction"] = invalid_mask

    # Invalid LLM outputs are kept as an explicit error category.
    df.loc[invalid_mask, "predicted_label"] = INVALID_LABEL

    family, strategy = MODEL_INFO[model_name]

    df["model"] = model_name
    df["family"] = family
    df["strategy"] = strategy

    df["correct"] = (
        df["true_label"] == df["predicted_label"]
    )

    return df


def load_csv_predictions(model_name, filename):
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    df = pd.read_csv(path)

    return standardise_dataframe(df, model_name)


def load_jsonl_predictions(model_name, filename):
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing prediction file: {path}"
        )

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    return standardise_dataframe(df, model_name)


def validate_dataframe(df, model_name):
    if len(df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"{model_name}: expected {EXPECTED_TEST_ROWS} rows, "
            f"found {len(df)}"
        )

    invalid_true = ~df["true_label"].isin(LABELS)

    if invalid_true.any():
        bad = df.loc[
            invalid_true,
            "true_label"
        ].unique()

        raise ValueError(
            f"{model_name}: invalid true labels: {bad}"
        )


def load_all_predictions():
    predictions = {}

    for model_name, filename in CSV_MODELS.items():
        df = load_csv_predictions(
            model_name,
            filename,
        )

        validate_dataframe(df, model_name)
        predictions[model_name] = df

    for model_name, filename in JSONL_MODELS.items():
        df = load_jsonl_predictions(
            model_name,
            filename,
        )

        validate_dataframe(df, model_name)
        predictions[model_name] = df

    return predictions


# ============================================================
# 1. Per-class error analysis
# ============================================================

def analyse_class_errors(predictions):
    rows = []

    for model_name, df in predictions.items():

        y_true = df["true_label"]
        y_pred = df["predicted_label"]

        precision, recall, f1, support = (
            precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=LABELS,
                zero_division=0,
            )
        )

        family, strategy = MODEL_INFO[model_name]

        for i, label in enumerate(LABELS):

            class_rows = df[
                df["true_label"] == label
            ]

            correct_count = int(
                class_rows["correct"].sum()
            )

            error_count = (
                len(class_rows) - correct_count
            )

            error_rate = (
                error_count / len(class_rows)
                if len(class_rows) > 0
                else 0
            )

            rows.append({
                "model": model_name,
                "family": family,
                "strategy": strategy,
                "label": label,
                "support": int(support[i]),
                "correct": correct_count,
                "errors": error_count,
                "error_rate": error_rate,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
            })

    return pd.DataFrame(rows)


# ============================================================
# 2. Confusion-pair analysis
# ============================================================

def analyse_confusion_pairs(predictions):
    rows = []

    for model_name, df in predictions.items():

        errors = df[~df["correct"]].copy()

        grouped = (
            errors
            .groupby(
                ["true_label", "predicted_label"],
                dropna=False,
            )
            .size()
            .reset_index(name="count")
        )

        family, strategy = MODEL_INFO[model_name]

        for _, row in grouped.iterrows():

            true_label = row["true_label"]
            predicted_label = row["predicted_label"]
            count = int(row["count"])

            true_total = int(
                (df["true_label"] == true_label).sum()
            )

            rate_within_true_class = (
                count / true_total
                if true_total > 0
                else 0
            )

            rows.append({
                "model": model_name,
                "family": family,
                "strategy": strategy,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": count,
                "rate_within_true_class":
                    rate_within_true_class,
            })

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["model", "count"],
        ascending=[True, False],
    )


# ============================================================
# 3. Prediction-distribution analysis
# ============================================================

def analyse_prediction_distribution(predictions):
    rows = []

    prediction_labels = LABELS + [INVALID_LABEL]

    for model_name, df in predictions.items():

        family, strategy = MODEL_INFO[model_name]

        true_counts = (
            df["true_label"]
            .value_counts()
            .to_dict()
        )

        predicted_counts = (
            df["predicted_label"]
            .value_counts()
            .to_dict()
        )

        total = len(df)

        for label in prediction_labels:

            true_count = (
                true_counts.get(label, 0)
                if label in LABELS
                else 0
            )

            predicted_count = predicted_counts.get(
                label,
                0,
            )

            rows.append({
                "model": model_name,
                "family": family,
                "strategy": strategy,
                "label": label,
                "true_count": true_count,
                "predicted_count": predicted_count,
                "true_rate": true_count / total,
                "predicted_rate":
                    predicted_count / total,
                "prediction_minus_true_rate":
                    (
                        predicted_count
                        - true_count
                    ) / total,
            })

    return pd.DataFrame(rows)


# ============================================================
# 4. Traditional ML domain-shift analysis
# ============================================================

def analyse_domain_shift(class_errors):
    traditional = class_errors[
        class_errors["family"]
        == "Traditional ML"
    ].copy()

    # Extract base classifier name.
    traditional["classifier"] = (
        traditional["model"]
        .str.replace(
            r" [ABC]$",
            "",
            regex=True,
        )
    )

    output = traditional[
        [
            "classifier",
            "model",
            "strategy",
            "label",
            "support",
            "error_rate",
            "precision",
            "recall",
            "f1",
        ]
    ].copy()

    return output.sort_values(
        ["classifier", "label", "strategy"]
    )


# ============================================================
# 5. BERT strategy analysis
# ============================================================

def analyse_bert_strategies(class_errors):
    bert = class_errors[
        class_errors["family"] == "BERT"
    ].copy()

    return bert[
        [
            "model",
            "strategy",
            "label",
            "support",
            "error_rate",
            "precision",
            "recall",
            "f1",
        ]
    ].sort_values(
        ["label", "model"]
    )


# ============================================================
# 6. LLM invalid-response analysis
# ============================================================

def analyse_llm_invalid(predictions):
    rows = []

    for model_name, df in predictions.items():

        if MODEL_INFO[model_name][0] != "LLM":
            continue

        invalid = df[
            df["invalid_prediction"]
        ].copy()

        for idx, row in invalid.iterrows():

            item = {
                "model": model_name,
                "strategy":
                    MODEL_INFO[model_name][1],
                "row_index": int(idx),
                "text": row.get("text", ""),
                "true_label":
                    row["true_label"],
                "predicted_label":
                    row["predicted_label"],
            }

            # Preserve raw generated response if available.
            if "raw_response" in df.columns:
                item["raw_response"] = row.get(
                    "raw_response",
                    "",
                )

            if "prompt_type" in df.columns:
                item["prompt_type"] = row.get(
                    "prompt_type",
                    "",
                )

            rows.append(item)

    return pd.DataFrame(rows)


# ============================================================
# 7. Representative classification errors
# ============================================================

def extract_representative_errors(
    predictions,
    examples_per_model=10,
):
    """
    Select reproducible representative errors.

    Preference is given to the most common confusion pairs
    for each model. The first available examples from those
    pairs are retained.

    This avoids randomly selecting errors.
    """

    rows = []

    for model_name, df in predictions.items():

        errors = df[
            (~df["correct"])
            & (~df["invalid_prediction"])
        ].copy()

        if errors.empty:
            continue

        confusion_counts = (
            errors
            .groupby(
                ["true_label", "predicted_label"]
            )
            .size()
            .reset_index(name="pair_count")
            .sort_values(
                "pair_count",
                ascending=False,
            )
        )

        selected = 0

        for _, pair in confusion_counts.iterrows():

            pair_examples = errors[
                (
                    errors["true_label"]
                    == pair["true_label"]
                )
                &
                (
                    errors["predicted_label"]
                    == pair["predicted_label"]
                )
            ]

            for idx, example in pair_examples.iterrows():

                rows.append({
                    "model": model_name,
                    "family":
                        MODEL_INFO[model_name][0],
                    "strategy":
                        MODEL_INFO[model_name][1],
                    "row_index": int(idx),
                    "text": example.get(
                        "text",
                        "",
                    ),
                    "true_label":
                        example["true_label"],
                    "predicted_label":
                        example["predicted_label"],
                    "confusion_pair_count":
                        int(pair["pair_count"]),
                })

                selected += 1

                if selected >= examples_per_model:
                    break

            if selected >= examples_per_model:
                break

    return pd.DataFrame(rows)


# ============================================================
# 8. Cross-model disagreement
# ============================================================

def analyse_model_disagreement(predictions):
    """
    Compare predictions row-by-row across all 16 experiments.

    This assumes all prediction files preserve the same
    target-test ordering, which is checked using true labels
    and text where possible.
    """

    model_names = list(predictions.keys())

    reference_name = model_names[0]
    reference = predictions[
        reference_name
    ].reset_index(drop=True)

    # Validate row alignment.
    for model_name in model_names[1:]:

        current = predictions[
            model_name
        ].reset_index(drop=True)

        if not reference[
            "true_label"
        ].equals(current["true_label"]):

            raise ValueError(
                f"Target-test row order differs between "
                f"{reference_name} and {model_name}."
            )

    rows = []

    for i in range(len(reference)):

        model_predictions = {
            model_name:
                predictions[model_name]
                .reset_index(drop=True)
                .loc[i, "predicted_label"]
            for model_name in model_names
        }

        values = list(
            model_predictions.values()
        )

        unique_predictions = len(set(values))

        true_label = reference.loc[
            i,
            "true_label",
        ]

        correct_models = sum(
            pred == true_label
            for pred in values
        )

        incorrect_models = (
            len(model_names) - correct_models
        )

        row = {
            "row_index": i,
            "text": reference.loc[
                i,
                "text",
            ],
            "true_label": true_label,
            "unique_predictions":
                unique_predictions,
            "correct_models":
                correct_models,
            "incorrect_models":
                incorrect_models,
        }

        for model_name, prediction in (
            model_predictions.items()
        ):
            row[model_name] = prediction

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        [
            "unique_predictions",
            "incorrect_models",
        ],
        ascending=[False, False],
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    predictions,
    class_errors,
    confusion_pairs,
):
    summary = {
        "models_analysed":
            len(predictions),

        "test_rows_per_model":
            EXPECTED_TEST_ROWS,

        "labels":
            LABELS,

        "llm_invalid_predictions": {},

        "lowest_recall_class_by_model": {},

        "top_confusion_by_model": {},
    }

    # LLM invalid counts.
    for model_name, df in predictions.items():

        if MODEL_INFO[model_name][0] == "LLM":

            count = int(
                df[
                    "invalid_prediction"
                ].sum()
            )

            summary[
                "llm_invalid_predictions"
            ][model_name] = {
                "count": count,
                "rate": count / len(df),
            }

    # Lowest recall class.
    for model_name in predictions:

        subset = class_errors[
            class_errors["model"]
            == model_name
        ]

        worst = subset.sort_values(
            "recall"
        ).iloc[0]

        summary[
            "lowest_recall_class_by_model"
        ][model_name] = {
            "label": worst["label"],
            "recall": float(
                worst["recall"]
            ),
        }

    # Most common confusion.
    for model_name in predictions:

        subset = confusion_pairs[
            confusion_pairs["model"]
            == model_name
        ]

        if subset.empty:
            continue

        top = subset.sort_values(
            "count",
            ascending=False,
        ).iloc[0]

        summary[
            "top_confusion_by_model"
        ][model_name] = {
            "true_label":
                top["true_label"],
            "predicted_label":
                top["predicted_label"],
            "count":
                int(top["count"]),
            "rate_within_true_class":
                float(
                    top[
                        "rate_within_true_class"
                    ]
                ),
        }

    return summary


# ============================================================
# Figures
# ============================================================

def plot_class_recall(class_errors):
    """
    Plot recall by emotion for the four BERT strategies.
    """

    bert = class_errors[
        class_errors["family"] == "BERT"
    ].copy()

    pivot = bert.pivot(
        index="label",
        columns="strategy",
        values="recall",
    )

    pivot = pivot.reindex(LABELS)

    ax = pivot.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        "BERT Recall by Emotion and Training Strategy"
    )
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "class_recall_comparison.png",
        dpi=300,
    )

    plt.close()


def plot_top_confusions(confusion_pairs):
    """
    Plot the most common confusion pairs across all models.
    """

    valid = confusion_pairs[
        confusion_pairs["predicted_label"]
        != INVALID_LABEL
    ].copy()

    grouped = (
        valid
        .groupby(
            ["true_label", "predicted_label"]
        )["count"]
        .sum()
        .reset_index()
    )

    grouped["pair"] = (
        grouped["true_label"]
        + " → "
        + grouped["predicted_label"]
    )

    top = (
        grouped
        .sort_values(
            "count",
            ascending=False,
        )
        .head(12)
        .sort_values("count")
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top["pair"],
        top["count"],
    )

    plt.xlabel(
        "Total Errors Across Models"
    )
    plt.ylabel(
        "True → Predicted"
    )
    plt.title(
        "Most Common Confusion Pairs"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "top_confusion_pairs.png",
        dpi=300,
    )

    plt.close()


def plot_prediction_distribution(
    prediction_distribution,
):
    """
    Compare true target-test distribution against selected
    representative model predictions.
    """

    selected_models = [
        "Linear SVM B",
        "BERT Domain-only",
        "BERT Mixed",
        "BERT Dataset Weighting",
        "FLAN-T5 Zero-shot",
    ]

    subset = prediction_distribution[
        prediction_distribution[
            "model"
        ].isin(selected_models)
        &
        prediction_distribution[
            "label"
        ].isin(LABELS)
    ].copy()

    pivot = subset.pivot(
        index="label",
        columns="model",
        values="predicted_rate",
    )

    pivot = pivot.reindex(LABELS)

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7),
    )

    ax.set_title(
        "Prediction Distribution Across Selected Models"
    )
    ax.set_xlabel("Emotion")
    ax.set_ylabel(
        "Proportion of Test Predictions"
    )
    ax.set_ylim(0, 1)

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "prediction_distribution.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SCRIPT 16 - ERROR ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load predictions
    # --------------------------------------------------------

    print("\nLoading prediction files...")

    predictions = load_all_predictions()

    print(
        f"Loaded {len(predictions)} model outputs."
    )

    # --------------------------------------------------------
    # 1. Class-level errors
    # --------------------------------------------------------

    print(
        "\n1. Analysing per-class errors..."
    )

    class_errors = analyse_class_errors(
        predictions
    )

    class_errors.to_csv(
        ERROR_DIR
        / "class_error_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Confusion pairs
    # --------------------------------------------------------

    print(
        "2. Analysing confusion pairs..."
    )

    confusion_pairs = (
        analyse_confusion_pairs(
            predictions
        )
    )

    confusion_pairs.to_csv(
        ERROR_DIR
        / "confusion_pairs.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Prediction distributions
    # --------------------------------------------------------

    print(
        "3. Analysing prediction distributions..."
    )

    prediction_distribution = (
        analyse_prediction_distribution(
            predictions
        )
    )

    prediction_distribution.to_csv(
        ERROR_DIR
        / "prediction_distribution.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Domain shift
    # --------------------------------------------------------

    print(
        "4. Analysing traditional ML domain shift..."
    )

    domain_shift = analyse_domain_shift(
        class_errors
    )

    domain_shift.to_csv(
        ERROR_DIR
        / "domain_shift_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 5. BERT strategies
    # --------------------------------------------------------

    print(
        "5. Analysing BERT strategies..."
    )

    bert_errors = (
        analyse_bert_strategies(
            class_errors
        )
    )

    bert_errors.to_csv(
        ERROR_DIR
        / "bert_error_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 6. LLM invalid responses
    # --------------------------------------------------------

    print(
        "6. Extracting invalid LLM responses..."
    )

    llm_invalid = analyse_llm_invalid(
        predictions
    )

    llm_invalid.to_csv(
        ERROR_DIR
        / "llm_invalid_responses.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Representative errors
    # --------------------------------------------------------

    print(
        "7. Extracting representative errors..."
    )

    representative_errors = (
        extract_representative_errors(
            predictions,
            examples_per_model=10,
        )
    )

    representative_errors.to_csv(
        ERROR_DIR
        / "representative_errors.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Model disagreement
    # --------------------------------------------------------

    print(
        "8. Analysing cross-model disagreement..."
    )

    disagreement = (
        analyse_model_disagreement(
            predictions
        )
    )

    disagreement.to_csv(
        ERROR_DIR
        / "model_disagreement.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        predictions,
        class_errors,
        confusion_pairs,
    )

    with open(
        ERROR_DIR
        / "error_analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print(
        "9. Creating figures..."
    )

    plot_class_recall(
        class_errors
    )

    plot_top_confusions(
        confusion_pairs
    )

    plot_prediction_distribution(
        prediction_distribution
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        "\nLowest-recall emotion by model:"
    )

    for model_name in predictions:

        subset = class_errors[
            class_errors["model"]
            == model_name
        ]

        worst = subset.sort_values(
            "recall"
        ).iloc[0]

        print(
            f"  {model_name:<28} "
            f"{worst['label']:<10} "
            f"Recall={worst['recall']:.4f}"
        )

    print(
        "\nLLM invalid predictions:"
    )

    for model_name in JSONL_MODELS:

        df = predictions[model_name]

        invalid_count = int(
            df[
                "invalid_prediction"
            ].sum()
        )

        invalid_rate = (
            invalid_count / len(df)
        )

        print(
            f"  {model_name:<28} "
            f"{invalid_count:>4} "
            f"({invalid_rate:.2%})"
        )

    print("\nSaved outputs to:")
    print(f"  {ERROR_DIR}")
    print(f"  {FIGURE_DIR}")

    print(
        "\nScript 16 completed successfully."
    )


if __name__ == "__main__":
    main()