"""
15_evaluate_models.py

Purpose:
    Evaluate and compare all trained/prompted models on the same
    target-domain test set.

Models evaluated:
    Traditional ML:
        - Logistic Regression (A/B/C)
        - Linear SVM (A/B/C)
        - Naive Bayes (A/B/C)

    BERT:
        - Domain-only
        - Sequential transfer
        - Mixed training
        - Dataset weighting

    LLM:
        - Zero-shot
        - Few-shot
        - Structured prompt

Outputs:
    results/tables/model_comparison.csv
    results/tables/per_class_metrics.csv
    results/metrics/evaluation_summary.json
    results/figures/model_macro_f1_comparison.png
    results/figures/traditional_ml_comparison.png
    results/figures/bert_strategy_comparison.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTION_DIR = ROOT / "results" / "predictions"
TABLE_DIR = ROOT / "results" / "tables"
METRIC_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
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
# Metadata
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
# Loading and standardisation
# ============================================================

def load_csv_predictions(model_name, filename):
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    df = pd.read_csv(path)

    # Some BERT outputs use "label" instead of "true_label".
    if "true_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "true_label"})

    required = {"true_label", "predicted_label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain true_label and predicted_label. "
            f"Found: {list(df.columns)}"
        )

    # Script 11 (BERT Dataset Weighting) stores labels as numeric IDs.
    label_id_to_name = {
        0: "anger",
        1: "disgust",
        2: "fear",
        3: "happy",
        4: "neutral",
        5: "sad",
        6: "surprise",
    }

    # Only convert columns when they are actually numeric.
    if pd.api.types.is_numeric_dtype(df["true_label"]):
        df["true_label"] = df["true_label"].map(label_id_to_name)

    if pd.api.types.is_numeric_dtype(df["predicted_label"]):
        df["predicted_label"] = df["predicted_label"].map(label_id_to_name)

    df["model_name"] = model_name

    return df


def load_jsonl_predictions(model_name, filename):
    path = PREDICTION_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    required = {"true_label", "predicted_label"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{filename} must contain true_label and predicted_label. "
            f"Found: {list(df.columns)}"
        )

    df["model_name"] = model_name

    return df


# ============================================================
# Validation
# ============================================================

def validate_predictions(df, model_name):
    if len(df) != 1500:
        raise ValueError(
            f"{model_name}: expected 1500 target-test rows, found {len(df)}"
        )

    invalid_true = ~df["true_label"].isin(LABELS)

    if invalid_true.any():
        bad_labels = df.loc[invalid_true, "true_label"].unique()
        raise ValueError(
            f"{model_name}: invalid true labels found: {bad_labels}"
        )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(df, model_name):
    validate_predictions(df, model_name)

    y_true = df["true_label"].astype(str)

    # Missing / unparseable LLM responses count as incorrect.
    y_pred = df["predicted_label"].copy()

    invalid_mask = y_pred.isna() | ~y_pred.isin(LABELS)

    invalid_count = int(invalid_mask.sum())
    invalid_rate = invalid_count / len(df)

    y_pred = y_pred.fillna(INVALID_LABEL).astype(str)

    y_pred.loc[~y_pred.isin(LABELS)] = INVALID_LABEL

    accuracy = accuracy_score(y_true, y_pred)

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    family, strategy = MODEL_INFO[model_name]

    overall = {
        "model": model_name,
        "family": family,
        "strategy": strategy,
        "test_rows": len(df),
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "invalid_predictions": invalid_count,
        "invalid_rate": invalid_rate,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    per_class = []

    for label in LABELS:
        values = report[label]

        per_class.append({
            "model": model_name,
            "family": family,
            "strategy": strategy,
            "label": label,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "support": int(values["support"]),
        })

    return overall, per_class


# ============================================================
# Plotting
# ============================================================

def save_bar_chart(df, title, filename):
    plot_df = df.sort_values("macro_f1", ascending=True)

    plt.figure(figsize=(10, max(6, len(plot_df) * 0.45)))

    plt.barh(
        plot_df["model"],
        plot_df["macro_f1"],
    )

    plt.xlabel("Macro F1")
    plt.ylabel("Model / Strategy")
    plt.title(title)
    plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300)
    plt.close()


def save_traditional_comparison(df):
    traditional = df[df["family"] == "Traditional ML"].copy()

    if traditional.empty:
        return

    pivot = traditional.pivot(
        index="model",
        columns="strategy",
        values="macro_f1",
    )

    pivot.plot(
        kind="bar",
        figsize=(11, 6),
    )

    plt.ylabel("Macro F1")
    plt.xlabel("Model")
    plt.title("Traditional ML: Effect of Training Data")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "traditional_ml_comparison.png",
        dpi=300,
    )

    plt.close()


def save_bert_comparison(df):
    bert = df[df["family"] == "BERT"].copy()

    if bert.empty:
        return

    bert = bert.sort_values("macro_f1")

    plt.figure(figsize=(9, 5))

    plt.barh(
        bert["strategy"],
        bert["macro_f1"],
    )

    plt.xlabel("Macro F1")
    plt.ylabel("Training Strategy")
    plt.title("BERT Training Strategy Comparison")
    plt.xlim(0, 1)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "bert_strategy_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    overall_results = []
    per_class_results = []

    print("=" * 70)
    print("SCRIPT 15 - MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # CSV predictions
    # --------------------------------------------------------

    for model_name, filename in CSV_MODELS.items():
        print(f"\nEvaluating: {model_name}")

        df = load_csv_predictions(model_name, filename)

        overall, per_class = evaluate_model(df, model_name)

        overall_results.append(overall)
        per_class_results.extend(per_class)

        print(
            f"  Accuracy:  {overall['accuracy']:.4f}\n"
            f"  Macro F1:  {overall['macro_f1']:.4f}\n"
            f"  Invalid:   {overall['invalid_predictions']}"
        )

    # --------------------------------------------------------
    # JSONL predictions
    # --------------------------------------------------------

    for model_name, filename in JSONL_MODELS.items():
        print(f"\nEvaluating: {model_name}")

        df = load_jsonl_predictions(model_name, filename)

        overall, per_class = evaluate_model(df, model_name)

        overall_results.append(overall)
        per_class_results.extend(per_class)

        print(
            f"  Accuracy:  {overall['accuracy']:.4f}\n"
            f"  Macro F1:  {overall['macro_f1']:.4f}\n"
            f"  Invalid:   {overall['invalid_predictions']}"
        )

    # --------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------

    overall_df = pd.DataFrame(overall_results)
    per_class_df = pd.DataFrame(per_class_results)

    overall_df = overall_df.sort_values(
        "macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    overall_path = TABLE_DIR / "model_comparison.csv"
    per_class_path = TABLE_DIR / "per_class_metrics.csv"

    overall_df.to_csv(overall_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    summary_path = METRIC_DIR / "evaluation_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            overall_df.to_dict(orient="records"),
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    save_bar_chart(
        overall_df,
        "Target-Domain Macro F1 Comparison",
        "model_macro_f1_comparison.png",
    )

    save_traditional_comparison(overall_df)
    save_bert_comparison(overall_df)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    display_columns = [
        "model",
        "strategy",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "invalid_predictions",
    ]

    print(
        overall_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(f"  {overall_path}")
    print(f"  {per_class_path}")
    print(f"  {summary_path}")
    print(f"  {FIGURE_DIR}")


if __name__ == "__main__":
    main()