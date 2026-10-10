from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions" / "dashboard"

LABELS = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FILES = {
    "Domain Only": "bert_domain_only.csv",
    "Sequential Transfer Learning": "bert_sequential.csv",
    "Mixed Training": "bert_mixed.csv",
    "Dataset Weighting": "bert_dataset_weighting.csv",
}

METRICS = [
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro-F1",
    "Weighted-F1",
]


@st.cache_data
def load_predictions():
    datasets = {}

    for model_name, filename in FILES.items():
        path = PREDICTIONS_DIR / filename

        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction file: {path}")

        df = pd.read_csv(path)

        required = {"text", "true_label", "predicted_label"}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns in {filename}")

        if df[list(required)].isna().any().any():
            raise ValueError(f"Missing values in {filename}")

        for column in ("true_label", "predicted_label"):
            if not df[column].isin(LABELS).all():
                raise ValueError(f"Unexpected emotion labels in {filename}: {column}")

        datasets[model_name] = df

    # Verify that every model was evaluated on the same
    # locked test examples, in the same order.
    reference = next(iter(datasets.values()))

    for name, df in datasets.items():
        if not df[["text", "true_label"]].equals(reference[["text", "true_label"]]):
            raise ValueError(f"{name} does not match the locked Target Test set.")

    return datasets


def calculate_metrics(datasets):
    rows = []

    for model_name, df in datasets.items():
        y_true = df["true_label"]
        y_pred = df["predicted_label"]

        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_true, y_pred),
                "Macro Precision": precision_score(
                    y_true, y_pred, labels=LABELS, average="macro", zero_division=0
                ),
                "Macro Recall": recall_score(
                    y_true, y_pred, labels=LABELS, average="macro", zero_division=0
                ),
                "Macro-F1": f1_score(
                    y_true, y_pred, labels=LABELS, average="macro", zero_division=0
                ),
                "Weighted-F1": f1_score(
                    y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Model")


def show_dashboard():
    st.header("Model Comparison Dashboard")
    st.write(
        "Evaluation of four BERT training strategies "
        "using the same locked financial Target Test set."
    )

    try:
        datasets = load_predictions()
        metrics_df = calculate_metrics(datasets)
    except Exception as error:
        st.error(f"Unable to load evaluation results: {error}")
        return

    st.caption(
        f"Test examples: {len(next(iter(datasets.values()))):,} "
        "| Primary metric: Macro-F1"
    )

    # Overall comparison
    st.subheader("Overall Model Performance")

    st.dataframe(
        metrics_df.style.format("{:.2%}").highlight_max(axis=0, color="#d9ead3"),
        width="stretch",
    )

    best_model = metrics_df["Macro-F1"].idxmax()
    best_score = metrics_df.loc[best_model, "Macro-F1"]

    st.info(f"Highest BERT test Macro-F1: {best_model} ({best_score:.2%})")
    st.caption(
        "Dataset Weighting is used for live predictions in the Emotion Classifier. "
        "All four BERT strategies remain available here for evaluation and error analysis."
    )

    # Metric comparison chart
    st.subheader("Performance Comparison")

    selected_metric = st.selectbox("Select Evaluation Metric", METRICS, index=3)

    chart_data = (
        metrics_df[[selected_metric]].sort_values(selected_metric, ascending=False)
        * 100
    )

    st.bar_chart(chart_data, y_label="Score (%)")

    # Per-class performance
    st.subheader("Per-Class Performance")

    selected_model = st.selectbox(
        "Select Model for Detailed Analysis",
        list(FILES.keys()),
        index=list(FILES.keys()).index("Dataset Weighting"),
    )

    df = datasets[selected_model]
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )

    class_df = pd.DataFrame(
        {
            "Emotion": LABELS,
            "Precision": [report[label]["precision"] for label in LABELS],
            "Recall": [report[label]["recall"] for label in LABELS],
            "F1": [report[label]["f1-score"] for label in LABELS],
            "Support": [report[label]["support"] for label in LABELS],
        }
    ).set_index("Emotion")

    st.dataframe(
        class_df.style.format(
            {
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1": "{:.2%}",
                "Support": "{:.0f}",
            }
        ),
        width="stretch",
    )

    # Confusion matrix
    st.subheader("Confusion Matrix")

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)

    matrix_df = pd.DataFrame(
        matrix,
        index=[label.capitalize() for label in LABELS],
        columns=[label.capitalize() for label in LABELS],
    )

    matrix_df.index.name = "Actual"
    matrix_df.columns.name = "Predicted"

    st.dataframe(
        matrix_df.style.background_gradient(cmap="Blues", axis=None), width="stretch"
    )

    st.caption(
        "Rows represent actual emotions; columns represent "
        "predicted emotions. Diagonal values are correct predictions."
    )

    # Misclassification analysis
    st.subheader("Misclassification Analysis")

    errors = df[df["true_label"] != df["predicted_label"]].copy()

    confusion_pairs = (
        errors.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
    )

    st.write("Most Frequent Emotion Confusions")

    st.dataframe(confusion_pairs.head(10), hide_index=True, width="stretch")

    st.write("Misclassified Test Examples")

    selected_emotion = st.selectbox("Filter by Actual Emotion", ["All"] + LABELS)

    if selected_emotion != "All":
        errors = errors[errors["true_label"] == selected_emotion]

    st.dataframe(
        errors[["text", "true_label", "predicted_label"]],
        hide_index=True,
        width="stretch",
    )
