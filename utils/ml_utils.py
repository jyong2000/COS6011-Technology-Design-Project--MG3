# ml_utils.py
"""
Shared helper functions for scripts 05, 06 and 07.

Scripts 05 (Logistic Regression), 06 (SVM) and 07 (Naive Bayes) do exactly
the same thing apart from which classifier they use. Everything that is the
same lives in this file, so that the three models are compared fairly and
all use the exact same dataset splits that 04_prepare_datasets.py produced.

Nothing in this file needs to be changed except the CONFIGURATION block below,
and even that should not need to change if the project folder layout matches
04_prepare_datasets.py's default output paths.

Owner:
Traditional ML Member
"""

import os
import json

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DIR = "data/processed"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

RANDOM_STATE = 42

# After the best hyperparameters are chosen using the validation set, refit
# the winning pipeline on train + validation combined before scoring the
# test set. This uses all the non-test data available without touching the
# test set, which 04_prepare_datasets.py reserves for final evaluation only.
REFIT_ON_TRAIN_VAL = True


os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# LOADING THE DATA
# ============================================================
#
# These are the exact files 04_prepare_datasets.py produces. Nothing here
# rebuilds or resplits anything - 04's docstring is explicit that every
# team member must use these exact splits, including the already-built
# mixed_train.csv for Experiment C, so this file only loads them.

SPLIT_FILENAMES = {
    "source_train": "source_train.csv",
    "source_validation": "source_validation.csv",
    "target_train": "target_train.csv",
    "target_validation": "target_validation.csv",
    "target_test": "target_test.csv",
    "mixed_train": "mixed_train.csv",
}


def clean_dataframe(df, dataset_name):
    """
    Light safety check only. 04_prepare_datasets.py already validates and
    cleans these files, so this just guards against an empty text field
    slipping through and makes sure labels are plain strings.
    """

    before_count = len(df)

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df = df[df[TEXT_COLUMN].str.strip() != ""]
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

    dropped_count = before_count - len(df)
    if dropped_count > 0:
        print(
            f"{dataset_name}: dropped {dropped_count} rows with empty text "
            f"(unexpected - 04_prepare_datasets.py should already guarantee this)"
        )

    return df.reset_index(drop=True)


def load_data():
    """
    Load the six files produced by 04_prepare_datasets.py, and report which
    target classes (if any) never appear in the source training data.
    This is expected, not an error - 04's own docstring states the source
    dataset contains five of the seven target emotion classes, and that
    fear and disgust are target-only.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    paths = {
        key: os.path.join(PROCESSED_DIR, filename)
        for key, filename in SPLIT_FILENAMES.items()
    }

    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            "Missing files from 04_prepare_datasets.py:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nRun 04_prepare_datasets.py first."
        )

    splits = {
        key: clean_dataframe(pd.read_csv(path), key)
        for key, path in paths.items()
    }

    for name, df in splits.items():
        print(f"{name:20s} rows: {len(df)}")

    source_labels = set(splits["source_train"][LABEL_COLUMN].unique())
    target_labels = set(splits["target_train"][LABEL_COLUMN].unique())

    print(f"\nSource training labels: {sorted(source_labels)}")
    print(f"Target training labels: {sorted(target_labels)}")

    target_only_labels = target_labels - source_labels
    if target_only_labels:
        print(
            f"\nNote: {sorted(target_only_labels)} appear only in target "
            "training data. This is expected per 03/04's approved taxonomy "
            "- the source dataset has no equivalent for these emotions. "
            "Experiment A (source-only) will never predict these classes; "
            "report this as a known, documented limitation rather than a bug."
        )

    return splits


def get_label_list(splits):
    """
    The fixed label order used for every report and confusion matrix.
    Uses the union of every split's labels, so "fear" and "disgust" still
    get a row and column in every confusion matrix, even for Experiment A,
    which can never predict them.
    """

    all_labels = set()
    for df in splits.values():
        all_labels |= set(df[LABEL_COLUMN].unique())

    return sorted(all_labels)


# ============================================================
# THE THREE EXPERIMENTS
# ============================================================
#
# Each experiment is trained on a different file but validated and tested
# on target data, since the target domain is what the project is ultimately
# evaluated on. This matches 04_prepare_datasets.py: source_validation is
# only ever used to check Experiment A hasn't overfit to source, never to
# pick a model that will be scored on the target test set.

def get_experiment(experiment, splits):
    """
    Return the {name, train, val, test} dictionary for one experiment.

    A - source only : train=source_train,  val=source_validation
    B - target only : train=target_train,  val=target_validation
    C - mixed       : train=mixed_train,   val=target_validation

    mixed_train.csv is loaded directly from 04_prepare_datasets.py's output,
    not rebuilt here, so that every model in the project uses the identical
    mixed training set.
    """

    if experiment == "A":
        train_df = splits["source_train"]
        val_df = splits["source_validation"]
    elif experiment == "B":
        train_df = splits["target_train"]
        val_df = splits["target_validation"]
    elif experiment == "C":
        train_df = splits["mixed_train"]
        val_df = splits["target_validation"]
    else:
        raise ValueError("experiment must be 'A', 'B' or 'C'")

    return {
        "name": experiment,
        "train": train_df,
        "val": val_df,
        "test": splits["target_test"],
    }


# ============================================================
# MODEL BUILDING AND TUNING
# ============================================================

def build_pipeline(classifier):
    """
    TF-IDF followed by a classifier.

    The vectoriser is inside the pipeline on purpose. It is fitted on the
    training data only, and never sees the validation or test set. Fitting
    TF-IDF on everything before splitting would leak validation/test
    vocabulary into training and make the scores look better than they
    really are.
    """

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )),
        ("classifier", classifier)
    ])

    return pipeline


def select_best_model(classifier, parameter_grid, train_df, val_df, label_list):
    """
    Try every combination in the parameter grid, fit it on train_df, score
    it on val_df, and keep the best one.

    This uses 04_prepare_datasets.py's fixed validation set rather than
    scikit-learn's own internal k-fold cross-validation, so that every
    model in the project is tuned against the exact same validation rows.

    macro F1 is used to choose the winner, not accuracy, because the
    emotion classes are imbalanced and accuracy would reward a model that
    ignores the rare emotions.
    """

    combinations = list(ParameterGrid(parameter_grid))
    print(f"Trying {len(combinations)} parameter combinations...")

    best_score = -1.0
    best_params = None
    history = []

    for params in combinations:
        pipeline = build_pipeline(classifier)
        pipeline.set_params(**params)
        pipeline.fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN])

        val_predictions = pipeline.predict(val_df[TEXT_COLUMN])
        score = f1_score(
            val_df[LABEL_COLUMN], val_predictions,
            labels=label_list, average="macro", zero_division=0
        )

        history.append({
            "params": {k: str(v) for k, v in params.items()},
            "val_macro_f1": round(float(score), 4),
        })

        if score > best_score:
            best_score = score
            best_params = params

    history.sort(key=lambda row: -row["val_macro_f1"])

    print(f"\nBest validation macro F1: {best_score:.4f}")
    print("Best parameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")

    return best_params, best_score, history


# ============================================================
# EVALUATION AND SAVING
# ============================================================

def evaluate_predictions(true_labels, predicted_labels, label_list):
    """
    Calculate the headline metrics for one model.
    """

    return {
        "accuracy": round(
            accuracy_score(true_labels, predicted_labels), 4
        ),
        "macro_f1": round(
            f1_score(
                true_labels, predicted_labels,
                labels=label_list, average="macro", zero_division=0
            ), 4
        ),
        "weighted_f1": round(
            f1_score(
                true_labels, predicted_labels,
                labels=label_list, average="weighted", zero_division=0
            ), 4
        ),
    }


def save_predictions(model_name, experiment, test_df,
                     predicted_labels, probabilities, label_list):
    """
    Save one row per test example. Script 16_error_analysis.py reads this
    file, so keep the column names stable.
    """

    predictions_df = pd.DataFrame({
        "text": test_df[TEXT_COLUMN].values,
        "true_label": test_df[LABEL_COLUMN].values,
        "predicted_label": predicted_labels,
        "correct": test_df[LABEL_COLUMN].values == predicted_labels,
        "model": model_name,
        "experiment": experiment,
    })

    if probabilities is not None:
        for index, label in enumerate(label_list):
            predictions_df[f"probability_{label}"] = probabilities[:, index]

    output_path = os.path.join(
        PREDICTIONS_DIR,
        f"{model_name}_{experiment}_target_test.csv"
    )

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions: {output_path}")


def save_confusion_matrix(model_name, experiment,
                          true_labels, predicted_labels, label_list):
    """
    Save the confusion matrix as both a CSV table and a PNG figure.
    Rows are the true label, columns are the predicted label.
    """

    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=label_list
    )

    matrix_df = pd.DataFrame(
        matrix, index=label_list, columns=label_list
    )

    matrix_df.to_csv(os.path.join(
        METRICS_DIR,
        f"{model_name}_{experiment}_confusion_matrix.csv"
    ))

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.title(f"{model_name} - Experiment {experiment}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(range(len(label_list)), label_list, rotation=45, ha="right")
    plt.yticks(range(len(label_list)), label_list)

    for row in range(len(label_list)):
        for column in range(len(label_list)):
            plt.text(
                column, row, matrix[row, column],
                ha="center", va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black"
            )

    plt.tight_layout()

    figure_path = os.path.join(
        FIGURES_DIR,
        f"{model_name}_{experiment}_confusion_matrix.png"
    )

    plt.savefig(figure_path)
    plt.close()

    print(f"Saved figure: {figure_path}")


def append_to_summary(record):
    """
    Add one row to results/metrics/summary.csv, the single table that
    compares every model and experiment in the project.
    """

    summary_path = os.path.join(METRICS_DIR, "summary.csv")

    row = pd.DataFrame([{
        "model": record["model"],
        "experiment": record["experiment"],
        "training_rows": record["training_rows"],
        "validation_rows": record["validation_rows"],
        "accuracy": record["accuracy"],
        "macro_f1": record["macro_f1"],
        "weighted_f1": record["weighted_f1"],
        "val_macro_f1": record["val_macro_f1"],
    }])

    row.to_csv(
        summary_path,
        mode="a",
        header=not os.path.exists(summary_path),
        index=False
    )


# ============================================================
# MAIN DRIVER
# ============================================================

def run_experiment(model_name, experiment, classifier,
                   parameter_grid, splits, label_list):
    """
    Run one model on one experiment, from training to saved results.
    """

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - EXPERIMENT {experiment}")
    print("=" * 70)

    exp = get_experiment(experiment, splits)
    train_df, val_df, test_df = exp["train"], exp["val"], exp["test"]

    print(f"Training rows:   {len(train_df)}")
    print(f"Validation rows: {len(val_df)}  (used to pick hyperparameters)")
    print(f"Test rows:       {len(test_df)}  (target_test.csv, scored once)")

    best_params, best_val_score, history = select_best_model(
        classifier, parameter_grid, train_df, val_df, label_list
    )

    if REFIT_ON_TRAIN_VAL:
        fit_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"\nRefitting on train + validation ({len(fit_df)} rows) "
              f"before scoring the test set.")
    else:
        fit_df = train_df

    final_pipeline = build_pipeline(classifier)
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(fit_df[TEXT_COLUMN], fit_df[LABEL_COLUMN])

    predicted_labels = final_pipeline.predict(test_df[TEXT_COLUMN])

    # Probabilities are useful for error analysis, but LinearSVC does not
    # provide them, so this is optional.
    probabilities = None
    if hasattr(final_pipeline.named_steps["classifier"], "predict_proba"):
        raw_probabilities = final_pipeline.predict_proba(test_df[TEXT_COLUMN])
        model_label_order = list(final_pipeline.named_steps["classifier"].classes_)
        column_order = [model_label_order.index(l) for l in label_list
                        if l in model_label_order]
        if len(column_order) == len(label_list):
            probabilities = raw_probabilities[:, column_order]

    metrics = evaluate_predictions(
        test_df[LABEL_COLUMN].values, predicted_labels, label_list
    )

    print(f"\nTarget test accuracy : {metrics['accuracy']}")
    print(f"Target test macro F1 : {metrics['macro_f1']}")
    print("\nFull classification report:")
    print(classification_report(
        test_df[LABEL_COLUMN].values,
        predicted_labels,
        labels=label_list,
        zero_division=0
    ))

    save_predictions(
        model_name, experiment, test_df,
        predicted_labels, probabilities, label_list
    )

    save_confusion_matrix(
        model_name, experiment,
        test_df[LABEL_COLUMN].values, predicted_labels, label_list
    )

    record = {
        "model": model_name,
        "experiment": experiment,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "refit_on_train_val": REFIT_ON_TRAIN_VAL,
        "val_macro_f1": round(float(best_val_score), 4),
        "best_parameters": {
            key: str(value) for key, value in best_params.items()
        },
        "per_class_report": classification_report(
            test_df[LABEL_COLUMN].values,
            predicted_labels,
            labels=label_list,
            zero_division=0,
            output_dict=True
        ),
    }
    record.update(metrics)

    metrics_path = os.path.join(
        METRICS_DIR, f"{model_name}_{experiment}.json"
    )

    with open(metrics_path, "w") as output_file:
        json.dump(record, output_file, indent=2)

    print(f"Saved metrics: {metrics_path}")

    search_history_path = os.path.join(
        METRICS_DIR, f"{model_name}_{experiment}_search_history.json"
    )
    with open(search_history_path, "w") as output_file:
        json.dump(history, output_file, indent=2)

    append_to_summary(record)

    return record


def run_all_experiments(model_name, classifier, parameter_grid):
    """
    Run experiments A, B and C for one model and print a small comparison
    table at the end. This is the only function scripts 05-07 need to call.
    """

    splits = load_data()
    label_list = get_label_list(splits)

    print(f"\nLabels used ({len(label_list)}): {label_list}")

    records = []

    for experiment in ["A", "B", "C"]:
        records.append(run_experiment(
            model_name, experiment, classifier,
            parameter_grid, splits, label_list
        ))

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} - SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame(records)[
        ["experiment", "training_rows", "accuracy", "macro_f1", "weighted_f1"]
    ]

    print(summary_df.to_string(index=False))

    return records
