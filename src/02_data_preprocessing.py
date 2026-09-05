# 02_data_preprocessing.py
"""
Purpose:
Clean and preprocess the source and target datasets based on EDA findings.

Main tasks:
- Validate required columns
- Remove unusable or invalid rows
- Normalize whitespace
- Remove duplicate source examples
- Remove conflicting source texts
- Preserve important financial-domain information
- Use target 'body' as the primary text field
- Use target 'label' as the primary ground-truth label
- Save a preprocessing summary

Important:
Do not perform label alignment in this script.

Outputs:
- Cleaned source dataset
- Cleaned target dataset
- Preprocessing summary

Owner:
Data Pipeline Member
"""

import os
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DATA_PATH = "data/Source Dataset.csv"
TARGET_DATA_PATH = "data/Target Dataset.csv"

PROCESSED_DATA_DIR = "data/processed"
RESULTS_TABLES_DIR = "results/tables"

SOURCE_OUTPUT_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "source_clean.csv"
)

TARGET_OUTPUT_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_clean.csv"
)

PREPROCESSING_SUMMARY_PATH = os.path.join(
    RESULTS_TABLES_DIR,
    "preprocessing_summary.csv"
)

SOURCE_TEXT_COLUMN = "text"
SOURCE_LABEL_COLUMN = "emotion"

TARGET_RAW_TEXT_COLUMN = "body"
TARGET_CLEAN_TEXT_COLUMN = "cleaned_text"
TARGET_LABEL_COLUMN = "label"
TARGET_CHATGPT_LABEL_COLUMN = "chatGPT_label"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    """
    Check that all required columns are present in the dataset.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"{dataset_name}: all required columns are present."
    )


def normalize_whitespace(text):
    """
    Normalize leading, trailing and repeated whitespace.

    This intentionally performs only conservative text cleaning.
    Punctuation, numbers, emojis, ticker symbols and other potentially
    meaningful information are preserved.
    """

    if pd.isna(text):
        return text

    text = str(text)

    # Replace repeated whitespace, tabs and line breaks with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    return text.strip()


def remove_missing_required_values(
    df,
    required_columns
):
    """
    Remove rows containing missing values in required columns.
    """

    original_rows = len(df)

    cleaned_df = df.dropna(
        subset=required_columns
    ).copy()

    removed_rows = original_rows - len(cleaned_df)

    return cleaned_df, removed_rows


def remove_empty_text_rows(
    df,
    text_column
):
    """
    Remove rows where the selected text column is empty after
    whitespace normalization.
    """

    original_rows = len(df)

    cleaned_df = df[
        df[text_column].astype(str).str.strip().ne("")
    ].copy()

    removed_rows = original_rows - len(cleaned_df)

    return cleaned_df, removed_rows


def find_conflicting_texts(
    df,
    text_column,
    label_column
):
    """
    Identify texts associated with more than one different label.
    """

    label_counts = (
        df.groupby(text_column)[label_column]
        .nunique()
    )

    conflicting_texts = label_counts[
        label_counts > 1
    ].index

    return conflicting_texts


def remove_conflicting_texts(
    df,
    text_column,
    label_column
):
    """
    Remove all rows containing text that is associated with
    multiple different labels.

    The script does not attempt to guess which label is correct.
    """

    conflicting_texts = find_conflicting_texts(
        df,
        text_column,
        label_column
    )

    conflict_mask = df[text_column].isin(
        conflicting_texts
    )

    removed_rows = int(conflict_mask.sum())

    cleaned_df = df[
        ~conflict_mask
    ].copy()

    return (
        cleaned_df,
        len(conflicting_texts),
        removed_rows
    )


def remove_duplicate_texts(
    df,
    text_column
):
    """
    Remove duplicate text entries while retaining the first occurrence.

    Conflicting labels should be removed before this function is used.
    """

    original_rows = len(df)

    cleaned_df = df.drop_duplicates(
        subset=[text_column],
        keep="first"
    ).copy()

    removed_rows = original_rows - len(cleaned_df)

    return cleaned_df, removed_rows


def validate_clean_dataset(
    df,
    text_column,
    label_column,
    dataset_name
):
    """
    Perform final checks on a cleaned dataset.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - FINAL VALIDATION")
    print("=" * 70)

    missing_text = df[text_column].isna().sum()
    missing_labels = df[label_column].isna().sum()

    empty_text = (
        df[text_column]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    duplicate_texts = df[text_column].duplicated().sum()

    conflicting_texts = len(
        find_conflicting_texts(
            df,
            text_column,
            label_column
        )
    )

    print(f"Missing text values: {missing_text}")
    print(f"Missing labels: {missing_labels}")
    print(f"Empty text rows: {empty_text}")
    print(f"Duplicate texts: {duplicate_texts}")
    print(f"Conflicting texts: {conflicting_texts}")

    if (
        missing_text != 0
        or missing_labels != 0
        or empty_text != 0
        or duplicate_texts != 0
        or conflicting_texts != 0
    ):
        raise ValueError(
            f"{dataset_name} failed final validation."
        )

    print(f"{dataset_name} passed final validation.")


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

source_df = pd.read_csv(SOURCE_DATA_PATH)
target_df = pd.read_csv(TARGET_DATA_PATH)

print(
    f"Source dataset loaded successfully: "
    f"{len(source_df)} rows"
)

print(
    f"Target dataset loaded successfully: "
    f"{len(target_df)} rows"
)


# ============================================================
# VALIDATE DATASET STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATASET STRUCTURE")
print("=" * 70)

validate_columns(
    source_df,
    [
        SOURCE_TEXT_COLUMN,
        SOURCE_LABEL_COLUMN
    ],
    "Source Dataset"
)

validate_columns(
    target_df,
    [
        TARGET_RAW_TEXT_COLUMN,
        TARGET_CLEAN_TEXT_COLUMN,
        TARGET_LABEL_COLUMN,
        TARGET_CHATGPT_LABEL_COLUMN
    ],
    "Target Dataset"
)


# ============================================================
# SOURCE DATASET PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("SOURCE DATASET PREPROCESSING")
print("=" * 70)

source_original_rows = len(source_df)

# ------------------------------------------------------------
# Remove missing required values
# ------------------------------------------------------------

source_df, source_missing_removed = (
    remove_missing_required_values(
        source_df,
        [
            SOURCE_TEXT_COLUMN,
            SOURCE_LABEL_COLUMN
        ]
    )
)

print(
    f"Rows removed due to missing text/label: "
    f"{source_missing_removed}"
)


# ------------------------------------------------------------
# Normalize whitespace
# ------------------------------------------------------------

source_df[SOURCE_TEXT_COLUMN] = (
    source_df[SOURCE_TEXT_COLUMN]
    .apply(normalize_whitespace)
)

print("Source text whitespace normalized.")


# ------------------------------------------------------------
# Remove empty text
# ------------------------------------------------------------

source_df, source_empty_removed = (
    remove_empty_text_rows(
        source_df,
        SOURCE_TEXT_COLUMN
    )
)

print(
    f"Whitespace-only rows removed: "
    f"{source_empty_removed}"
)


# ------------------------------------------------------------
# Remove conflicting texts
# ------------------------------------------------------------

(
    source_df,
    source_conflicting_text_count,
    source_conflicting_rows_removed
) = remove_conflicting_texts(
    source_df,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN
)

print(
    f"Unique conflicting texts identified: "
    f"{source_conflicting_text_count}"
)

print(
    f"Rows removed because of conflicting labels: "
    f"{source_conflicting_rows_removed}"
)


# ------------------------------------------------------------
# Remove duplicate texts
# ------------------------------------------------------------

source_df, source_duplicates_removed = (
    remove_duplicate_texts(
        source_df,
        SOURCE_TEXT_COLUMN
    )
)

print(
    f"Duplicate text rows removed: "
    f"{source_duplicates_removed}"
)


# Reset index after cleaning
source_df = source_df.reset_index(drop=True)

source_final_rows = len(source_df)


# ============================================================
# TARGET DATASET PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("TARGET DATASET PREPROCESSING")
print("=" * 70)

target_original_rows = len(target_df)

print(
    "Primary target text field: "
    f"'{TARGET_RAW_TEXT_COLUMN}'"
)

print(
    "Primary target label field: "
    f"'{TARGET_LABEL_COLUMN}'"
)


# ------------------------------------------------------------
# Remove missing required values
# ------------------------------------------------------------

# cleaned_text is intentionally NOT required here.
# The modelling text field is body.
target_df, target_missing_removed = (
    remove_missing_required_values(
        target_df,
        [
            TARGET_RAW_TEXT_COLUMN,
            TARGET_LABEL_COLUMN
        ]
    )
)

print(
    f"Rows removed due to missing body/label: "
    f"{target_missing_removed}"
)


# ------------------------------------------------------------
# Normalize primary target text
# ------------------------------------------------------------

target_df[TARGET_RAW_TEXT_COLUMN] = (
    target_df[TARGET_RAW_TEXT_COLUMN]
    .apply(normalize_whitespace)
)

print("Target body whitespace normalized.")


# ------------------------------------------------------------
# Normalize cleaned_text where available
# ------------------------------------------------------------

target_df[TARGET_CLEAN_TEXT_COLUMN] = (
    target_df[TARGET_CLEAN_TEXT_COLUMN]
    .apply(normalize_whitespace)
)

print(
    "Target cleaned_text whitespace normalized "
    "where available."
)


# ------------------------------------------------------------
# Remove empty primary text
# ------------------------------------------------------------

target_df, target_empty_removed = (
    remove_empty_text_rows(
        target_df,
        TARGET_RAW_TEXT_COLUMN
    )
)

print(
    f"Whitespace-only body rows removed: "
    f"{target_empty_removed}"
)


# ------------------------------------------------------------
# Remove conflicting texts if any
# ------------------------------------------------------------

(
    target_df,
    target_conflicting_text_count,
    target_conflicting_rows_removed
) = remove_conflicting_texts(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    TARGET_LABEL_COLUMN
)

print(
    f"Unique conflicting target texts identified: "
    f"{target_conflicting_text_count}"
)

print(
    f"Target rows removed because of conflicting labels: "
    f"{target_conflicting_rows_removed}"
)


# ------------------------------------------------------------
# Remove duplicate primary texts if any
# ------------------------------------------------------------

target_df, target_duplicates_removed = (
    remove_duplicate_texts(
        target_df,
        TARGET_RAW_TEXT_COLUMN
    )
)

print(
    f"Duplicate target text rows removed: "
    f"{target_duplicates_removed}"
)


# Reset index
target_df = target_df.reset_index(drop=True)

target_final_rows = len(target_df)


# ============================================================
# FINAL VALIDATION
# ============================================================

validate_clean_dataset(
    source_df,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN,
    "Source Dataset"
)

validate_clean_dataset(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    TARGET_LABEL_COLUMN,
    "Target Dataset"
)


# ============================================================
# SAVE CLEANED DATASETS
# ============================================================

source_df.to_csv(
    SOURCE_OUTPUT_PATH,
    index=False
)

target_df.to_csv(
    TARGET_OUTPUT_PATH,
    index=False
)

print("\n" + "=" * 70)
print("CLEANED DATASETS SAVED")
print("=" * 70)

print(f"Source: {SOURCE_OUTPUT_PATH}")
print(f"Target: {TARGET_OUTPUT_PATH}")


# ============================================================
# CREATE PREPROCESSING SUMMARY
# ============================================================

summary_data = [
    {
        "dataset": "Source",
        "original_rows": source_original_rows,
        "missing_required_removed": source_missing_removed,
        "empty_text_removed": source_empty_removed,
        "unique_conflicting_texts":
            source_conflicting_text_count,
        "conflicting_rows_removed":
            source_conflicting_rows_removed,
        "duplicate_text_rows_removed":
            source_duplicates_removed,
        "final_rows": source_final_rows
    },
    {
        "dataset": "Target",
        "original_rows": target_original_rows,
        "missing_required_removed": target_missing_removed,
        "empty_text_removed": target_empty_removed,
        "unique_conflicting_texts":
            target_conflicting_text_count,
        "conflicting_rows_removed":
            target_conflicting_rows_removed,
        "duplicate_text_rows_removed":
            target_duplicates_removed,
        "final_rows": target_final_rows
    }
]

summary_df = pd.DataFrame(summary_data)

summary_df.to_csv(
    PREPROCESSING_SUMMARY_PATH,
    index=False
)


# ============================================================
# PREPROCESSING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING SUMMARY")
print("=" * 70)

print(summary_df.to_string(index=False))

print(
    f"\nSummary saved to: "
    f"{PREPROCESSING_SUMMARY_PATH}"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING COMPLETE")
print("=" * 70)

print(
    "\nNo label alignment was performed. "
    "Label alignment will be handled by "
    "03_label_alignment.py."
)