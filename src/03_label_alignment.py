# 03_label_alignment.py
"""
Purpose:
Align the source dataset's emotion labels with the target dataset's
seven-class emotion taxonomy using a fixed, documented semantic mapping.

Target taxonomy:
- anger
- disgust
- fear
- happy
- neutral
- sad
- surprise

Main tasks:
- Inspect source and target label spaces
- Apply the approved source-to-target label mapping
- Retain all cleaned source and target records
- Preserve original source labels for traceability
- Standardize dataset columns to:
    text
    label
    domain
- Verify that every aligned source label belongs to the target taxonomy
- Report the final source and target label distributions
- Identify target labels that are absent from the aligned source dataset

Important:
- No external emotion classifier is used.
- No valid cleaned samples are removed during label alignment.
- The mapping is a taxonomy-harmonisation decision and does not imply
  that the mapped emotions are perfectly equivalent.
- The target dataset's 'label' column remains unchanged and is used as
  the target classification label.
- The source dataset does not contain direct equivalents for every
  target emotion. In particular, fear and disgust are expected to be
  absent from the aligned source dataset.

Outputs:
- source_aligned_audit.csv
- source_aligned.csv
- target_aligned.csv
- label_alignment_summary.csv
- source_label_alignment_distribution.png

Owner:
Data Pipeline Member
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_DIR = "data/processed"
RESULTS_TABLES_DIR = "results/tables"
RESULTS_FIGURES_DIR = os.path.join("results", "figures")

SOURCE_CLEAN_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "source_clean.csv"
)

TARGET_CLEAN_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_clean.csv"
)

SOURCE_ALIGNED_AUDIT_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "source_aligned_audit.csv"
)

SOURCE_ALIGNED_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "source_aligned.csv"
)

TARGET_ALIGNED_PATH = os.path.join(
    PROCESSED_DATA_DIR,
    "target_aligned.csv"
)

ALIGNMENT_SUMMARY_PATH = os.path.join(
    RESULTS_TABLES_DIR,
    "label_alignment_summary.csv"
)


# ============================================================
# COLUMN NAMES
# ============================================================

SOURCE_TEXT_COLUMN = "text"
SOURCE_LABEL_COLUMN = "emotion"

TARGET_TEXT_COLUMN = "body"
TARGET_LABEL_COLUMN = "label"


# ============================================================
# APPROVED TARGET TAXONOMY
# ============================================================

TARGET_LABELS = {
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
}


# ============================================================
# APPROVED SOURCE -> TARGET LABEL MAPPING
# ============================================================

SOURCE_TO_TARGET_LABEL = {
    "anger": "anger",
    "hate": "anger",

    "happiness": "happy",
    "fun": "happy",
    "love": "happy",
    "enthusiasm": "happy",
    "relief": "happy",

    "sadness": "sad",

    "neutral": "neutral",
    "empty": "neutral",

    "surprise": "surprise",
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    """
    Check that all required columns are present.
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
        f"{dataset_name}: required columns validated."
    )


def print_label_distribution(df, label_column, dataset_name):
    """
    Print label counts and percentages.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - LABEL DISTRIBUTION")
    print("=" * 70)

    counts = df[label_column].value_counts(
        dropna=False
    )

    percentages = (
        df[label_column]
        .value_counts(
            normalize=True,
            dropna=False
        )
        * 100
    )

    distribution_df = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2)
    })

    print(distribution_df)

    return distribution_df


def validate_no_missing_values(
    df,
    columns,
    dataset_name
):
    """
    Verify that required fields contain no missing values.
    """

    for column in columns:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"{dataset_name}: column '{column}' "
                f"contains {missing_count} missing values."
            )


def validate_non_empty_text(
    df,
    text_column,
    dataset_name
):
    """
    Verify that text values are not empty or whitespace only.
    """

    empty_count = (
        df[text_column]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_count > 0:
        raise ValueError(
            f"{dataset_name} contains "
            f"{empty_count} empty text values."
        )


# ============================================================
# LOAD CLEANED DATASETS
# ============================================================

print("=" * 70)
print("LOADING CLEANED DATASETS")
print("=" * 70)

source_df = pd.read_csv(
    SOURCE_CLEAN_PATH
)

target_df = pd.read_csv(
    TARGET_CLEAN_PATH
)

source_original_rows = len(source_df)
target_original_rows = len(target_df)

print(
    f"Source cleaned dataset: "
    f"{source_original_rows} rows"
)

print(
    f"Target cleaned dataset: "
    f"{target_original_rows} rows"
)


# ============================================================
# VALIDATE INPUT STRUCTURE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING INPUT DATASETS")
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
        TARGET_TEXT_COLUMN,
        TARGET_LABEL_COLUMN
    ],
    "Target Dataset"
)

validate_no_missing_values(
    source_df,
    [
        SOURCE_TEXT_COLUMN,
        SOURCE_LABEL_COLUMN
    ],
    "Source Dataset"
)

validate_no_missing_values(
    target_df,
    [
        TARGET_TEXT_COLUMN,
        TARGET_LABEL_COLUMN
    ],
    "Target Dataset"
)

validate_non_empty_text(
    source_df,
    SOURCE_TEXT_COLUMN,
    "Source Dataset"
)

validate_non_empty_text(
    target_df,
    TARGET_TEXT_COLUMN,
    "Target Dataset"
)


# ============================================================
# NORMALIZE LABEL FORMATTING
# ============================================================

source_df[SOURCE_LABEL_COLUMN] = (
    source_df[SOURCE_LABEL_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)

target_df[TARGET_LABEL_COLUMN] = (
    target_df[TARGET_LABEL_COLUMN]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# INSPECT ORIGINAL LABEL SPACES
# ============================================================

print_label_distribution(
    source_df,
    SOURCE_LABEL_COLUMN,
    "Source Original Labels"
)

print_label_distribution(
    target_df,
    TARGET_LABEL_COLUMN,
    "Target Original Labels"
)

source_original_labels = set(
    source_df[SOURCE_LABEL_COLUMN].unique()
)

target_original_labels = set(
    target_df[TARGET_LABEL_COLUMN].unique()
)

print("\nSource original label set:")
print(sorted(source_original_labels))

print("\nTarget original label set:")
print(sorted(target_original_labels))


# ============================================================
# VALIDATE SOURCE MAPPING COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING SOURCE LABEL MAPPING")
print("=" * 70)

mapping_source_labels = set(
    SOURCE_TO_TARGET_LABEL.keys()
)

unmapped_source_labels = (
    source_original_labels
    - mapping_source_labels
)

if unmapped_source_labels:
    raise ValueError(
        "Some source labels do not have an approved mapping: "
        f"{sorted(unmapped_source_labels)}"
    )

unused_mapping_labels = (
    mapping_source_labels
    - source_original_labels
)

if unused_mapping_labels:
    print(
        "Warning: mapping contains labels not currently "
        "present in source data:"
    )
    print(sorted(unused_mapping_labels))

print(
    "All source labels have an approved mapping."
)


# ============================================================
# VALIDATE TARGET TAXONOMY
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING TARGET TAXONOMY")
print("=" * 70)

invalid_target_labels = (
    target_original_labels
    - TARGET_LABELS
)

if invalid_target_labels:
    raise ValueError(
        "Target dataset contains labels outside the "
        "approved taxonomy: "
        f"{sorted(invalid_target_labels)}"
    )

missing_target_labels = (
    TARGET_LABELS
    - target_original_labels
)

if missing_target_labels:
    print(
        "Warning: some approved target labels are not "
        "present in the current target dataset:"
    )
    print(sorted(missing_target_labels))
else:
    print(
        "Target dataset contains all seven approved labels."
    )


# ============================================================
# APPLY SOURCE LABEL ALIGNMENT
# ============================================================

print("\n" + "=" * 70)
print("APPLYING SOURCE LABEL ALIGNMENT")
print("=" * 70)

source_df["aligned_label"] = (
    source_df[SOURCE_LABEL_COLUMN]
    .map(SOURCE_TO_TARGET_LABEL)
)

missing_aligned_labels = (
    source_df["aligned_label"]
    .isna()
    .sum()
)

if missing_aligned_labels > 0:
    raise ValueError(
        f"{missing_aligned_labels} source rows "
        "could not be aligned."
    )

invalid_aligned_source_labels = (
    set(source_df["aligned_label"].unique())
    - TARGET_LABELS
)

if invalid_aligned_source_labels:
    raise ValueError(
        "Aligned source dataset contains labels outside "
        "the approved target taxonomy: "
        f"{sorted(invalid_aligned_source_labels)}"
    )

print(
    "Source label alignment completed successfully."
)


# ============================================================
# CREATE SOURCE AUDIT DATASET
# ============================================================

source_audit_df = pd.DataFrame({
    "text":
        source_df[SOURCE_TEXT_COLUMN],

    "original_label":
        source_df[SOURCE_LABEL_COLUMN],

    "aligned_label":
        source_df["aligned_label"],

    "domain":
        "source"
})

source_audit_df.to_csv(
    SOURCE_ALIGNED_AUDIT_PATH,
    index=False
)


# ============================================================
# CREATE MODEL-READY SOURCE DATASET
# ============================================================

source_aligned_df = pd.DataFrame({
    "text":
        source_df[SOURCE_TEXT_COLUMN],

    "label":
        source_df["aligned_label"],

    "domain":
        "source"
})


# ============================================================
# CREATE MODEL-READY TARGET DATASET
# ============================================================

target_aligned_df = pd.DataFrame({
    "text":
        target_df[TARGET_TEXT_COLUMN],

    "label":
        target_df[TARGET_LABEL_COLUMN],

    "domain":
        "target"
})


# ============================================================
# VERIFY ROW PRESERVATION
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING ROW PRESERVATION")
print("=" * 70)

if len(source_aligned_df) != source_original_rows:
    raise ValueError(
        "Source rows were lost during label alignment."
    )

if len(target_aligned_df) != target_original_rows:
    raise ValueError(
        "Target rows were lost during label alignment."
    )

print(
    f"Source rows preserved: "
    f"{len(source_aligned_df)} / "
    f"{source_original_rows}"
)

print(
    f"Target rows preserved: "
    f"{len(target_aligned_df)} / "
    f"{target_original_rows}"
)


# ============================================================
# VALIDATE FINAL DATASETS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING FINAL ALIGNED DATASETS")
print("=" * 70)

validate_no_missing_values(
    source_aligned_df,
    ["text", "label", "domain"],
    "Source Aligned Dataset"
)

validate_no_missing_values(
    target_aligned_df,
    ["text", "label", "domain"],
    "Target Aligned Dataset"
)

validate_non_empty_text(
    source_aligned_df,
    "text",
    "Source Aligned Dataset"
)

validate_non_empty_text(
    target_aligned_df,
    "text",
    "Target Aligned Dataset"
)

source_final_labels = set(
    source_aligned_df["label"].unique()
)

target_final_labels = set(
    target_aligned_df["label"].unique()
)

if not source_final_labels.issubset(
    TARGET_LABELS
):
    raise ValueError(
        "Source aligned labels are not a subset "
        "of the target taxonomy."
    )

if not target_final_labels.issubset(
    TARGET_LABELS
):
    raise ValueError(
        "Target labels are not a subset "
        "of the target taxonomy."
    )

print(
    "Final aligned datasets passed validation."
)


# ============================================================
# REPORT LABEL SETS
# ============================================================

print("\n" + "=" * 70)
print("FINAL LABEL SETS")
print("=" * 70)

print("\nAligned source labels:")
for label in sorted(source_final_labels):
    print(f"- {label}")

print("\nTarget labels:")
for label in sorted(target_final_labels):
    print(f"- {label}")

labels_absent_from_source = (
    target_final_labels
    - source_final_labels
)

print("\nTarget labels absent from aligned source:")

if labels_absent_from_source:
    for label in sorted(
        labels_absent_from_source
    ):
        print(f"- {label}")
else:
    print("None")


# ============================================================
# PRINT FINAL LABEL DISTRIBUTIONS
# ============================================================

source_aligned_distribution = (
    print_label_distribution(
        source_aligned_df,
        "label",
        "Source Aligned Labels"
    )
)

target_aligned_distribution = (
    print_label_distribution(
        target_aligned_df,
        "label",
        "Target Aligned Labels"
    )
)


# ============================================================
# SAVE MODEL-READY DATASETS
# ============================================================

source_aligned_df.to_csv(
    SOURCE_ALIGNED_PATH,
    index=False
)

target_aligned_df.to_csv(
    TARGET_ALIGNED_PATH,
    index=False
)

print("\n" + "=" * 70)
print("ALIGNED DATASETS SAVED")
print("=" * 70)

print(
    f"Source audit dataset: "
    f"{SOURCE_ALIGNED_AUDIT_PATH}"
)

print(
    f"Source aligned dataset: "
    f"{SOURCE_ALIGNED_PATH}"
)

print(
    f"Target aligned dataset: "
    f"{TARGET_ALIGNED_PATH}"
)


# ============================================================
# CREATE ALIGNMENT SUMMARY TABLE
# ============================================================

summary_rows = []

for label in sorted(TARGET_LABELS):

    source_count = int(
        (
            source_aligned_df["label"]
            == label
        ).sum()
    )

    target_count = int(
        (
            target_aligned_df["label"]
            == label
        ).sum()
    )

    source_percentage = (
        source_count
        / len(source_aligned_df)
        * 100
    )

    target_percentage = (
        target_count
        / len(target_aligned_df)
        * 100
    )

    summary_rows.append({
        "label": label,
        "source_count": source_count,
        "source_percentage":
            round(source_percentage, 2),
        "target_count": target_count,
        "target_percentage":
            round(target_percentage, 2),
        "present_in_source":
            source_count > 0,
        "present_in_target":
            target_count > 0,
    })


alignment_summary_df = pd.DataFrame(
    summary_rows
)

alignment_summary_df.to_csv(
    ALIGNMENT_SUMMARY_PATH,
    index=False
)

print("\n" + "=" * 70)
print("LABEL ALIGNMENT SUMMARY")
print("=" * 70)

print(
    alignment_summary_df.to_string(
        index=False
    )
)

print(
    f"\nSummary saved to: "
    f"{ALIGNMENT_SUMMARY_PATH}"
)

# ============================================================
# CREATE SOURCE LABEL ALIGNMENT DISTRIBUTION FIGURE
# ============================================================

print("\n" + "=" * 70)
print("CREATING LABEL ALIGNMENT DISTRIBUTION FIGURE")
print("=" * 70)

# Order of final aligned labels
aligned_label_order = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# Count how each original source label contributes
# to each final aligned label
alignment_counts = (
    source_audit_df
    .groupby(["aligned_label", "original_label"])
    .size()
    .unstack(fill_value=0)
    .reindex(aligned_label_order, fill_value=0)
)

# Keep original labels in a logical order within each stack
original_label_order = [
    "anger",
    "hate",
    "happiness",
    "fun",
    "love",
    "enthusiasm",
    "relief",
    "neutral",
    "empty",
    "sadness",
    "surprise"
]

alignment_counts = alignment_counts.reindex(
    columns=original_label_order,
    fill_value=0
)

fig, ax = plt.subplots(figsize=(12, 7))

bottom = [0] * len(alignment_counts)

for original_label in original_label_order:

    values = alignment_counts[original_label].values

    bars = ax.bar(
        [label.capitalize() for label in alignment_counts.index],
        values,
        bottom=bottom,
        label=original_label
    )

    # Put the original source emotion name inside
    # sufficiently large bar segments
    for i, (bar, value) in enumerate(zip(bars, values)):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom[i] + value / 2,
                original_label,
                ha="center",
                va="center",
                fontsize=9
            )

    bottom = [
        bottom[i] + values[i]
        for i in range(len(values))
    ]

ax.set_title(
    "Source Label Distribution After Emotion Taxonomy Alignment",
    fontsize=15
)

ax.set_xlabel("Final Aligned Label")
ax.set_ylabel("Number of Texts")

ax.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

figure_path = os.path.join(
    RESULTS_FIGURES_DIR,
    "source_label_alignment_distribution.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Figure saved to: {figure_path}")

# ============================================================
# PRINT APPROVED MAPPING
# ============================================================

print("\n" + "=" * 70)
print("SOURCE -> TARGET LABEL MAPPING")
print("=" * 70)

for source_label, target_label in (
    SOURCE_TO_TARGET_LABEL.items()
):
    print(
        f"{source_label:<15} -> "
        f"{target_label}"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("LABEL ALIGNMENT COMPLETE")
print("=" * 70)

print(
    f"Source rows retained: "
    f"{len(source_aligned_df)}"
)

print(
    f"Target rows retained: "
    f"{len(target_aligned_df)}"
)

print(
    f"Aligned source classes: "
    f"{len(source_final_labels)}"
)

print(
    f"Target classes: "
    f"{len(target_final_labels)}"
)

if labels_absent_from_source:
    print(
        "Target-only classes: "
        + ", ".join(
            sorted(
                labels_absent_from_source
            )
        )
    )

print(
    "\nNo cleaned records were removed during "
    "label alignment."
)