# 04_prepare_datasets.py
"""
Purpose:
Create the fixed and reproducible train, validation and test datasets
used by all project experiments.

Main tasks:
- Load the aligned source and target datasets
- Use a fixed random seed for reproducibility
- Create stratified dataset splits
- Create source training and validation sets
- Create target training, validation and test sets
- Lock the target test set for final evaluation only
- Create the mixed source + target training dataset
- Preserve domain labels in all generated datasets
- Verify that no records are lost during splitting
- Check for duplicate-text leakage between train, validation and test sets
- Check for cross-domain overlap involving the target test set
- Verify class distributions after splitting
- Report source classes that are absent from the source domain
- Save dataset split statistics for later reporting

Split strategy:
Source:
- 85% training
- 15% validation

Target:
- 70% training
- 15% validation
- 15% test

Random seed:
- 42

Important:
- All team members must use these exact dataset splits.
- The target test set must not be used for model training,
  hyperparameter tuning or model selection.
- Stratification is performed using the aligned label column.
- The source dataset contains five of the seven target emotion classes.
  Fear and disgust are target-only classes.
- No samples are removed simply to balance classes.
- The mixed training dataset is created from source_train and
  target_train only.
- The target test set must never be included in mixed training data.
- Cross-domain exact-text overlap is checked before the final datasets
  are accepted.

Outputs:
- source_train.csv
- source_validation.csv
- target_train.csv
- target_validation.csv
- target_test.csv
- mixed_train.csv
- dataset_split_summary.csv
- dataset_class_distribution.csv

Owner:
Data Pipeline Member
"""

import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

PROCESSED_DATA_DIR = "data/processed"
RESULTS_TABLES_DIR = "results/tables"

SOURCE_ALIGNED_PATH = os.path.join(PROCESSED_DATA_DIR, "source_aligned.csv")

TARGET_ALIGNED_PATH = os.path.join(PROCESSED_DATA_DIR, "target_aligned.csv")

SOURCE_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "source_train.csv")

SOURCE_VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "source_validation.csv")

TARGET_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "target_train.csv")

TARGET_VALIDATION_PATH = os.path.join(PROCESSED_DATA_DIR, "target_validation.csv")

TARGET_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "target_test.csv")

MIXED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "mixed_train.csv")

SPLIT_SUMMARY_PATH = os.path.join(RESULTS_TABLES_DIR, "dataset_split_summary.csv")

CLASS_DISTRIBUTION_PATH = os.path.join(
    RESULTS_TABLES_DIR, "dataset_class_distribution.csv"
)


# ============================================================
# SPLIT SETTINGS
# ============================================================

RANDOM_SEED = 42

SOURCE_TRAIN_SIZE = 0.85
SOURCE_VALIDATION_SIZE = 0.15

TARGET_TRAIN_SIZE = 0.70
TARGET_VALIDATION_SIZE = 0.15
TARGET_TEST_SIZE = 0.15


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
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_TABLES_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def validate_columns(df, required_columns, dataset_name):
    """
    Verify that all required columns exist.
    """

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: " f"{missing_columns}"
        )

    print(f"{dataset_name}: required columns validated.")


def validate_no_missing_values(df, columns, dataset_name):
    """
    Verify that required columns contain no missing values.
    """

    for column in columns:
        missing_count = df[column].isna().sum()

        if missing_count > 0:
            raise ValueError(
                f"{dataset_name}: column '{column}' "
                f"contains {missing_count} missing values."
            )


def validate_non_empty_text(df, dataset_name):
    """
    Verify that no text values are empty or whitespace only.
    """

    empty_count = df["text"].astype(str).str.strip().eq("").sum()

    if empty_count > 0:
        raise ValueError(
            f"{dataset_name} contains " f"{empty_count} empty text values."
        )


def normalize_text_for_leakage(text):
    """
    Normalize text only for leakage comparison.

    This does NOT modify the text stored in the datasets.

    Normalization:
    - convert to lowercase
    - collapse repeated whitespace
    - remove leading/trailing whitespace
    """

    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_leakage_keys(df):
    """
    Create normalized text keys for leakage checking.
    """

    return set(df["text"].apply(normalize_text_for_leakage).tolist())


def check_overlap(df_a, df_b, name_a, name_b):
    """
    Check exact normalized text overlap between two datasets.

    Returns the overlapping text keys.
    """

    keys_a = create_leakage_keys(df_a)
    keys_b = create_leakage_keys(df_b)

    overlap = keys_a.intersection(keys_b)

    print(f"{name_a} <-> {name_b}: " f"{len(overlap)} overlapping texts")

    return overlap


def print_label_distribution(df, dataset_name):
    """
    Print label counts and percentages.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - CLASS DISTRIBUTION")
    print("=" * 70)

    counts = df["label"].value_counts().sort_index()

    percentages = df["label"].value_counts(normalize=True).sort_index() * 100

    distribution_df = pd.DataFrame(
        {"count": counts, "percentage": percentages.round(2)}
    )

    print(distribution_df)

    return distribution_df


def validate_dataset(df, dataset_name):
    """
    Perform final structural validation.
    """

    validate_columns(df, ["text", "label", "domain"], dataset_name)

    validate_no_missing_values(df, ["text", "label", "domain"], dataset_name)

    validate_non_empty_text(df, dataset_name)

    invalid_labels = set(df["label"].unique()) - TARGET_LABELS

    if invalid_labels:
        raise ValueError(
            f"{dataset_name} contains labels outside "
            f"the approved taxonomy: "
            f"{sorted(invalid_labels)}"
        )


def add_distribution_rows(output_rows, df, dataset_name):
    """
    Add per-class statistics for a dataset.
    """

    total_rows = len(df)

    for label in sorted(TARGET_LABELS):

        count = int((df["label"] == label).sum())

        percentage = count / total_rows * 100 if total_rows > 0 else 0

        output_rows.append(
            {
                "dataset": dataset_name,
                "label": label,
                "count": count,
                "percentage": round(percentage, 2),
            }
        )


# ============================================================
# LOAD ALIGNED DATASETS
# ============================================================

print("=" * 70)
print("LOADING ALIGNED DATASETS")
print("=" * 70)

source_df = pd.read_csv(SOURCE_ALIGNED_PATH)

target_df = pd.read_csv(TARGET_ALIGNED_PATH)

source_total_rows = len(source_df)
target_total_rows = len(target_df)

print(f"Aligned source dataset: " f"{source_total_rows} rows")

print(f"Aligned target dataset: " f"{target_total_rows} rows")


# ============================================================
# VALIDATE INPUT DATASETS
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING INPUT DATASETS")
print("=" * 70)

validate_dataset(source_df, "Source Aligned Dataset")

validate_dataset(target_df, "Target Aligned Dataset")

print("Aligned source and target datasets " "passed validation.")


# ============================================================
# VERIFY DOMAIN VALUES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING DOMAIN LABELS")
print("=" * 70)

source_domains = set(source_df["domain"].unique())

target_domains = set(target_df["domain"].unique())

if source_domains != {"source"}:
    raise ValueError(
        "Source aligned dataset contains unexpected " f"domain values: {source_domains}"
    )

if target_domains != {"target"}:
    raise ValueError(
        "Target aligned dataset contains unexpected " f"domain values: {target_domains}"
    )

print("Source domain label validated: source")

print("Target domain label validated: target")


# ============================================================
# INSPECT CLASS COVERAGE
# ============================================================

print("\n" + "=" * 70)
print("CLASS COVERAGE BEFORE SPLITTING")
print("=" * 70)

source_labels = set(source_df["label"].unique())

target_labels = set(target_df["label"].unique())

print("\nSource labels:")
for label in sorted(source_labels):
    print(f"- {label}")

print("\nTarget labels:")
for label in sorted(target_labels):
    print(f"- {label}")

target_only_labels = target_labels - source_labels

print("\nTarget-only labels:")

if target_only_labels:
    for label in sorted(target_only_labels):
        print(f"- {label}")
else:
    print("None")


# ============================================================
# CHECK CROSS-DOMAIN OVERLAP BEFORE SPLITTING
# ============================================================

print("\n" + "=" * 70)
print("CHECKING SOURCE <-> TARGET TEXT OVERLAP")
print("=" * 70)

source_target_overlap = check_overlap(
    source_df, target_df, "Full Source", "Full Target"
)

if source_target_overlap:
    print("\nWARNING:")

    print(
        f"{len(source_target_overlap)} normalized exact-text "
        "overlaps were found between the source and target datasets."
    )

    print(
        "The script will continue to create the splits, but the "
        "target test set will be checked separately before the "
        "outputs are accepted."
    )
else:
    print(
        "\nNo exact normalized text overlap was found "
        "between the full source and target datasets."
    )


# ============================================================
# CHECK NORMALIZED SOURCE TEXT DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING NORMALIZED SOURCE TEXT DUPLICATES")
print("=" * 70)

source_df["_leakage_key"] = source_df["text"].apply(normalize_text_for_leakage)

normalized_duplicate_mask = source_df["_leakage_key"].duplicated(keep=False)

normalized_duplicate_df = source_df[normalized_duplicate_mask].copy()

duplicate_groups = normalized_duplicate_df["_leakage_key"].nunique()

duplicate_rows = len(normalized_duplicate_df)

print(f"Normalized duplicate groups: " f"{duplicate_groups}")

print(f"Rows involved: " f"{duplicate_rows}")

normalized_conflicts = normalized_duplicate_df.groupby("_leakage_key")[
    "label"
].nunique()

conflicting_normalized_groups = int((normalized_conflicts > 1).sum())

print(
    f"Normalized duplicate groups with "
    f"different labels: "
    f"{conflicting_normalized_groups}"
)

if conflicting_normalized_groups > 0:
    print("\nWARNING:")

    print(
        f"{conflicting_normalized_groups} normalized duplicate "
        "groups contain different labels."
    )

    print(
        "These groups will be kept together in the source "
        "training set to prevent train-validation leakage."
    )


# ============================================================
# CREATE LEAKAGE-SAFE SOURCE TRAIN / VALIDATION SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CREATING LEAKAGE-SAFE SOURCE SPLITS")
print("=" * 70)

# Rows belonging to normalized duplicate groups are kept
# together in the training set so they cannot leak into
# source validation.

source_duplicate_rows_df = (
    source_df[
        source_df["_leakage_key"].isin(
            normalized_duplicate_df["_leakage_key"].unique()
        )
    ]
    .copy()
)

# Remaining rows have unique normalized text keys.

source_unique_rows_df = (
    source_df[
        ~source_df["_leakage_key"].isin(
            normalized_duplicate_df["_leakage_key"].unique()
        )
    ]
    .copy()
)

print(
    f"Rows in normalized duplicate groups: "
    f"{len(source_duplicate_rows_df)}"
)

print(
    f"Rows with unique normalized text: "
    f"{len(source_unique_rows_df)}"
)

# We want approximately 15% of the FULL source dataset
# to become validation.

desired_validation_rows = round(
    source_total_rows * SOURCE_VALIDATION_SIZE
)

# Calculate validation fraction needed from only the
# unique-text rows.

unique_validation_fraction = (
    desired_validation_rows
    / len(source_unique_rows_df)
)

print(
    f"Desired source validation rows: "
    f"{desired_validation_rows}"
)

print(
    f"Validation fraction applied to unique rows: "
    f"{unique_validation_fraction:.6f}"
)

source_unique_train_df, source_validation_df = (
    train_test_split(
        source_unique_rows_df,
        test_size=unique_validation_fraction,
        random_state=RANDOM_SEED,
        stratify=source_unique_rows_df["label"],
        shuffle=True
    )
)

# Add every normalized duplicate group to training.

source_train_df = pd.concat(
    [
        source_unique_train_df,
        source_duplicate_rows_df
    ],
    ignore_index=True
)

# Reproducibly shuffle source training rows.

source_train_df = (
    source_train_df
    .sample(
        frac=1,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)

source_validation_df = (
    source_validation_df
    .reset_index(drop=True)
)

print(
    f"Source train: "
    f"{len(source_train_df)} rows"
)

print(
    f"Source validation: "
    f"{len(source_validation_df)} rows"
)


# ============================================================
# CREATE TARGET TRAIN / VALIDATION / TEST SPLITS
# ============================================================

print("\n" + "=" * 70)
print("CREATING TARGET SPLITS")
print("=" * 70)

# First split:
# 70% train
# 30% temporary validation + test

target_train_df, target_temp_df = train_test_split(
    target_df,
    train_size=TARGET_TRAIN_SIZE,
    test_size=(TARGET_VALIDATION_SIZE + TARGET_TEST_SIZE),
    random_state=RANDOM_SEED,
    stratify=target_df["label"],
    shuffle=True,
)

# Second split:
# temporary 30% becomes 15% validation + 15% test

target_validation_df, target_test_df = train_test_split(
    target_temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=target_temp_df["label"],
    shuffle=True,
)

target_train_df = target_train_df.reset_index(drop=True)

target_validation_df = target_validation_df.reset_index(drop=True)

target_test_df = target_test_df.reset_index(drop=True)

print(f"Target train: " f"{len(target_train_df)} rows")

print(f"Target validation: " f"{len(target_validation_df)} rows")

print(f"Target test: " f"{len(target_test_df)} rows")


# ============================================================
# VERIFY ROW PRESERVATION
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING ROW PRESERVATION")
print("=" * 70)

source_split_total = len(source_train_df) + len(source_validation_df)

target_split_total = (
    len(target_train_df) + len(target_validation_df) + len(target_test_df)
)

if source_split_total != source_total_rows:
    raise ValueError("Source row count changed during splitting.")

if target_split_total != target_total_rows:
    raise ValueError("Target row count changed during splitting.")

print(f"Source rows preserved: " f"{source_split_total} / " f"{source_total_rows}")

print(f"Target rows preserved: " f"{target_split_total} / " f"{target_total_rows}")


# ============================================================
# CHECK WITHIN-DOMAIN LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING WITHIN-DOMAIN DATA LEAKAGE")
print("=" * 70)

source_train_validation_overlap = check_overlap(
    source_train_df, source_validation_df, "Source Train", "Source Validation"
)

target_train_validation_overlap = check_overlap(
    target_train_df, target_validation_df, "Target Train", "Target Validation"
)

target_train_test_overlap = check_overlap(
    target_train_df, target_test_df, "Target Train", "Target Test"
)

target_validation_test_overlap = check_overlap(
    target_validation_df, target_test_df, "Target Validation", "Target Test"
)

within_domain_leakage = (
    len(source_train_validation_overlap)
    + len(target_train_validation_overlap)
    + len(target_train_test_overlap)
    + len(target_validation_test_overlap)
)

if within_domain_leakage > 0:
    raise ValueError(
        "Exact-text leakage was detected between " "train/validation/test splits."
    )

print("\nNo within-domain exact-text leakage detected.")


# ============================================================
# CHECK TARGET TEST AGAINST SOURCE DATA
# ============================================================

print("\n" + "=" * 70)
print("CHECKING TARGET TEST AGAINST SOURCE DATA")
print("=" * 70)

source_train_target_test_overlap = check_overlap(
    source_train_df, target_test_df, "Source Train", "Target Test"
)

source_validation_target_test_overlap = check_overlap(
    source_validation_df, target_test_df, "Source Validation", "Target Test"
)

cross_domain_test_overlap = len(source_train_target_test_overlap) + len(
    source_validation_target_test_overlap
)

if cross_domain_test_overlap > 0:
    raise ValueError(
        "\nTarget-test leakage detected.\n"
        "One or more target test texts also appear in "
        "the source dataset.\n"
        "Do not proceed with these splits until the "
        "overlapping records have been reviewed."
    )

print(
    "\nNo target-test text appears in the "
    "source train or source validation datasets."
)


# ============================================================
# CREATE MIXED TRAINING DATASET
# ============================================================

print("\n" + "=" * 70)
print("CREATING MIXED TRAINING DATASET")
print("=" * 70)

# Remove temporary leakage-check column.
# It is not part of the model-ready dataset.

source_train_df = source_train_df.drop(
    columns=["_leakage_key"],
    errors="ignore"
)

source_validation_df = source_validation_df.drop(
    columns=["_leakage_key"],
    errors="ignore"
)

mixed_train_df = pd.concat([source_train_df, target_train_df], ignore_index=True)

# Reproducible shuffle

mixed_train_df = mixed_train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(
    drop=True
)

expected_mixed_rows = len(source_train_df) + len(target_train_df)

if len(mixed_train_df) != expected_mixed_rows:
    raise ValueError("Mixed training row count is incorrect.")

print(f"Mixed train: " f"{len(mixed_train_df)} rows")

print(f"  Source rows: " f"{len(source_train_df)}")

print(f"  Target rows: " f"{len(target_train_df)}")


# ============================================================
# VERIFY MIXED TRAIN DOES NOT CONTAIN TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING MIXED TRAIN AGAINST TARGET TEST")
print("=" * 70)

mixed_target_test_overlap = check_overlap(
    mixed_train_df, target_test_df, "Mixed Train", "Target Test"
)

if mixed_target_test_overlap:
    raise ValueError("Target test leakage detected in mixed_train.csv.")

print("\nMixed training data contains no " "target test examples.")


# ============================================================
# FINAL DATASET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION")
print("=" * 70)

datasets_to_validate = {
    "Source Train": source_train_df,
    "Source Validation": source_validation_df,
    "Target Train": target_train_df,
    "Target Validation": target_validation_df,
    "Target Test": target_test_df,
    "Mixed Train": mixed_train_df,
}

for dataset_name, dataset_df in datasets_to_validate.items():
    validate_dataset(dataset_df, dataset_name)

    print(f"{dataset_name}: passed validation.")


# ============================================================
# PRINT CLASS DISTRIBUTIONS
# ============================================================

for dataset_name, dataset_df in datasets_to_validate.items():
    print_label_distribution(dataset_df, dataset_name)


# ============================================================
# SAVE DATASET SPLITS
# ============================================================

print("\n" + "=" * 70)
print("SAVING DATASET SPLITS")
print("=" * 70)

source_train_df.to_csv(SOURCE_TRAIN_PATH, index=False)

source_validation_df.to_csv(SOURCE_VALIDATION_PATH, index=False)

target_train_df.to_csv(TARGET_TRAIN_PATH, index=False)

target_validation_df.to_csv(TARGET_VALIDATION_PATH, index=False)

target_test_df.to_csv(TARGET_TEST_PATH, index=False)

mixed_train_df.to_csv(MIXED_TRAIN_PATH, index=False)

print(f"Saved: {SOURCE_TRAIN_PATH}")

print(f"Saved: {SOURCE_VALIDATION_PATH}")

print(f"Saved: {TARGET_TRAIN_PATH}")

print(f"Saved: {TARGET_VALIDATION_PATH}")

print(f"Saved: {TARGET_TEST_PATH}")

print(f"Saved: {MIXED_TRAIN_PATH}")


# ============================================================
# CREATE DATASET SPLIT SUMMARY
# ============================================================

split_summary_rows = []

split_definitions = [
    ("source_train", source_train_df, "source", SOURCE_TRAIN_SIZE),
    ("source_validation", source_validation_df, "source", SOURCE_VALIDATION_SIZE),
    ("target_train", target_train_df, "target", TARGET_TRAIN_SIZE),
    ("target_validation", target_validation_df, "target", TARGET_VALIDATION_SIZE),
    ("target_test", target_test_df, "target", TARGET_TEST_SIZE),
    ("mixed_train", mixed_train_df, "mixed", None),
]

for dataset_name, dataset_df, domain, planned_fraction in split_definitions:

    split_summary_rows.append(
        {
            "dataset": dataset_name,
            "domain": domain,
            "rows": len(dataset_df),
            "planned_fraction": planned_fraction,
            "random_seed": RANDOM_SEED,
            "number_of_classes": dataset_df["label"].nunique(),
        }
    )


split_summary_df = pd.DataFrame(split_summary_rows)

split_summary_df.to_csv(SPLIT_SUMMARY_PATH, index=False)

print(f"\nSplit summary saved to: " f"{SPLIT_SUMMARY_PATH}")


# ============================================================
# CREATE CLASS DISTRIBUTION SUMMARY
# ============================================================

class_distribution_rows = []

for dataset_name, dataset_df in datasets_to_validate.items():
    add_distribution_rows(class_distribution_rows, dataset_df, dataset_name)

class_distribution_df = pd.DataFrame(class_distribution_rows)

class_distribution_df.to_csv(CLASS_DISTRIBUTION_PATH, index=False)

print(f"Class distribution summary saved to: " f"{CLASS_DISTRIBUTION_PATH}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET PREPARATION COMPLETE")
print("=" * 70)

print(f"Random seed: {RANDOM_SEED}")

print("\nSource split:")
print(f"- Train: " f"{len(source_train_df)}")
print(f"- Validation: " f"{len(source_validation_df)}")

print("\nTarget split:")
print(f"- Train: " f"{len(target_train_df)}")
print(f"- Validation: " f"{len(target_validation_df)}")
print(f"- Test: " f"{len(target_test_df)}")

print("\nMixed training set:")
print(f"- Total: " f"{len(mixed_train_df)}")
print(f"- Source contribution: " f"{len(source_train_df)}")
print(f"- Target contribution: " f"{len(target_train_df)}")

print("\nTarget-only classes:")
if target_only_labels:
    for label in sorted(target_only_labels):
        print(f"- {label}")
else:
    print("- None")

print("\nTarget test set is reserved for " "final model evaluation only.")
