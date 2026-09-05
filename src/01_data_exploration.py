import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DATA_PATH = "data/Source Dataset.csv"
TARGET_DATA_PATH = "data/Target Dataset.csv"

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

SOURCE_TEXT_COLUMN = "text"
SOURCE_LABEL_COLUMN = "emotion"

TARGET_RAW_TEXT_COLUMN = "body"
TARGET_CLEAN_TEXT_COLUMN = "cleaned_text"
TARGET_LABEL_COLUMN = "label"
TARGET_CHATGPT_LABEL_COLUMN = "chatGPT_label"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

source_df = pd.read_csv(SOURCE_DATA_PATH)
target_df = pd.read_csv(TARGET_DATA_PATH)

print("Source dataset loaded successfully.")
print("Target dataset loaded successfully.")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_basic_information(df, dataset_name):
    """
    Print basic information about a dataset.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - BASIC INFORMATION")
    print("=" * 70)

    print(f"\nNumber of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nRandom 5 rows:")
    print(df.sample(min(5, len(df)), random_state=42))


def check_missing_values(df, dataset_name):
    """
    Check for missing values in every column.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - MISSING VALUES")
    print("=" * 70)

    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100

    missing_table = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percentage": missing_percentages
    })

    print(missing_table)

    filename = f"{dataset_name.lower().replace(' ', '_')}_missing_values.csv"
    missing_table.to_csv(
        os.path.join(TABLES_DIR, filename),
        index=True
    )


def check_empty_text(df, text_column, dataset_name):
    """
    Check for text entries that exist but contain only whitespace.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - EMPTY TEXT CHECK")
    print("=" * 70)

    valid_text = df[text_column].dropna().astype(str)

    empty_count = valid_text.str.strip().eq("").sum()

    print(f"Whitespace-only text rows in '{text_column}': {empty_count}")


def check_duplicates(df, text_column, label_column, dataset_name):
    """
    Check duplicate rows, duplicate texts, and potentially conflicting labels.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - DUPLICATE ANALYSIS")
    print("=" * 70)

    duplicate_rows = df.duplicated().sum()

    duplicate_texts = df[text_column].duplicated().sum()

    print(f"Exact duplicate rows: {duplicate_rows}")
    print(f"Duplicate text entries: {duplicate_texts}")

    # Check whether the same text has multiple different labels
    valid_df = df[[text_column, label_column]].dropna()

    label_counts_per_text = (
        valid_df.groupby(text_column)[label_column]
        .nunique()
    )

    conflicting_texts = label_counts_per_text[
        label_counts_per_text > 1
    ]

    print(
        f"Texts associated with more than one label: "
        f"{len(conflicting_texts)}"
    )

    if len(conflicting_texts) > 0:
        conflicting_examples = valid_df[
            valid_df[text_column].isin(conflicting_texts.index)
        ].sort_values(text_column)

        conflicting_examples.to_csv(
            os.path.join(
                TABLES_DIR,
                f"{dataset_name.lower().replace(' ', '_')}_conflicting_duplicates.csv"
            ),
            index=False
        )


def analyse_labels(df, label_column, dataset_name):
    """
    Analyse label counts and percentages.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - LABEL DISTRIBUTION")
    print("=" * 70)

    label_counts = df[label_column].value_counts(dropna=False)

    label_percentages = (
        df[label_column]
        .value_counts(normalize=True, dropna=False)
        * 100
    )

    label_table = pd.DataFrame({
        "count": label_counts,
        "percentage": label_percentages
    })

    print("\nUnique labels:")
    print(df[label_column].dropna().unique())

    print("\nLabel distribution:")
    print(label_table)

    label_table.to_csv(
        os.path.join(
            TABLES_DIR,
            f"{dataset_name.lower().replace(' ', '_')}_label_distribution.csv"
        ),
        index=True
    )

    return label_counts


def plot_label_distribution(label_counts, dataset_name):
    """
    Create and save a label-distribution bar chart.
    """

    plt.figure(figsize=(10, 6))

    label_counts.plot(kind="bar")

    plt.title(f"{dataset_name} Label Distribution")
    plt.xlabel("Emotion Label")
    plt.ylabel("Number of Samples")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{dataset_name.lower().replace(' ', '_')}_label_distribution.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(f"Saved figure: {output_path}")


def analyse_text_length(df, text_column, dataset_name):
    """
    Calculate character-count and word-count statistics.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - TEXT LENGTH ANALYSIS")
    print("=" * 70)

    text_series = df[text_column].fillna("").astype(str)

    character_lengths = text_series.str.len()

    word_counts = text_series.str.split().str.len()

    statistics = pd.DataFrame({
        "character_length": character_lengths.describe(),
        "word_count": word_counts.describe()
    })

    print(statistics)

    statistics.to_csv(
        os.path.join(
            TABLES_DIR,
            f"{dataset_name.lower().replace(' ', '_')}_text_length_statistics.csv"
        ),
        index=True
    )

    return character_lengths, word_counts


def plot_text_length_distribution(word_counts, dataset_name):
    """
    Plot the distribution of text lengths measured in words.
    """

    plt.figure(figsize=(10, 6))

    plt.hist(word_counts, bins=50)

    plt.title(f"{dataset_name} Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")

    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"{dataset_name.lower().replace(' ', '_')}_text_length_distribution.png"
    )

    plt.savefig(output_path)
    plt.close()

    print(f"Saved figure: {output_path}")


def inspect_short_and_long_texts(
    df,
    text_column,
    label_column,
    dataset_name
):
    """
    Display examples of shortest and longest text entries.
    """

    print("\n" + "=" * 70)
    print(f"{dataset_name.upper()} - SHORTEST AND LONGEST TEXTS")
    print("=" * 70)

    temp_df = df[[text_column, label_column]].copy()

    temp_df["word_count"] = (
        temp_df[text_column]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("\n5 shortest texts:")
    print(
        temp_df
        .sort_values("word_count")
        .head(5)
        [[text_column, label_column, "word_count"]]
    )

    print("\n5 longest texts:")
    print(
        temp_df
        .sort_values("word_count", ascending=False)
        .head(5)
        [[text_column, label_column, "word_count"]]
    )


def compare_target_text_columns(df):
    """
    Compare raw target text ('body') against 'cleaned_text'.
    """

    print("\n" + "=" * 70)
    print("TARGET DATASET - BODY VS CLEANED_TEXT")
    print("=" * 70)

    comparison_df = df[
        [TARGET_RAW_TEXT_COLUMN, TARGET_CLEAN_TEXT_COLUMN]
    ].copy()

    comparison_df["same_text"] = (
        comparison_df[TARGET_RAW_TEXT_COLUMN].fillna("").astype(str)
        ==
        comparison_df[TARGET_CLEAN_TEXT_COLUMN].fillna("").astype(str)
    )

    same_count = comparison_df["same_text"].sum()
    different_count = (~comparison_df["same_text"]).sum()

    print(f"Rows where body and cleaned_text are identical: {same_count}")
    print(f"Rows where body and cleaned_text differ: {different_count}")

    sample_differences = comparison_df[
        comparison_df["same_text"] == False
    ].head(10)

    print("\nExample differences:")
    print(sample_differences)

    sample_differences.to_csv(
        os.path.join(
            TABLES_DIR,
            "target_body_vs_cleaned_text_examples.csv"
        ),
        index=False
    )


def compare_target_labels(df):
    """
    Compare the official label column with chatGPT_label.
    """

    print("\n" + "=" * 70)
    print("TARGET DATASET - LABEL VS CHATGPT_LABEL")
    print("=" * 70)

    comparison_df = df[
        [TARGET_LABEL_COLUMN, TARGET_CHATGPT_LABEL_COLUMN]
    ].dropna()

    matches = (
        comparison_df[TARGET_LABEL_COLUMN]
        ==
        comparison_df[TARGET_CHATGPT_LABEL_COLUMN]
    )

    agreement_rate = matches.mean() * 100

    print(
        f"Overall label agreement: "
        f"{agreement_rate:.2f}%"
    )

    cross_tab = pd.crosstab(
        comparison_df[TARGET_LABEL_COLUMN],
        comparison_df[TARGET_CHATGPT_LABEL_COLUMN]
    )

    print("\nCross-tabulation:")
    print(cross_tab)

    cross_tab.to_csv(
        os.path.join(
            TABLES_DIR,
            "target_label_vs_chatgpt_label.csv"
        )
    )


# ============================================================
# SOURCE DATASET ANALYSIS
# ============================================================

print_basic_information(
    source_df,
    "Source Dataset"
)

check_missing_values(
    source_df,
    "Source Dataset"
)

check_empty_text(
    source_df,
    SOURCE_TEXT_COLUMN,
    "Source Dataset"
)

check_duplicates(
    source_df,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN,
    "Source Dataset"
)

source_label_counts = analyse_labels(
    source_df,
    SOURCE_LABEL_COLUMN,
    "Source Dataset"
)

plot_label_distribution(
    source_label_counts,
    "Source Dataset"
)

source_character_lengths, source_word_counts = analyse_text_length(
    source_df,
    SOURCE_TEXT_COLUMN,
    "Source Dataset"
)

plot_text_length_distribution(
    source_word_counts,
    "Source Dataset"
)

inspect_short_and_long_texts(
    source_df,
    SOURCE_TEXT_COLUMN,
    SOURCE_LABEL_COLUMN,
    "Source Dataset"
)


# ============================================================
# TARGET DATASET ANALYSIS
# ============================================================

print_basic_information(
    target_df,
    "Target Dataset"
)

check_missing_values(
    target_df,
    "Target Dataset"
)

check_empty_text(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    "Target Dataset - Body"
)

check_empty_text(
    target_df,
    TARGET_CLEAN_TEXT_COLUMN,
    "Target Dataset - Cleaned Text"
)

check_duplicates(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    TARGET_LABEL_COLUMN,
    "Target Dataset"
)

target_label_counts = analyse_labels(
    target_df,
    TARGET_LABEL_COLUMN,
    "Target Dataset"
)

plot_label_distribution(
    target_label_counts,
    "Target Dataset"
)

target_character_lengths, target_word_counts = analyse_text_length(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    "Target Dataset"
)

plot_text_length_distribution(
    target_word_counts,
    "Target Dataset"
)

inspect_short_and_long_texts(
    target_df,
    TARGET_RAW_TEXT_COLUMN,
    TARGET_LABEL_COLUMN,
    "Target Dataset"
)


# ============================================================
# TARGET-SPECIFIC ANALYSIS
# ============================================================

compare_target_text_columns(target_df)

compare_target_labels(target_df)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(
    "\nFigures have been saved to:",
    FIGURES_DIR
)

print(
    "Tables have been saved to:",
    TABLES_DIR
)

print(
    "\nNo dataset rows or values were modified by this script."
)