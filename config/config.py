# config/config.py
"""
Shared project configuration.

Purpose:
Store settings that must remain consistent across team members.
"""

RANDOM_SEED = 42

BASE_MODEL = "google-bert/bert-base-uncased"

# Dataset paths
SOURCE_RAW_PATH = "data/Source Dataset.csv"
TARGET_RAW_PATH = "data/Target Dataset.csv"

SOURCE_CLEAN_PATH = "data/processed/source_clean.csv"
TARGET_CLEAN_PATH = "data/processed/target_clean.csv"

SOURCE_ALIGNED_PATH = "data/processed/source_aligned.csv"
TARGET_ALIGNED_PATH = "data/processed/target_aligned.csv"

SOURCE_TRAIN_PATH = "data/processed/source_train.csv"
SOURCE_VALIDATION_PATH = "data/processed/source_validation.csv"

TARGET_TRAIN_PATH = "data/processed/target_train.csv"
TARGET_VALIDATION_PATH = "data/processed/target_validation.csv"
TARGET_TEST_PATH = "data/processed/target_test.csv"

MIXED_TRAIN_PATH = "data/processed/mixed_train.csv"

# Results folders
FIGURES_DIR = "results/figures"
METRICS_DIR = "results/metrics"
PREDICTIONS_DIR = "results/predictions"
TABLES_DIR = "results/tables"