# 04_prepare_datasets.py
"""
Purpose:
Create the fixed train, validation and test datasets used by all experiments.

Main tasks:
- Load aligned source and target data
- Create reproducible stratified splits
- Lock the target test set
- Create source training/validation data
- Create target training/validation/test data
- Create the mixed source + target training dataset
- Check for data leakage
- Save split statistics

Important:
All team members must use these exact dataset splits.

Outputs:
- source_train.csv
- source_validation.csv
- target_train.csv
- target_validation.csv
- target_test.csv
- mixed_train.csv
- dataset split summaries

Owner:
Data Pipeline Member
"""