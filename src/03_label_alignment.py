# 03_label_alignment.py
"""
Purpose:
Create a common classification label space between the source and target datasets.

Main tasks:
- Inspect the source and target emotion labels
- Define the approved label-alignment strategy
- Map or filter labels consistently
- Standardize label names
- Standardize dataset columns to:
    text
    label
    domain
- Verify that source and target use the same final label set

Important:
Do not invent label mappings without team agreement and justification.

Outputs:
- source_aligned.csv
- target_aligned.csv

Owner:
Data Pipeline Member
"""