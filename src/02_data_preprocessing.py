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

Important:
Do not perform label alignment in this script.

Outputs:
- Cleaned source dataset
- Cleaned target dataset

Owner:
Data Pipeline Member
"""