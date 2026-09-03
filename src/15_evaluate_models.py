# 15_evaluate_models.py
"""
Purpose:
Evaluate all trained and prompted models using one consistent evaluation
framework.

Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Main tasks:
- Load predictions from all experiments
- Compare them against the same target test labels
- Calculate consistent metrics
- Generate comparison tables
- Generate confusion matrices
- Rank and compare models

Outputs:
- metrics comparison table
- confusion matrix figures
- summary results

Owner:
LLM / Evaluation Member
"""