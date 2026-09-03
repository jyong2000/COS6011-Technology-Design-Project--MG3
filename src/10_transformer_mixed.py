# 10_transformer_mixed.py
"""
Purpose:
Fine-tune BERT using a mixed source + target training dataset.

Method:
- Start with the same pretrained BERT model
- Train using mixed_train
- Validate using target_validation
- Evaluate using target_test

Research role:
Tests whether combining general and domain-specific training data improves
financial-domain performance.

Outputs:
- Evaluation metrics
- Predictions
- Training history

Owner:
BERT / Transfer Learning Member
"""