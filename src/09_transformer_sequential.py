# 09_transformer_sequential.py
"""
Purpose:
Perform sequential transfer learning from general-domain emotion data to the
financial-domain dataset.

Method:
1. Start with pretrained BERT
2. Fine-tune using source/general data
3. Continue fine-tuning using target financial data
4. Evaluate on the same target test set

Research role:
Primary transfer-learning experiment.

Outputs:
- Evaluation metrics
- Predictions
- Training history
- Model checkpoint if required

Owner:
BERT / Transfer Learning Member
"""