# 13_llm_few_shot.py
"""
Purpose:
Evaluate an LLM using few-shot prompting.

Main tasks:
- Define representative labelled examples
- Build a consistent few-shot prompt
- Classify the target test data
- Save predictions and responses

Important:
Few-shot examples must not come from the locked target test set.

Outputs:
- Predictions
- Prompt configuration
- Raw responses

Owner:
LLM / Evaluation Member
"""