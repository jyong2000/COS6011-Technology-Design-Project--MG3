# 12_llm_zero_shot.py
"""
Purpose:
Evaluate an LLM on the financial emotion classification task using zero-shot
prompting.

Main tasks:
- Create a consistent zero-shot prompt
- Provide the allowed emotion labels
- Classify target test examples
- Save raw model responses
- Convert responses into standardized labels

Important:
Use the same target test examples used by the trained models.

Outputs:
- Predictions
- Prompt configuration
- Raw responses if required

Owner:
LLM / Evaluation Member
"""