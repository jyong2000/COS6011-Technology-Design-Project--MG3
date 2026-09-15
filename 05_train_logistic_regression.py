# 05_train_logistic_regression.py
"""
Purpose:
Train and evaluate Logistic Regression sentiment/emotion classifiers.

Experiments:
A - Source/general training only
B - Target/domain training only
C - Mixed source + target training

Method:
- TF-IDF text representation
- Logistic Regression classifier

Outputs:
- Model metrics
- Predictions on the target test set
- Experiment configuration

Owner:
Traditional ML Member
"""

from sklearn.linear_model import LogisticRegression

from ml_utils import run_all_experiments


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "logistic_regression"


# The classifier itself.
#
# scikit-learn uses the multinomial (softmax) version for this solver, which
# is the right choice here: a message has exactly one emotion label, so the
# six classes compete against each other rather than being six separate
# yes/no decisions.
CLASSIFIER = LogisticRegression(
    solver="saga",
    max_iter=3000,
    random_state=42
)


# The settings that GridSearchCV will try.
#
# C            : controls regularisation. A SMALL C means stronger
#                regularisation, i.e. the model is forced to stay simple.
#                This usually helps Experiment A, where the model must not
#                memorise vocabulary that only exists in the source domain.
# class_weight : 'balanced' makes rare emotions count more during training,
#                which normally raises macro F1 on an imbalanced dataset.
# ngram_range  : (1, 1) is single words, (1, 2) adds word pairs, which lets
#                the model learn phrases such as "not happy".
#
# This grid has 4 x 2 x 2 x 2 = 32 combinations, each fitted 5 times by
# cross-validation. If that is too slow, shorten the C list to [0.1, 1.0]
# first, since that is the setting that matters most.
PARAMETER_GRID = {
    "classifier__C": [0.1, 1.0, 5.0, 10.0],
    "classifier__class_weight": [None, "balanced"],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_all_experiments(MODEL_NAME, CLASSIFIER, PARAMETER_GRID)
