# 06_train_svm.py
"""
Purpose:
Train and evaluate Support Vector Machine classifiers.

Experiments:
A - Source/general training only
B - Target/domain training only
C - Mixed source + target training

Method:
- TF-IDF text representation
- Linear SVM classifier

Outputs:
- Model metrics
- Predictions on the target test set

Owner:
Traditional ML Member
"""

from sklearn.svm import LinearSVC

from ml_utils import run_all_experiments


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "linear_svm"


# A LINEAR kernel is used, not RBF.
#
# TF-IDF produces a very high-dimensional, sparse representation, and text
# data in that form is already close to linearly separable. An RBF kernel
# would be far slower and would rarely do better. Say this in the report so
# the choice does not look arbitrary.
#
# Note: LinearSVC has no predict_proba, so the saved prediction files for
# this model will not contain probability columns. That is expected.
CLASSIFIER = LinearSVC(
    max_iter=5000,
    random_state=42
)


# C for an SVM works the same way as for logistic regression: small C means
# a wider margin and a simpler model. SVMs on text usually prefer a smaller
# C than logistic regression does, which is why the values here are lower.
PARAMETER_GRID = {
    "classifier__C": [0.01, 0.1, 0.5, 1.0],
    "classifier__class_weight": [None, "balanced"],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_all_experiments(MODEL_NAME, CLASSIFIER, PARAMETER_GRID)
