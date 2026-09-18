# 07_train_naive_bayes.py
"""
Purpose:
Train and evaluate Naive Bayes classifiers.

Experiments:
A - Source/general training only
B - Target/domain training only
C - Mixed source + target training

Method:
- TF-IDF text representation
- Naive Bayes classifier

Outputs:
- Model metrics
- Predictions on the target test set

Owner:
Traditional ML Member
"""

from sklearn.naive_bayes import MultinomialNB, ComplementNB

from ml_utils import run_all_experiments


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "naive_bayes"


# A placeholder. The grid below swaps in the actual Naive Bayes variant, so
# whatever is set here is replaced during the search.
CLASSIFIER = MultinomialNB()


# Two variants are compared:
#
# MultinomialNB : the classic text Naive Bayes.
# ComplementNB  : a version designed specifically for imbalanced text data.
#                 It usually beats MultinomialNB on emotion datasets, where
#                 some emotions have far fewer examples than others.
#
# alpha is smoothing. It decides what probability an unseen word gets. A
# larger alpha smooths more, which matters most in Experiment A, where many
# words in the target test set never appeared in the source training data.
#
# Note on theory: Naive Bayes technically assumes word COUNTS, and TF-IDF
# values are not counts. Using them anyway is standard practice and works
# because the maths only requires non-negative feature values. Mention this
# in the report as a known limitation of this baseline.
PARAMETER_GRID = {
    "classifier": [MultinomialNB(), ComplementNB()],
    "classifier__alpha": [0.05, 0.1, 0.5, 1.0],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
}


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    run_all_experiments(MODEL_NAME, CLASSIFIER, PARAMETER_GRID)
