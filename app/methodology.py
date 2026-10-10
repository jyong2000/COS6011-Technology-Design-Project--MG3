import streamlit as st
import pandas as pd


def show_methodology():
    st.header("About / Methodology")
    st.caption(
        "COS60011 Technology Design Project | "
        "Cross-Domain Financial Emotion Classification"
    )

    # --------------------------------------------------
    # 1. PROJECT OVERVIEW
    # --------------------------------------------------
    st.subheader("1. Project Overview")

    st.write(
        "This project investigates transfer learning and domain "
        "adaptation for financial-domain emotion classification. "
        "It examines whether knowledge learned from general-domain "
        "text can be transferred effectively to financial text, "
        "where vocabulary, context, and emotional expressions differ."
    )

    st.write(
        "The overall project compares three modelling families: "
        "traditional machine learning, BERT-based models, and "
        "large language model (LLM) prompting approaches."
    )

    st.markdown("**Research Objectives**")
    st.markdown("""
        - Prepare and align general-domain and financial-domain datasets.
        - Establish traditional machine-learning baselines.
        - Investigate four BERT training and transfer strategies.
        - Investigate zero-shot, few-shot, and structured LLM prompting.
        - Evaluate models using a consistent financial-domain test set.
        - Analyse class-specific errors and domain-transfer limitations.
        """)

    # --------------------------------------------------
    # 2. DATASET SELECTION
    # --------------------------------------------------
    st.subheader("2. Dataset Selection and Exploration")

    dataset_df = pd.DataFrame(
        {
            "Characteristic": [
                "Dataset",
                "Source",
                "Original Records",
                "Original Emotion Classes",
                "Duplicate Texts",
                "Conflicting Texts",
                "Mean Words",
            ],
            "General Domain": [
                "Text Emotion",
                "Kaggle",
                "106,355",
                "11",
                "25,642",
                "29",
                "20.24",
            ],
            "Financial Domain": [
                "Finance Emotions",
                "Hugging Face",
                "10,000",
                "7",
                "0",
                "0",
                "15.53",
            ],
        }
    )

    st.dataframe(dataset_df, hide_index=True, width="stretch")

    st.write(
        "Exploratory data analysis examined class distributions, "
        "missing values, duplicate and conflicting texts, and "
        "text lengths to identify differences between the domains."
    )

    # --------------------------------------------------
    # 3. PREPROCESSING
    # --------------------------------------------------
    st.subheader("3. Data Preprocessing")

    st.write(
        "Conservative text preprocessing was applied to preserve "
        "emotion-related information. Unnecessary whitespace was "
        "normalised while punctuation, numbers, emojis, ticker "
        "symbols, and other meaningful content were retained."
    )

    preprocessing_df = pd.DataFrame(
        {
            "Dataset": ["Source", "Target"],
            "Initial Records": [106355, 10000],
            "Conflicting Removed": [160, 0],
            "Duplicates Removed": [25511, 0],
            "Final Records": [80684, 10000],
        }
    )

    st.dataframe(
        preprocessing_df,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------
    # 4. LABEL ALIGNMENT
    # --------------------------------------------------
    st.subheader("4. Label Taxonomy Alignment")

    st.write(
        "The source dataset originally contained 11 emotion labels, "
        "while the target dataset contained seven. A deterministic "
        "semantic mapping aligned the source labels with five "
        "target-compatible emotions."
    )

    alignment_df = pd.DataFrame(
        {
            "Target Emotion": [
                "Anger",
                "Happy",
                "Neutral",
                "Sad",
                "Surprise",
                "Fear",
                "Disgust",
            ],
            "Original Source Labels": [
                "anger, hate",
                "happiness, fun, love, enthusiasm, relief",
                "neutral, empty",
                "sadness",
                "surprise",
                "No source equivalent",
                "No source equivalent",
            ],
        }
    )

    st.dataframe(
        alignment_df,
        hide_index=True,
        width="stretch",
    )

    st.warning(
        "Fear and Disgust are absent from the aligned source "
        "dataset but remain in the financial target dataset. "
        "All models retain the seven-class classification task."
    )

    st.markdown(
        "**Seven target emotions:** "
        "Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise."
    )

    # --------------------------------------------------
    # 5. DATASET PARTITIONING
    # --------------------------------------------------
    st.subheader("5. Dataset Partitioning and Leakage Prevention")

    st.write(
        "The aligned source dataset was divided into 85% training "
        "and 15% validation. The financial target dataset was "
        "divided into 70% training, 15% validation, and 15% test "
        "using stratified sampling and a fixed random seed."
    )

    splits_df = pd.DataFrame(
        {
            "Dataset Split": [
                "Source Train",
                "Source Validation",
                "Target Train",
                "Target Validation",
                "Target Test",
                "Mixed Train",
            ],
            "Records": [68581, 12103, 7000, 1500, 1500, 75581],
            "Purpose": [
                "Source-domain training",
                "Source-domain validation",
                "Financial-domain training",
                "Model validation",
                "Final evaluation only",
                "Combined source and target training",
            ],
        }
    )

    st.dataframe(
        splits_df,
        hide_index=True,
        width="stretch",
    )

    st.info(
        "The Target Test set contains 1,500 examples and is "
        "locked for final evaluation. Normalised text overlap "
        "checks found no overlap between training data and "
        "the Target Test set."
    )

    # --------------------------------------------------
    # 6. MODELLING APPROACHES
    # --------------------------------------------------
    st.subheader("6. Modelling Approaches")

    st.write("The team design investigates three modelling families.")

    with st.expander("Traditional Machine Learning"):
        st.write(
            "Traditional machine-learning baselines use TF-IDF "
            "features with unigrams and bigrams."
        )
        st.markdown("""
            - Logistic Regression
            - Linear Support Vector Machine (SVM)
            - Naive Bayes
            """)
        st.write(
            "The design compares source-only, target-only, "
            "and mixed-domain training configurations."
        )

    with st.expander("BERT and Transfer Learning", expanded=True):
        st.write(
            "Four training strategies use the pretrained "
            "BERT-base-uncased model for seven-class "
            "financial emotion classification."
        )
        st.write(
            "All four BERT strategies are evaluated in the Model Comparison tab. "
            "Dataset Weighting is selected for interactive inference in the Emotion Classifier tab."
        )

    with st.expander("LLM Prompting Approaches"):
        st.write(
            "The design uses FLAN-T5-base without task-specific "
            "fine-tuning to investigate three approaches:"
        )
        st.markdown("""
            - **Zero-shot:** Classify text using instructions
              and seven permitted emotion labels.
            - **Few-shot:** Include one labelled example
              per emotion class from Target Train.
            - **Structured prompting:** Require a predefined
              prediction response format.
            """)

    # --------------------------------------------------
    # 7. BERT TRAINING STRATEGIES
    # --------------------------------------------------
    st.subheader("7. BERT Training Strategies")

    st.write(
        "All four BERT models use google-bert/bert-base-uncased "
        "and a seven-class classification head. Text is tokenised "
        "using a maximum sequence length of 128 tokens."
    )

    strategies = {
        "Domain Only": (
            "Fine-tunes BERT using Target Train and validates "
            "on Target Validation. This provides a "
            "financial-domain baseline."
        ),
        "Sequential Transfer Learning": (
            "Fine-tunes BERT on general-domain source data "
            "first, then further fine-tunes it using "
            "financial-domain target data."
        ),
        "Mixed Training": (
            "Combines Source Train and Target Train into "
            "one training dataset, using Target Validation "
            "for model validation."
        ),
        "Dataset Weighting": (
            "Combines source and target training examples, "
            "applying training-loss weights of 1.0 to "
            "source examples and 2.0 to target examples."
        ),
    }

    for name, description in strategies.items():
        with st.expander(name):
            st.write(description)

    st.info(
        "Deployment selection: Dataset Weighting achieved the highest test macro-F1 "
        "among the four evaluated BERT strategies (50.06%). "
        "Sequential Transfer Learning achieved the highest test accuracy (64.13%). "
        "The differences are small, and this selection does not establish the "
        "best approach across traditional ML and LLM models."
    )

    # --------------------------------------------------
    # 8. WORKFLOW
    # --------------------------------------------------
    st.subheader("8. Project Workflow")

    st.markdown("""
        **Dataset Selection and Exploration**

        ↓

        **Data Preprocessing**

        ↓

        **Label Taxonomy Alignment**

        ↓

        **Dataset Partitioning and Leakage Prevention**

        ↓

        **Traditional ML | BERT | LLM Prompting**

        ↓

        **Evaluation on Locked Target Test Set**

        ↓

        **Confusion Matrices and Error Analysis**

        ↓

        **Comparison of Modelling Approaches**
        """)

    # --------------------------------------------------
    # 9. EVALUATION
    # --------------------------------------------------
    st.subheader("9. Model Evaluation and Error Analysis")

    st.write(
        "The design evaluates all modelling approaches on the "
        "same locked financial Target Test set using accuracy, "
        "precision, recall, macro-F1, and weighted-F1."
    )

    st.info(
        "Macro-F1 is the primary final comparison metric "
        "because it gives equal importance to each emotion "
        "class despite class imbalance."
    )

    st.write(
        "Confusion matrices, per-class precision, recall, "
        "F1-scores, and misclassified examples are used "
        "to investigate class-specific weaknesses and "
        "differences between modelling approaches."
    )

    st.caption(
        "The current BERT checkpoints were selected using "
        "validation weighted-F1. The Model Comparison dashboard "
        "emphasises test macro-F1."
    )

    # --------------------------------------------------
    # 10. DESIGN CONSTRAINTS
    # --------------------------------------------------
    st.subheader("10. Design Constraints and Limitations")

    st.markdown("""
        - **Domain shift:** General and financial text differ
          in vocabulary, context, and emotional expression.
        - **Dataset imbalance:** The larger source dataset
          may dominate mixed-domain training.
        - **Incomplete label coverage:** Fear and Disgust
          have no equivalents in the aligned source dataset.
        - **Computational resources:** BERT fine-tuning
          requires substantial processing time and memory.
        - **Prediction uncertainty:** Softmax confidence
          is not a guarantee of classification correctness.
        """)

    # --------------------------------------------------
    # 11. APPLICATION SCOPE
    # --------------------------------------------------
    st.subheader("11. Streamlit Application Scope")

    st.write(
        "This application demonstrates the BERT component " "of the wider team project."
    )

    st.markdown("""
        **Emotion Classifier**
        - Interactive predictions using the selected Dataset Weighting BERT model.
        - Displays predicted emotion, confidence, and
          seven-class probability scores.

        **Model Comparison**
        - Compares all four evaluated BERT strategies on the locked Target Test set.
        - Displays evaluation metrics, per-class results,
          confusion matrices, and misclassified examples.

        **About / Methodology**
        - Documents the wider research design and explains
          the implemented BERT component.
        """)

    st.caption(
        "Traditional ML and LLM approaches are part of the "
        "team research design but are not currently available "
        "for interactive inference in this Streamlit application."
    )
