# Domain-Adaptive Sentiment & Public Opinion Analysis

## Project Overview

This project investigates domain-adaptive sentiment and emotion classification using traditional machine learning, Transformer-based models, transfer learning, and Large Language Models (LLMs).

The project examines how the choice of training data affects model performance when transferring knowledge from a large general-domain dataset to a smaller domain-specific financial dataset.

The primary objective is to experimentally evaluate whether domain adaptation improves model performance on specialized text.

## Datasets

This project uses two datasets with different roles.

### Source Dataset — General Domain

**Dataset:** Text Emotion  
**Source:** Kaggle — Prajwal Nayaka T  
**Size:** Approximately 106,000 text samples  
**Domain:** General/social text  
**Task:** Multi-class emotion classification

The dataset contains the following columns:

- `text` — input text
- `emotion` — emotion label

The dataset contains 11 emotion classes:

- anger
- empty
- enthusiasm
- fun
- happiness
- hate
- love
- neutral
- relief
- sadness
- surprise

This dataset is used as the large general-domain source dataset for learning general language and emotion patterns.

### Target Dataset — Financial Domain

**Dataset:** finance_emotions  
**Source:** Hugging Face — vamossyd/finance_emotions  
**Size:** 10,000 text samples  
**Domain:** Financial/social-media text  
**Task:** Multi-class emotion classification

The downloaded dataset contains the following columns:

- `body` — original financial text
- `cleaned_text` — preprocessed version of the text
- `label` — emotion label
- `chatGPT_label` — additional label provided in the dataset

The `label` column contains 7 emotion classes:

- anger
- disgust
- fear
- happy
- neutral
- sad
- surprise

This dataset is used as the smaller domain-specific target dataset for adapting and evaluating models on financial language.

> **Note:** The source and target datasets use different emotion label sets. A common label strategy or another appropriate transfer-learning approach must therefore be established before controlled model comparisons are performed.

## Experimental Approach

The project will investigate several approaches to emotion classification and domain adaptation.

### 1. Data Preparation and Exploratory Data Analysis

Both datasets will be explored and prepared before modelling.

This stage includes:

- Dataset structure analysis
- Missing-value analysis
- Duplicate detection
- Class-distribution analysis
- Text-length analysis
- Text cleaning and preprocessing
- Tokenization
- Dataset visualisation

### 2. Traditional Machine Learning Baselines

Traditional machine learning models will be trained using TF-IDF features.

Models may include:

- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes

The experiments will compare different training-data configurations to establish baseline performance.

### 3. Transformer-Based Modelling

A pretrained Transformer model will be used for emotion classification.

The project will compare:

- Domain-only fine-tuning
- Sequential transfer learning
- Mixed training

Sequential transfer learning will involve training on the general-domain dataset before fine-tuning on the financial-domain dataset.

### 4. Transfer Learning

At least two transfer-learning techniques will be investigated.

Potential techniques include:

- Sequential fine-tuning
- Domain-Adaptive Pretraining (DAPT)
- Dataset weighting
- Few-shot fine-tuning
- Domain tags

### 5. Large Language Model Evaluation

LLM-based emotion classification will also be evaluated without full model training.

Experiments will include:

- Zero-shot prompting
- Few-shot prompting
- Structured prompting

### 6. Evaluation

Model performance will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Qualitative analysis will also investigate prediction failures, domain-specific vocabulary, bias patterns, and cross-domain performance differences.

