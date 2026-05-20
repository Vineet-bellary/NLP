# Fake News Detection System

A classical NLP project for detecting fake news articles using TF-IDF vectorization and sparse linear machine learning models.

This project focuses on:

- practical NLP engineering
- reusable ML pipelines
- model comparison
- error analysis
- gradual engineering maturity without over-engineering

---

# Project Goals

The main objectives of this project were:

- build a complete classical NLP pipeline
- understand sparse text representations
- compare linear models for NLP
- practice reusable ML engineering workflows
- improve experimentation and evaluation habits

---

# Features

- Dataset auditing and cleaning
- Text preprocessing pipeline
- TF-IDF vectorization
- Model comparison framework
- Error analysis pipeline
- Model persistence using `joblib`
- Reusable evaluation utilities
- Structured logging system

---

# Models Used

## Logistic Regression

- strong sparse baseline
- balanced performance
- interpretable linear model

## Multinomial Naive Bayes

- extremely fast
- lightweight probabilistic model
- strong classical NLP baseline

## Linear SVM

- excellent sparse text classifier
- historically dominant in classical NLP tasks
- strongest performing model in this project

---

# Dataset

Dataset used:

Fake and Real News Dataset from Kaggle.

The dataset contains:

- fake news articles
- real news articles
- article titles
- article text
- subject categories

---

# NLP Pipeline

```
Raw Text
    ↓
Preprocessing
    ↓
Train/Test Split
    ↓
TF-IDF Vectorization
    ↓
Sparse Feature Matrix
    ↓
Model Training
    ↓
Evaluation
    ↓
Error Analysis
```

---

# Project Structure

```text
project/
├── data/
├── logs/
├── models/
├── notebooks/
├── reports/
├── tests/
├── src/
│   └── fake_news_detector/
│       ├── compare_models.py
│       ├── error_analysis.py
│       ├── predict.py
│       ├── train.py
│       ├── config.py
│       └── util/
│           ├── data_audit.py
│           ├── data_prep.py
│           ├── evaluate.py
│           ├── load_data.py
│           ├── logger.py
│           ├── models.py
│           ├── preprocessing.py
│           └── vectorizer.py
├── pyproject.toml
└── README.md
```

---

# Preprocessing

The preprocessing pipeline includes:

- lowercase conversion
- punctuation handling
- whitespace normalization

The cleaned text is stored separately as:

```python
cleaned_text
```

to preserve original raw text.

---

# Vectorization

TF-IDF Vectorizer configuration:

```python
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    max_features=100000
)
```

Key concepts learned:

- sparse matrices
- vocabulary explosion
- n-gram feature engineering
- data leakage prevention

---

# Model Evaluation

Evaluation metrics used:

- Accuracy
- Precision
- Recall
- F1-score
- Macro Average
- Weighted Average
- Confusion Matrix

---

# Final Results

| Model               | Accuracy | Macro F1 |
| ------------------- | -------- | -------- |
| Logistic Regression | 0.98     | 0.98     |
| Naive Bayes         | 0.94     | 0.94     |
| Linear SVM          | 0.99     | 0.99     |

---

# Important Learnings

## NLP Learnings

- classical NLP models learn statistical language patterns
- fake news detection is not true factual reasoning
- TF-IDF creates very high-dimensional sparse spaces
- preprocessing strongly affects vocabulary quality

## ML Learnings

- Linear SVM performs extremely well on sparse NLP tasks
- different models fail differently
- confusion matrix analysis is critical
- train/test split must happen before vectorization

## Engineering Learnings

- reusable utilities improve experimentation
- logging helps debugging and experiment tracking
- modularization should happen gradually
- vectorizers are part of trained model pipelines

---

# Error Analysis

Error analysis was performed by:

- inspecting false positives
- inspecting false negatives
- analyzing model failure patterns

This helped reveal:

- stylistic learning behavior
- emotional language bias
- dataset limitations
- preprocessing edge cases

---

# Running the Project

## Create virtual environment

```bash
uv venv
```

## Activate environment

### Windows

```bash
.venv\Scripts\activate
```

## Install dependencies

```bash
uv sync
```

---

# Train Single Model

```bash
uv run python src/fake_news_detector/train.py
```

---

# Compare Models

```bash
uv run python src/fake_news_detector/compare_models.py
```

---

# Run Prediction Pipeline

```bash
uv run python src/fake_news_detector/predict.py
```

---

# Run Error Analysis

```bash
uv run python src/fake_news_detector/error_analysis.py
```

---

# Future Improvements

- advanced preprocessing
- lemmatization
- hyperparameter tuning
- feature importance analysis
- transformer-based models
- deployment API
- experiment tracking

---
