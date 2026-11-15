# Saldo — Transaction Classifier

An automatic financial transaction classification system built with **Python**, using **BERTimbau** for text feature extraction and **Logistic Regression** for classification.

This project is part of [Saldo](https://github.com/DovCaio), a personal finance management application. Its purpose is to automatically categorize financial transactions described in Brazilian Portuguese.

## Objective

Classify financial transaction descriptions into nine categories, making it easier to organize and analyze financial activities.

Supported categories:

| Category        | Description                                                    |
| --------------- | -------------------------------------------------------------- |
| `education`     | Education, courses, tuition fees, and learning materials       |
| `emi`           | Loans, financing, and installment payments                     |
| `entertainment` | Entertainment, streaming services, games, and events           |
| `food`          | Food, restaurants, supermarkets, and delivery services         |
| `healthcare`    | Healthcare, medical appointments, examinations, and medication |
| `investment`    | Investments, financial applications, and contributions         |
| `shopping`      | Consumer goods, electronics, clothing, and accessories         |
| `travel`        | Transportation, fuel, tolls, and mobility services             |
| `utilities`     | Electricity, water, internet, and telephone bills              |

## Technologies

- **Python** — Main programming language.
- **PyTorch** — Neural network execution.
- **Hugging Face Transformers** — BERTimbau model integration.
- **BERTimbau Base** — Text embedding extraction.
- **Scikit-learn** — Classification and model evaluation.
- **Pandas / NumPy** — Data manipulation.
- **Matplotlib / Seaborn** — Data visualization.

### Base Model

[neuralmind/bert-base-portuguese-cased](https://huggingface.co/neuralmind/bert-base-portuguese-cased)

BERTimbau is a BERT-based language model pretrained on Brazilian Portuguese text.

In this project, it is used as a **feature extractor**. Its parameters remain frozen during classifier training.

## Architecture

The classification pipeline consists of two main stages:

1. **Feature Extraction:** BERTimbau converts transaction descriptions into numerical vector representations (embeddings).
2. **Classification:** A Logistic Regression model uses these embeddings to predict the transaction category.

```text
             Transaction Description
                       |
                       v
                  Tokenization
                       |
                       v
                    BERTimbau
                       |
                       v
                   Mean Pooling
                       |
                       v
                Embedding (768D)
                       |
                       v
              Logistic Regression
                       |
                       v
                Predicted Category
```

### Model Configuration

| Parameter               | Value               |
| ----------------------- | ------------------- |
| Base model              | BERTimbau Base      |
| Maximum sequence length | 64 tokens           |
| Batch size              | 16                  |
| Pooling strategy        | Mean Pooling        |
| Classifier              | Logistic Regression |
| Regularization C        | 1.0                 |
| Maximum iterations      | 2000                |
| Class weights           | Balanced            |
| Cross-validation        | Stratified K-Fold   |
| Number of folds         | 5                   |
| Random state            | 42                  |

### Mean Pooling

The project uses Mean Pooling to generate a vector representation for each transaction description.

The calculation considers the embeddings of valid tokens, using the `attention_mask` to ignore padding tokens.

This produces a 768-dimensional embedding for each transaction.

## Dataset

The dataset contains Brazilian Portuguese financial transaction descriptions associated with their respective categories.

The current version contains approximately 2,176 records distributed across nine categories.

The data includes bank statement descriptions, merchants, services, and financial operations.

**Data origin:** The dataset combines manually prepared examples and synthetic examples generated with the assistance of large language models (LLMs). Therefore, it should not be considered a statistically representative sample of real-world banking transactions.

### Dataset Structure

```text
data/
└── processed/
    └── transactions_pt_br_v4.csv
```

CSV format:

```csv
description,category
UBER *TRIP HELP,travel
CONSULTA MEDICA,healthcare
AMAZON PRIME VIDEO,entertainment
MENSALIDADE CURSO INGLES,education
APLICACAO CDB,investment
```

## Project Structure

```text
transaction-classifier/
│
├── data/
│   └── processed/
│       └── transactions_pt_br_v4.csv
│
├── models/
│   ├── transaction_classifier.joblib
│   ├── metadata.json
│   ├── confusion_matrix.png
│   └── evaluation_predictions.csv
│
├── src/
│   ├── common.py
│   ├── train.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

### Module Responsibilities

| File          | Responsibility                                                     |
| ------------- | ------------------------------------------------------------------ |
| `common.py`   | Shared configurations, BERTimbau loading, and embedding extraction |
| `train.py`    | Final classifier training and model persistence                    |
| `evaluate.py` | Cross-validation, performance metrics, and error analysis          |

## Installation

### Requirements

- Python 3.10 or later.
- Git.
- Internet connection to download the pretrained model from Hugging Face.

### 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd transaction-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

The project has two independent workflows: evaluation and training.

### Model Evaluation

Run:

```bash
python src/evaluate.py
```

The evaluation script performs:

1. Dataset loading.
2. Duplicate and conflicting-label auditing.
3. Embedding extraction using BERTimbau.
4. Five-fold Stratified K-Fold cross-validation.
5. Accuracy and Macro F1 calculation.
6. Classification report generation.
7. Confusion matrix generation.
8. Out-of-fold prediction export.

Generated artifacts:

- `models/confusion_matrix.png`
- `models/evaluation_predictions.csv`

### Model Training

Run:

```bash
python src/train.py
```

The training script:

1. Loads the complete dataset.
2. Extracts text embeddings.
3. Trains the Logistic Regression classifier.
4. Saves the trained classifier.
5. Stores experiment metadata.

Generated artifacts:

- `models/transaction_classifier.joblib`
- `models/metadata.json`

**Note:** Evaluation does not train or modify the final classifier. Instead, it trains temporary classifiers for each fold, while the training workflow uses the entire available dataset.

## Model Evaluation Metrics

The following metrics are used:

- **Accuracy:** The proportion of correctly classified transactions.
- **Precision:** The proportion of predicted positive instances that actually belong to a given category.
- **Recall:** The proportion of instances belonging to a category that were correctly identified.
- **Macro F1-score:** The average F1-score across all categories, giving equal weight to each category.

The confusion matrix helps identify categories with greater semantic overlap.

Cross-validation uses five stratified folds, approximately preserving the category distribution in each split.

Evaluation results reflect performance on the available dataset and do not guarantee the same performance on previously unseen real-world banking transactions.

## Roadmap

- [ ] Improve dataset diversity and semantic quality.
- [ ] Investigate recurring classification errors through confusion matrix analysis.
- [ ] Evaluate generalization using unseen real-world transactions.
- [ ] Implement consistent text normalization across training and inference.
- [ ] Evaluate confidence-based prediction rejection.
- [ ] Expose the classifier through an API.
- [ ] Integrate the classifier with Saldo's Go backend.
- [ ] Investigate BERTimbau fine-tuning.

## License

MIT - Licence

---

Developed as part of **Saldo**, a personal finance management project.

Caio Jhonatan Alves Pereira https://github.com/DovCaio
