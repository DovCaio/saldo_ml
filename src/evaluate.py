
import re
import unicodedata

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModel, AutoTokenizer

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 1. Congigurations
# ==========================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

MAX_LENGTH = 64
BATCH_SIZE = 16
N_SPLITS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo: {DEVICE}")


# ==========================================================
# 2. Loading DATASET
# ==========================================================

df = pd.read_csv(
    "../data/processed/transactions_pt_br_v4.csv"
)

df = df.dropna(
    subset=["description", "category"]
).reset_index(drop=True)

X = df["description"].astype(str)
y = df["category"].astype(str)

print("\nDistribuição das categorias:")
print(y.value_counts())

print("\nTotal de registros:", len(df))
print("Total de categorias:", y.nunique())


# ==========================================================
# 3. Data audit
# ==========================================================

def normalize_description(text):

    text = str(text).lower().strip()

    text = unicodedata.normalize(
        "NFKD",
        text
    )

    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["normalized_description"] = (
    df["description"].apply(normalize_description)
)

duplicates = df[
    df["normalized_description"].duplicated(keep=False)
]

print("\nDuplicatas normalizadas:")
print(len(duplicates))

conflicts = (
    df.groupby("normalized_description")["category"]
    .nunique()
)

conflicts = conflicts[conflicts > 1]

print("\nDescrições com categorias conflitantes:")
print(conflicts)


# ==========================================================
# 4. Loading of BERTIMBAU
# ==========================================================

print("\nCarregando BERTimbau...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert = AutoModel.from_pretrained(
    MODEL_NAME
)

bert.to(DEVICE)
bert.eval()


# ==========================================================
# 5. Embbeding exctration
# ==========================================================

def extract_embeddings(texts, pooling="mean"):

    embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i:i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = bert(**inputs)

            token_embeddings = outputs.last_hidden_state

            if pooling == "cls":

                batch_embeddings = token_embeddings[:, 0, :]

            elif pooling == "mean":

                attention_mask = inputs[
                    "attention_mask"
                ]

                mask = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).float()

                summed = torch.sum(
                    token_embeddings * mask,
                    dim=1
                )

                counts = torch.clamp(
                    mask.sum(dim=1),
                    min=1e-9
                )

                batch_embeddings = summed / counts

            else:

                raise ValueError(
                    f"Pooling inválido: {pooling}"
                )

        embeddings.extend(
            batch_embeddings.cpu().numpy()
        )

        print(
            f"Embeddings: {min(i + BATCH_SIZE, len(texts))}/{len(texts)}",
            end="\r"
        )

    print()

    return np.array(embeddings)


# ==========================================================
# 6. Embedding data set extractions
# ==========================================================

print("\nExtraindo embeddings do dataset...")

X_embeddings = extract_embeddings(
    X.tolist(),
    pooling="mean"
)

y_array = y.to_numpy()


# ==========================================================
# 7. STRATIFIED K-FOLD
# ==========================================================

print("\nIniciando validação cruzada...")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

results_cv = []

all_predictions = []
all_real_labels = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X_embeddings, y_array),
    start=1
):

    print(f"\nTreinando Fold {fold}/{N_SPLITS}...")

    X_fold_train = X_embeddings[train_idx]
    X_fold_val = X_embeddings[val_idx]

    y_fold_train = y_array[train_idx]
    y_fold_val = y_array[val_idx]

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced"
    )

    classifier.fit(
        X_fold_train,
        y_fold_train
    )

    predictions = classifier.predict(
        X_fold_val
    )

    accuracy = accuracy_score(
        y_fold_val,
        predictions
    )

    f1 = f1_score(
        y_fold_val,
        predictions,
        average="macro"
    )

    results_cv.append({
        "fold": fold,
        "accuracy": accuracy,
        "f1_macro": f1
    })

    all_predictions.extend(predictions)
    all_real_labels.extend(y_fold_val)

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"F1 Macro: {f1:.4f}"
    )


# ==========================================================
# 8. Result of cross avaliation
# ==========================================================

results_cv = pd.DataFrame(results_cv)

print("\n================================")
print("RESULTADOS DOS FOLDS")
print("================================")

print(results_cv.to_string(index=False))

print("\nMÉDIA:")
print(
    results_cv[
        ["accuracy", "f1_macro"]
    ].mean()
)

print("\nDESVIO PADRÃO:")
print(
    results_cv[
        ["accuracy", "f1_macro"]
    ].std()
)


# ==========================================================
# 9. CLASSIFICATION REPORT AGREGADO
# ==========================================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        all_real_labels,
        all_predictions,
        labels=np.unique(y_array),
        zero_division=0
    )
)


# ==========================================================
# 10. Confunsion matrix
# ==========================================================

labels = np.unique(y_array)

cm = confusion_matrix(
    all_real_labels,
    all_predictions,
    labels=labels
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - BERTimbau + K-Fold")

plt.tight_layout()
plt.show()


# ==========================================================
# 11. Tests with transactions
# ==========================================================

samples = [

    "UBER *TRIP HELP",
    "99 POP CORRIDA",
    "POSTO SHELL ABASTECIMENTO",

    "PAGAMENTO RESTAURANTE",
    "LANCHONETE DO ZE",
    "DROGARIA SAO PAULO",

    "CONSULTA MEDICA",
    "MENSALIDADE ACADEMIA",
    "AMAZON PRIME VIDEO",

    "MENSALIDADE CURSO INGLES",
    "PAGAMENTO CONTA DE ENERGIA",
    "COMPRA MAGAZINE LUIZA",

]


# ==========================================================
# 12. Final train with the data
# ==========================================================

print("\nTreinando classificador final...")

final_classifier = LogisticRegression(
    C=1.0,
    max_iter=2000,
    class_weight="balanced"
)

final_classifier.fit(
    X_embeddings,
    y_array
)


# ==========================================================
# 13. Predictions
# ==========================================================

sample_embeddings = extract_embeddings(
    samples,
    pooling="mean"
)

predictions = final_classifier.predict(
    sample_embeddings
)

probabilities = final_classifier.predict_proba(
    sample_embeddings
)

print("\n================================")
print("NOVAS TRANSAÇÕES")
print("================================")

for description, category, probability in zip(
    samples,
    predictions,
    probabilities
):

    confidence = np.max(probability) * 100

    print(
        f"{description:40} -> "
        f"{category:20} "
        f"(Confiança: {confidence:.2f}%)"
    )

print("\nAvaliação concluída!")