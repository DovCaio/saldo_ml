
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common import (
    load_dataset,
    load_bert,
    extract_embeddings,
    MODELS_DIR,
    POOLING
)

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("Carregando dataset...")

df, X, y = load_dataset()

y_array = y.to_numpy()

print("\nDistribuição das categorias:")
print(y.value_counts())

print("\nTotal de registros:", len(df))
print("Total de categorias:", y.nunique())


# ==========================================================
# 2. DATA AUDIT
# ==========================================================

def normalize_for_audit(text):

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
    df["description"].apply(normalize_for_audit)
)

duplicates = df[
    df["normalized_description"].duplicated(
        keep=False
    )
]

conflicts = (
    df.groupby("normalized_description")["category"]
    .nunique()
)

conflicts = conflicts[conflicts > 1]


print("\n================================")
print("DATASET AUDIT")
print("================================")

print(
    "\nRegistros envolvidos em duplicatas:",
    len(duplicates)
)

print(
    "\nDescrições com categorias conflitantes:"
)

print(conflicts)


# ==========================================================
# 3. LOAD BERTIMBAU
# ==========================================================

tokenizer, bert = load_bert()


# ==========================================================
# 4. EXTRACT EMBEDDINGS
# ==========================================================

print("\nExtraindo embeddings...")

X_embeddings = extract_embeddings(
    X.tolist(),
    tokenizer,
    bert,
    pooling=POOLING
)


# ==========================================================
# 5. STRATIFIED K-FOLD
# ==========================================================

N_SPLITS = 5

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

results_cv = []

oof_predictions = np.empty(
    len(y_array),
    dtype=object
)


print("\nIniciando validação cruzada...")

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

    oof_predictions[val_idx] = predictions

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

    print(
        f"Accuracy: {accuracy:.4f} | "
        f"F1 Macro: {f1:.4f}"
    )


# ==========================================================
# 6. CROSS VALIDATION RESULTS
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
# 7. CLASSIFICATION REPORT
# ==========================================================

labels = sorted(
    np.unique(y_array)
)

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_array,
        oof_predictions,
        labels=labels,
        zero_division=0
    )
)


# ==========================================================
# 8. CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_array,
    oof_predictions,
    labels=labels
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel("Predito")
plt.ylabel("Real")

plt.title(
    "Matriz de Confusão - BERTimbau + Stratified K-Fold"
)

plt.tight_layout()

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    MODELS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================================
# 9. SAVE OUT-OF-FOLD PREDICTIONS
# ==========================================================

evaluation_results = df[
    ["description", "category"]
].copy()

evaluation_results["predicted_category"] = (
    oof_predictions
)

evaluation_results["correct"] = (
    evaluation_results["category"]
    == evaluation_results["predicted_category"]
)

evaluation_results.to_csv(
    MODELS_DIR / "evaluation_predictions.csv",
    index=False
)


print("\n================================")
print("AVALIAÇÃO CONCLUÍDA")
print("================================")

print(
    "Matriz de confusão salva em:",
    MODELS_DIR / "confusion_matrix.png"
)

print(
    "Previsões OOF salvas em:",
    MODELS_DIR / "evaluation_predictions.csv"
)
