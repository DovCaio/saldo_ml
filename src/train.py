
import json
import joblib

from common import (
    load_dataset,
    load_bert,
    extract_embeddings,
    MODELS_DIR,
    CLASSIFIER_PATH,
    METADATA_PATH,
    MODEL_NAME,
    MAX_LENGTH,
    POOLING
)

from sklearn.linear_model import LogisticRegression


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("Carregando dataset...")

df, X, y = load_dataset()

print("\nDistribuição das categorias:")
print(y.value_counts())

print("\nTotal de registros:", len(df))
print("Total de categorias:", y.nunique())


# ==========================================================
# 2. LOAD BERTIMBAU
# ==========================================================

tokenizer, bert = load_bert()


# ==========================================================
# 3. EXTRACT EMBEDDINGS
# ==========================================================

print("\nExtraindo embeddings...")

X_embeddings = extract_embeddings(
    X.tolist(),
    tokenizer,
    bert,
    pooling=POOLING
)

y_array = y.to_numpy()


# ==========================================================
# 4. TRAIN FINAL CLASSIFIER
# ==========================================================

print("\nTreinando classificador final...")

classifier = LogisticRegression(
    C=1.0,
    max_iter=2000,
    class_weight="balanced"
)

classifier.fit(
    X_embeddings,
    y_array
)


# ==========================================================
# 5. SAVE MODEL
# ==========================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    classifier,
    CLASSIFIER_PATH
)

metadata = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "pooling": POOLING,
    "classifier": "LogisticRegression",
    "classifier_params": {
        "C": 1.0,
        "max_iter": 2000,
        "class_weight": "balanced"
    },
    "dataset": str(
        "transactions_pt_br_v4.csv"
    ),
    "training_samples": len(df),
    "categories": classifier.classes_.tolist()
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )


print("\n================================")
print("TREINAMENTO CONCLUÍDO")
print("================================")

print("Classificador salvo em:", CLASSIFIER_PATH)
print("Metadados salvos em:", METADATA_PATH)
