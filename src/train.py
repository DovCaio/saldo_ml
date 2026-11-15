
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModel, AutoTokenizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# 1. Configurações
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
MAX_LENGTH = 64
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Dispositivo: {DEVICE}")


# 2. Carregando o dataset
df = pd.read_csv("../data/processed/transactions_pt_br.csv")



df = df.dropna(subset=["description", "category"])

X = df["description"].astype(str)
y = df["category"].astype(str)

print("\nDistribuição das categorias:")
print(y.value_counts())


# 3. Carregando o BERTimbau
print("\nCarregando o BERTimbau...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bert = AutoModel.from_pretrained(MODEL_NAME)

bert.to(DEVICE)
bert.eval()


# 4. Extração de embeddings utilizando Mean Pooling

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

                attention_mask = inputs["attention_mask"]

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
                raise ValueError("Pooling inválido")

        embeddings.extend(
            batch_embeddings.cpu().numpy()
        )

    return np.array(embeddings)



# 5. Separando treino e teste

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("Treino:", len(X_train))
print("Validação:", len(X_val))
print("Teste:", len(X_test))



from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score

experiments = []

# Extraindo embeddings uma vez para cada pooling
for pooling in ["mean", "cls"]:

    print(f"\nExtraindo embeddings: {pooling}")

    X_train_emb = extract_embeddings(
        X_train.tolist(),
        pooling=pooling
    )

    X_val_emb = extract_embeddings(
        X_val.tolist(),
        pooling=pooling
    )

    for normalize_embeddings in [False, True]:

        if normalize_embeddings:

            train_emb = normalize(X_train_emb, norm="l2")
            val_emb = normalize(X_val_emb, norm="l2")

        else:

            train_emb = X_train_emb
            val_emb = X_val_emb

        for c in [0.01, 0.1, 1.0, 10.0]:

            classifier = LogisticRegression(
                C=c,
                max_iter=2000,
                class_weight="balanced"
            )

            classifier.fit(
                train_emb,
                y_train
            )

            predictions = classifier.predict(val_emb)

            accuracy = accuracy_score(
                y_val,
                predictions
            )

            f1 = f1_score(
                y_val,
                predictions,
                average="macro",
                zero_division=0
            )

            experiments.append({
                "pooling": pooling,
                "normalized": normalize_embeddings,
                "C": c,
                "accuracy": accuracy,
                "f1_macro": f1
            })

            print(
                f"Pooling: {pooling:5} | "
                f"Norm: {normalize_embeddings!s:5} | "
                f"C: {c:5} | "
                f"Accuracy: {accuracy:.4f} | "
                f"F1: {f1:.4f}"
            )



results = pd.DataFrame(experiments)

results = results.sort_values(
    by="f1_macro",
    ascending=False
)

print("\nRanking dos experimentos:")
print(results.to_string(index=False))



from sklearn.metrics import classification_report, confusion_matrix

best = results.iloc[0]

print("\nMelhor configuração:")
print(best)

best_pooling = best["pooling"]
best_normalized = best["normalized"]
best_c = best["C"]

X_train_val = pd.concat([X_train, X_val]).tolist()
y_train_val = pd.concat([y_train, y_val])

train_val_emb = extract_embeddings(
    X_train_val,
    pooling=best_pooling
)

test_emb = extract_embeddings(
    X_test.tolist(),
    pooling=best_pooling
)

if best_normalized:
    train_val_emb = normalize(train_val_emb, norm="l2")
    test_emb = normalize(test_emb, norm="l2")

final_classifier = LogisticRegression(
    C=best_c,
    max_iter=2000,
    class_weight="balanced"
)

final_classifier.fit(
    train_val_emb,
    y_train_val
)

predictions = final_classifier.predict(test_emb)

print("\nResultado final:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=final_classifier.classes_
)

plt.figure(figsize=(12, 9))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=final_classifier.classes_,
    yticklabels=final_classifier.classes_
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("BERTimbau - Melhor configuração")

plt.tight_layout()
plt.show()

