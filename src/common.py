
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoModel, AutoTokenizer


# ==========================================================
# CONFIGURATIONS
# ==========================================================

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

MAX_LENGTH = 64
BATCH_SIZE = 16
POOLING = "mean"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT_DIR / "data" / "processed" / "transactions_pt_br_v4.csv"
)

MODELS_DIR = ROOT_DIR / "models"

CLASSIFIER_PATH = MODELS_DIR / "transaction_classifier.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


# ==========================================================
# DATASET
# ==========================================================

def load_dataset():

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(
        subset=["description", "category"]
    ).reset_index(drop=True)

    X = df["description"].astype(str)
    y = df["category"].astype(str)

    return df, X, y


# ==========================================================
# BERTIMBAU
# ==========================================================

def load_bert():

    print(f"Dispositivo: {DEVICE}")
    print(f"Carregando {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    bert = AutoModel.from_pretrained(
        MODEL_NAME
    )

    bert.to(DEVICE)
    bert.eval()

    return tokenizer, bert


# ==========================================================
# EMBEDDING EXTRACTION
# ==========================================================

def extract_embeddings(
    texts,
    tokenizer,
    bert,
    pooling=POOLING
):

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

        processed = min(
            i + BATCH_SIZE,
            len(texts)
        )

        print(
            f"Embeddings: {processed}/{len(texts)}",
            end="\r"
        )

    print()

    return np.array(embeddings)
