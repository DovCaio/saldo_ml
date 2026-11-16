
import json
import joblib

from dataclasses import dataclass
from pathlib import Path


from src.common import (
    MODEL_NAME,
    DEVICE,
    load_bert,
    POOLING,
    MAX_LENGTH,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT_DIR / "models"

CLASSIFIER_PATH = MODELS_DIR / "transaction_classifier.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


@dataclass
class ModelBundle:
    tokenizer: object
    bert: object
    classifier: object
    metadata: dict


def load_models() -> ModelBundle:
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Classifier not found: {CLASSIFIER_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    tokenizer, bert = load_bert()

    classifier = joblib.load(CLASSIFIER_PATH)

    print(f"BERT model loaded: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Pooling: {POOLING}")
    print("Logistic Regression loaded successfully.")

    return ModelBundle(
        tokenizer=tokenizer,
        bert=bert,
        classifier=classifier,
        metadata=metadata,
    )
