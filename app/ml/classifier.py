
# app/ml/classifier.py

import numpy as np

from app.ml.preprocessing import normalize_description
from app.ml.model_loader import ModelBundle
from src.common import extract_embeddings, POOLING



CONFIDENCE_THRESHOLD = 0.60


class TransactionClassifier:

    def __init__(self, model: ModelBundle):
        self.model = model

    def predict(self, description: str) -> dict:

        if not description or not description.strip():
            raise ValueError("Transaction description cannot be empty.")

        normalized = normalize_description(description)

        embeddings = extract_embeddings(
            [normalized],
            self.model.tokenizer,
            self.model.bert,
            pooling=POOLING,
            show_progress=False,
        )

        probabilities = self.model.classifier.predict_proba(embeddings)[0]

        predicted_index = np.argmax(probabilities)

        predicted_category = self.model.classifier.classes_[predicted_index]

        confidence = float(probabilities[predicted_index])

        category = (
            predicted_category
            if confidence >= CONFIDENCE_THRESHOLD
            else "others"
        )

        return {
            "category": str(category),
            "predicted_category": str(predicted_category),
            "confidence": confidence,
        }
