
import numpy as np

from src.common import extract_embeddings, POOLING

from app.ml.model_loader import ModelBundle


class TransactionClassifier:

    def __init__(self, models: ModelBundle):
        self.tokenizer = models.tokenizer
        self.bert = models.bert
        self.classifier = models.classifier
        self.metadata = models.metadata

    def predict(self, description: str) -> dict:
        if not description or not description.strip():
            raise ValueError("Transaction description cannot be empty.")

        embeddings = extract_embeddings(
            [description],
            self.tokenizer,
            self.bert,
            pooling=POOLING,
        )

        prediction = self.classifier.predict(embeddings)[0]

        probabilities = self.classifier.predict_proba(embeddings)[0]

        confidence = float(np.max(probabilities))

        return {
            "category": str(prediction),
            "confidence": confidence,
        }
