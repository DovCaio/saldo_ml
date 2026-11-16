
import numpy as np

from app.ml.preprocessing import normalize_description
from src.common import extract_embeddings, POOLING

from app.ml.model_loader import ModelBundle


CONFIDENCE_THRESHOLD = 0.60


class TransactionClassifier:

    def __init__(self, models: ModelBundle):
        self.tokenizer = models.tokenizer
        self.bert = models.bert
        self.classifier = models.classifier
        self.metadata = models.metadata

    def predict(self, description: str) -> dict:
        if not description or not description.strip():
            raise ValueError("Transaction description cannot be empty.")
        
        description = normalize_description(description)

        embeddings = extract_embeddings(
            [description],
            self.tokenizer,
            self.bert,
            pooling=POOLING,
        )

        probabilities = self.classifier.predict_proba(embeddings)[0]

        predicted_index = np.argmax(probabilities)
        predicted_category = self.classifier.classes_[predicted_index]
        confidence = float(probabilities[predicted_index])

        category = (
            predicted_category
            if confidence >= CONFIDENCE_THRESHOLD
            else "others"
        )

        return {
            "category": str(category),
            "confidence": confidence,
            "predicted_category": str(predicted_category),
        }