
from app.ml.model_loader import load_models
from app.ml.classifier import TransactionClassifier
from app.ml.preprocessing import normalize_description


def main():

    print("Loading ML models...")

    model_bundle = load_models()

    classifier = TransactionClassifier(model_bundle)

    print("\nModels loaded successfully!")

    descriptions = [
        "Pagamento de conta de energia elétrica",
        "Compra realizada no supermercado",
        "Transferência enviada pelo Pix - 12 324 123 DAVID NOBELINO - 12.324.123/0001-75 - BS2 PAYMENTS IP S.A. Agência: 1 Conta: 3123-4",
        "Mensalidade da academia",
        "Pagamento de aluguel",
    ]

    print("\nClassification results:\n")

    for description in descriptions:

        normalized = normalize_description(description)

        print(f"Original: {description}")
        print(f"Normalized: {normalized}")

        prediction = classifier.predict(description)

        print(f"Description: {description}")
        print(f"Category: {prediction['category']}")
        print(f"Predicted category: {prediction['predicted_category']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print("-" * 70)


if __name__ == "__main__":
    main()
