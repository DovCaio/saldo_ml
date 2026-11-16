
from app.ml.model_loader import load_models
from app.ml.classifier import TransactionClassifier


def main():
    models = load_models()

    classifier = TransactionClassifier(models)

    descriptions = [
        "UBER TRIP HELP",
        "MENSALIDADE ACADEMIA",
        "PAGAMENTO CONTA ENERGIA",
        "COMPRA AMAZON",
        "CONSULTA MEDICA",
    ]

    for description in descriptions:
        result = classifier.predict(description)

        print({
            "description": description,
            **result,
        })


if __name__ == "__main__":
    main()
