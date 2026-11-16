
import logging

from app.messaging.connection import create_connection
from app.messaging.consumer import start_consumer

from app.ml.classifier import TransactionClassifier
from app.ml.model_loader import load_models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting transaction classifier worker...")

    connection = None

    try:
        model_bundle = load_models()
        classifier = TransactionClassifier(model_bundle)

        logger.info("ML model loaded successfully.")

        connection = create_connection()
        channel = connection.channel()

        channel.confirm_delivery()

        start_consumer(
            channel=channel,
            classifier=classifier,
        )

    except KeyboardInterrupt:
        logger.info("Worker interrupted.")

    except Exception:
        logger.exception("Worker stopped due to an unexpected error.")
        raise

    finally:
        if connection is not None and connection.is_open:
            connection.close()

        logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
