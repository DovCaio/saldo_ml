
import logging

from messaging.connection import create_connection
from messaging.consumer import start_consumer
from messaging.status import WORKER_ID
from messaging.status_publisher import (
    declare_status_queue,
    publish_status,
)

from ml.classifier import TransactionClassifier
from ml.model_loader import load_models


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting transaction classifier worker...")

    connection = None
    channel = None

    try:
        model_bundle = load_models()
        classifier = TransactionClassifier(model_bundle)

        logger.info("ML model loaded successfully.")

        connection = create_connection()
        channel = connection.channel()

        channel.confirm_delivery()

        declare_status_queue(channel)

        publish_status(
            channel=channel,
            worker_id=WORKER_ID,
            status="ready",
            model_loaded=True,
        )

        start_consumer(
            channel=channel,
            classifier=classifier,
        )

    except KeyboardInterrupt:
        logger.info("Worker interrupted.")

    except Exception:
        logger.exception("Worker stopped unexpectedly.")
        raise

    finally:
        if channel is not None and channel.is_open:
            try:
                publish_status(
                    channel=channel,
                    worker_id=WORKER_ID,
                    status="stopped",
                    model_loaded=False,
                )
            except Exception:
                logger.exception("Failed to publish shutdown status.")

        if connection is not None and connection.is_open:
            connection.close()

        logger.info("Worker stopped.")


if __name__ == "__main__":
    main()
