import json
import logging

import pika

from app.messaging.publisher import (
    RESULT_QUEUE,
    publish_result,
)

logger = logging.getLogger(__name__)

REQUEST_QUEUE = "transactions.classification.requested"
DEAD_QUEUE = "transactions.classification.dead"
DEAD_EXCHANGE = "transactions.dlx"
DEAD_ROUTING_KEY = "classification.dead"


def declare_queues(channel):
    channel.exchange_declare(
        exchange=DEAD_EXCHANGE,
        exchange_type="direct",
        durable=True,
    )

    channel.queue_declare(
        queue=DEAD_QUEUE,
        durable=True,
    )

    channel.queue_bind(
        exchange=DEAD_EXCHANGE,
        queue=DEAD_QUEUE,
        routing_key=DEAD_ROUTING_KEY,
    )

    channel.queue_declare(
        queue=REQUEST_QUEUE,
        durable=True,
        arguments={
            "x-dead-letter-exchange": DEAD_EXCHANGE,
            "x-dead-letter-routing-key": DEAD_ROUTING_KEY,
        },
    )

    channel.queue_declare(
        queue=RESULT_QUEUE,
        durable=True,
    )


def process_message(channel, method, properties, body, classifier):

    try:
        message = json.loads(body)

        transaction_id = message["transaction_id"]
        event_id = message["event_id"]
        description = message["description"]
        description_version = message["description_version"]

        if not all([
            transaction_id,
            event_id,
            description,
            description_version is not None,
        ]):
            raise ValueError("Invalid transaction message.")

        prediction = classifier.predict(description)

        result = {
            "transaction_id": transaction_id,
            "event_id": event_id,
            "description_version": description_version,
            "category": prediction["category"],
            "predicted_category": prediction["predicted_category"],
            "confidence": prediction["confidence"],
        }

        publish_result(
            channel=channel,
            result=result,
            correlation_id=properties.correlation_id,
        )

        channel.basic_ack(delivery_tag=method.delivery_tag)

        logger.info(
            "Transaction %s classified successfully.",
            transaction_id,
        )

    except Exception:
        logger.exception("Failed to process classification message.")

        channel.basic_nack(
            delivery_tag=method.delivery_tag,
            requeue=False,
        )


def start_consumer(channel, classifier):

    declare_queues(channel)

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue=REQUEST_QUEUE,
        on_message_callback=lambda ch, method, properties, body: (
            process_message(
                ch,
                method,
                properties,
                body,
                classifier,
            )
        ),
        auto_ack=False,
    )

    logger.info("Waiting for classification messages...")

    channel.start_consuming()
