
import json
import logging

import pika

from app.messaging.status import WORKER_STATUS_QUEUE

logger = logging.getLogger(__name__)


def declare_status_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
) -> None:
    channel.queue_declare(
        queue=WORKER_STATUS_QUEUE,
        durable=True,
    )


def publish_status(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    worker_id: str,
    status: str,
    model_loaded: bool,
) -> None:
    message = {
        "worker_id": worker_id,
        "status": status,
        "model_loaded": model_loaded,
    }

    properties = pika.BasicProperties(
        content_type="application/json",
        content_encoding="utf-8",
        delivery_mode=pika.DeliveryMode.Persistent,
    )

    channel.basic_publish(
        exchange="",
        routing_key=WORKER_STATUS_QUEUE,
        body=json.dumps(message).encode("utf-8"),
        properties=properties,
        mandatory=True,
    )

    logger.info("Worker status published: %s", status)
