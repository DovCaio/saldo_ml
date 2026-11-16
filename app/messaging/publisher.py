
import json

import pika


RESULT_QUEUE = "transactions.classification.completed"


def declare_result_queue(
    channel: pika.adapters.blocking_connection.BlockingChannel,
) -> None:
    channel.queue_declare(
        queue=RESULT_QUEUE,
        durable=True,
    )


def publish_result(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    result: dict,
    correlation_id: str | None = None,
) -> None:
    properties = pika.BasicProperties(
        content_type="application/json",
        content_encoding="utf-8",
        delivery_mode=pika.DeliveryMode.Persistent,
        correlation_id=correlation_id,
    )

    channel.basic_publish(
        exchange="",
        routing_key=RESULT_QUEUE,
        body=json.dumps(result, ensure_ascii=False).encode("utf-8"),
        properties=properties,
        mandatory=True,
    )
