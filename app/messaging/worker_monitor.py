
import json
import logging
import threading
import time

import pika

from app.messaging.connection import create_connection
from app.messaging.status import WORKER_STATUS_QUEUE

logger = logging.getLogger(__name__)


class WorkerMonitor:
    def __init__(self, app):
        self.app = app
        self.thread = None
        self.stopping = threading.Event()

    def start(self):
        self.thread = threading.Thread(
            target=self._consume,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stopping.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _consume(self):
        connection = None

        try:
            connection = create_connection()
            channel = connection.channel()

            channel.queue_declare(
                queue=WORKER_STATUS_QUEUE,
                durable=True,
            )

            self.app.state.rabbitmq_connected = True

            def callback(ch, method, properties, body):
                try:
                    message = json.loads(body)

                    if message.get("status") == "ready":
                        self.app.state.model_loaded = (
                            message.get("model_loaded", False)
                        )
                        self.app.state.worker_ready = True

                    elif message.get("status") == "stopped":
                        self.app.state.worker_ready = False
                        self.app.state.model_loaded = False

                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except Exception:
                    logger.exception("Failed to process worker status.")
                    ch.basic_nack(
                        delivery_tag=method.delivery_tag,
                        requeue=False,
                    )

            channel.basic_consume(
                queue=WORKER_STATUS_QUEUE,
                on_message_callback=callback,
                auto_ack=False,
            )

            logger.info("Worker monitor started.")

            while not self.stopping.is_set():
                connection.process_data_events(time_limit=1)

        except Exception:
            logger.exception("Worker monitor connection failed.")
            self.app.state.rabbitmq_connected = False
            self.app.state.worker_ready = False

        finally:
            if connection and connection.is_open:
                connection.close()
