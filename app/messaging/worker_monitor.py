
import json
import logging
import threading
import time

from messaging.connection import create_connection
from messaging.status import WORKER_STATUS_QUEUE, WORKER_TIMEOUT

logger = logging.getLogger(__name__)


class WorkerMonitor:
    def __init__(self, app):
        self.app = app
        self.thread = None
        self.stopping = threading.Event()

        self.lock = threading.Lock()
        self.last_heartbeat = None
        self.model_loaded = False
        self.worker_id = None

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="worker-monitor",
        )
        self.thread.start()

    def stop(self):
        self.stopping.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _update_state(self):
        with self.lock:
            heartbeat_age = (
                time.time() - self.last_heartbeat
                if self.last_heartbeat is not None
                else None
            )

            worker_ready = (
                heartbeat_age is not None
                and heartbeat_age <= WORKER_TIMEOUT
                and self.model_loaded
            )

            self.app.state.worker_ready = worker_ready
            self.app.state.model_loaded = (
                worker_ready and self.model_loaded
            )

    def _handle_message(self, message):
        status = message.get("status")
        worker_id = message.get("worker_id")

        if not worker_id:
            raise ValueError("Missing worker_id.")

        with self.lock:
            self.worker_id = worker_id

            if status == "ready":
                self.model_loaded = message.get("model_loaded") is True
                self.last_heartbeat = time.time()

            elif status == "heartbeat":
                self.model_loaded = message.get("model_loaded") is True
                self.last_heartbeat = time.time()

            elif status == "stopped":
                self.model_loaded = False
                self.last_heartbeat = None

            else:
                raise ValueError(f"Unknown worker status: {status}")

        self._update_state()

        logger.info(
            "Worker %s status updated: %s",
            worker_id,
            status,
        )

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
                    self._handle_message(message)

                    ch.basic_ack(
                        delivery_tag=method.delivery_tag,
                    )

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

            logger.info("Worker monitor connected to RabbitMQ.")

            while not self.stopping.is_set():
                connection.process_data_events(time_limit=1)
                self._update_state()

        finally:
            if connection and connection.is_open:
                connection.close()

    def _run(self):
        while not self.stopping.is_set():
            try:
                self._consume()

            except Exception:
                logger.exception("Worker monitor connection failed.")

                self.app.state.rabbitmq_connected = False
                self.app.state.worker_ready = False
                self.app.state.model_loaded = False

                with self.lock:
                    self.last_heartbeat = None
                    self.model_loaded = False

            if not self.stopping.is_set():
                self.stopping.wait(5)
