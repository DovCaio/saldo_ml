from contextlib import asynccontextmanager
import json
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import health
from app.api.routes import models
from app.messaging.worker_monitor import WorkerMonitor

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application initialization
    app.state.model_loaded = False
    app.state.rabbitmq_connected = False
    app.state.worker_ready = False
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        app.state.model_metadata = json.load(file)

    monitor = WorkerMonitor(app)
    app.state.worker_monitor = monitor
    monitor.start()
    yield

    monitor.stop()

    # Application shutdown
    # Resources will be released here later.


app = FastAPI(
    title="Transaction Classifier API",
    description="Financial transaction classification service",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router)
app.include_router(models.router)
