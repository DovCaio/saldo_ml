import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_metadata = None
    app.state.model_loaded = False
    app.state.rabbitmq_connected = False
    app.state.worker_ready = False
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        app.state.model_metadata = json.load(file)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/model/info")
def models_info(request: Request):
    return request.app.state.model_metadata

@app.get("/health/live")
def health_live():
    return {
        "status": "alive"
    }


@app.get("/health/ready")
def health_ready(request: Request):
    state = request.app.state

    checks = {
        "metadata": state.model_metadata is not None,
        "model": state.model_loaded,
        "rabbitmq": state.rabbitmq_connected,
        "worker": state.worker_ready,
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks,
            },
        )

    return {
        "status": "ready",
        "checks": checks,
    }