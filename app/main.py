
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Application initialization
    app.state.model_loaded = False
    app.state.rabbitmq_connected = False
    app.state.worker_ready = False

    yield

    # Application shutdown
    # Resources will be released here later.


app = FastAPI(
    title="Transaction Classifier API",
    description="Financial transaction classification service",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(health.router)
