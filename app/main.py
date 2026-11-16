import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT_DIR / "models" / "metadata.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        app.state.model_metadata = json.load(file)

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/model/info")
def models_info(request: Request):
    return request.app.state.model_metadata