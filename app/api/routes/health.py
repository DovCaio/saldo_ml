
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
def health_live():
    return {
        "status": "alive"
    }


@router.get("/ready")
def health_ready(request: Request):
    state = request.app.state

    checks = {
        "model": state.model_loaded,
        "rabbitmq": state.rabbitmq_connected,
        "worker": state.worker_ready
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "checks": checks
            }
        )

    return {
        "status": "ready",
        "checks": checks
    }
