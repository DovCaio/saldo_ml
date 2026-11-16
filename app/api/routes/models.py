from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/models", tags=["Models"])

@router.get("/info")
def models_info(request: Request):
    return request.app.state.model_metadata