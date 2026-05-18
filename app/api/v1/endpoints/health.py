from typing import Annotated

from fastapi import APIRouter

from app.schemas.health import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Check the health of the API.
    """
    return HealthResponse(status="ok", version="1.0.0")
