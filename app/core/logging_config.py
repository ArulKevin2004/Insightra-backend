import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("mlops_pipeline.log", mode="a", encoding="utf-8")
        ],
    )

    # Prevent uvicorn access logs from being too noisy if needed
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
