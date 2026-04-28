"""
app/main.py
-----------
FastAPI application factory.

The lifespan context manager handles:
  1. Model loading at startup
  2. Graceful shutdown cleanup

This pattern (lifespan over @app.on_event) is the modern FastAPI best practice.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router, set_service
from app.core.config import settings
from app.core.logging import logger
from app.models.model_registry import registry
from app.services.inference import InferenceService, ModelWrapper


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Startup / shutdown lifecycle."""
    logger.info("Starting Neuro-AI Platform {}", settings.MODEL_VERSION)

    try:
        cnn_model = registry.load_cnn()
        wrapper   = ModelWrapper(model=cnn_model)
        service   = InferenceService(
            model_wrapper=wrapper,
            window_size=cnn_model.window_size,
            stride=max(cnn_model.window_size // 2, 1),
        )
        set_service(service)
        logger.info("Model loaded successfully. Inference service ready.")
    except FileNotFoundError:
        logger.warning(
            "No trained model found at {}. "
            "Run `python scripts/train.py` first. "
            "The API will start but /predict_* endpoints will return 503.",
            settings.model_path,
        )

    yield  # ← application runs here

    logger.info("Shutting down Neuro-AI Platform.")


def create_app() -> FastAPI:
    frontend_dir = settings.FRONTEND_DIR
    app = FastAPI(
        title="Neuro-AI Seizure Detection Platform",
        description=(
            "Production-grade EEG-based seizure detection API. "
            "Trained on CHB-MIT dataset with 1D CNN architecture."
        ),
        version=settings.MODEL_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — restrict origins in production via environment variable
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=frontend_dir), name="frontend")

        @app.get("/", include_in_schema=False)
        async def frontend_index():
            return FileResponse(frontend_dir / "index.html")

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.exception("Unhandled exception: {}", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Check logs."},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
