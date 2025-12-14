import os
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.routers import submit
from app.database.db import Base, engine

structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
log = structlog.get_logger()

app = FastAPI(title="Agentic Form Handler", version="0.2.0")

allowed_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    os.getenv("FRONTEND_ORIGIN", "http://localhost:8501"),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    log.info("startup_complete")


@app.get("/health")
def healthcheck() -> dict[str, Any]:
    return {"status": "ok"}


app.include_router(submit.router)
