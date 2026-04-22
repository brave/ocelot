"""FastAPI app: mounts v1 router at /v1."""

from fastapi import FastAPI

from .routers import v1_router

app = FastAPI(title="Ocelot data API")
app.include_router(v1_router, prefix="/v1")
