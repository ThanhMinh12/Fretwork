#!/usr/bin/env python3
"""Development entrypoint for the FastAPI + React Fretwork application."""

import os

import uvicorn

from fretwork_api.main import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("FRETWORK_HOST", "127.0.0.1"),
        port=int(os.getenv("FRETWORK_PORT", "8000")),
    )

