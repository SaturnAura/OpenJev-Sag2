"""HTTP service for SAG2's native ``/v1/decide`` judge."""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from openjev.runtime import load_engine
from openjev.schema import build_question


class DecideRequest(BaseModel):
    model: str = ""
    state: str
    questions: dict


def create_app(engine=None) -> FastAPI:
    app = FastAPI(title="OpenJev-SAG2", version="1.0.0")
    app.state.engine = engine

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "readout": app.state.engine.readout.name,
            "model": app.state.engine.config.model_path,
        }

    @app.post("/v1/decide")
    def decide(request: DecideRequest):
        engine = app.state.engine
        for qid, spec in request.questions.items():
            build_question(qid, spec)
        response = engine.decide(request.state, request.questions)
        payload = response.to_dict()
        payload["model"] = request.model or engine.config.model_path
        return payload

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="9b", help="9b or a local backbone path")
    parser.add_argument("--checkpoint", default=None, help="trained readout.pt path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8300)
    args = parser.parse_args()
    engine = load_engine(
        args.model,
        args.checkpoint,
        device=args.device,
        batch_size=args.batch,
    )
    uvicorn.run(create_app(engine), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

