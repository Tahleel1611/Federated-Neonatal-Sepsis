from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.constants import MODEL_FEATURE_COLUMNS
from src.deploy.service import ModelBundle, DeploymentError, load_model_bundle, predict_from_rows


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1)
    seq_len_steps: int | None = Field(default=None, ge=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    probability: float
    threshold: float
    label: int
    risk_level: str
    seq_len_steps: int
    rows_used: int
    feature_columns: list[str]
    checkpoint_path: str


def create_app() -> FastAPI:
    bundle = load_model_bundle()

    app = FastAPI(title="FedNeo-Guard API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.model_bundle = bundle

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "FedNeo-Guard API",
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        current_bundle: ModelBundle = app.state.model_bundle
        return {
            "status": "ok",
            "model_loaded": True,
            "checkpoint_path": str(current_bundle.checkpoint_path),
            "seq_len_steps": current_bundle.seq_len_steps,
            "threshold": current_bundle.threshold,
            "feature_count": len(MODEL_FEATURE_COLUMNS),
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        current_bundle: ModelBundle = app.state.model_bundle
        return {
            "checkpoint_path": str(current_bundle.checkpoint_path),
            "seq_len_steps": current_bundle.seq_len_steps,
            "threshold": current_bundle.threshold,
        }

    @app.get("/example-payload")
    def example_payload() -> dict[str, Any]:
        current_bundle: ModelBundle = app.state.model_bundle
        row = {column: 0.0 for column in MODEL_FEATURE_COLUMNS}
        row["HR"] = 140.0
        row["SpO2"] = 97.0
        row["RR"] = 42.0
        row["Temp"] = 36.8
        row["Birth_Weight"] = 2.8
        row["Gestational_Age"] = 38.0
        return {
            "seq_len_steps": current_bundle.seq_len_steps,
            "rows": [dict(row) for _ in range(current_bundle.seq_len_steps)],
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        current_bundle: ModelBundle = app.state.model_bundle
        try:
            result = predict_from_rows(
                current_bundle,
                request.rows,
                seq_len_steps=request.seq_len_steps,
                threshold=request.threshold,
            )
        except DeploymentError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(**result)

    return app


app = create_app()
