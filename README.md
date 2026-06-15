# FedNeo-Guard (MVP)

Neonatal sepsis prediction pipeline with:

- MIMIC-compatible preprocessing on synthetic NICU logs
- Transformer + LSTM hybrid model
- Flower-based federated training with Opacus differential privacy
- SHAP explainability and clinical validation metrics

## Quick Start

1. Create env and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run centralized baseline:

```powershell
python -m src.train.centralized_train
```

3. Run federated simulation (3 hospitals):

```powershell
# Terminal 1
python -m src.fl.server

# Terminal 2/3/4
python -m src.fl.client --hospital_id H1
python -m src.fl.client --hospital_id H2
python -m src.fl.client --hospital_id H3
```

4. Run explainability:

```powershell
python -m src.xai.shap_explainer
```

## Deployment

This repo deploys as two parts:

1. **API backend**: the FastAPI app in `api/app.py`, run in a Python host or container.
2. **Clinical UI**: the Streamlit app in `app.py`, deployable to Streamlit Community Cloud.

### Backend

Build and run the container:

```powershell
docker build -t fedneo-guard .
docker run -p 8000:8000 -e SEPSIS_CHECKPOINT_PATH=/models/federated_global_model_final.pt fedneo-guard
```

Required runtime settings:

| Variable | Purpose |
| --- | --- |
| `SEPSIS_CHECKPOINT_PATH` | Path to a saved `.pt` checkpoint |
| `SEPSIS_SEQ_LEN_STEPS` | Fallback sequence length when the checkpoint does not store one |
| `SEPSIS_PREDICTION_THRESHOLD` | Fallback decision threshold |

If no checkpoint is mounted or `SEPSIS_CHECKPOINT_PATH` is not set to a valid file, the API now generates a temporary fallback checkpoint at startup so the container can still boot for demo and smoke-test use.

The API exposes `GET /health`, `GET /model-info`, `GET /example-payload`, and `POST /predict`.

### Streamlit UI

Run locally with:

```powershell
streamlit run app.py
```

For Streamlit Community Cloud, set the app entrypoint to `app.py`. The UI loads the federated checkpoint if present, otherwise it boots in demo mode with a temporary fallback model.

### Static frontend

The legacy static demo remains in `frontend/` for direct API testing.

Set the Vercel project root to `frontend/`. The page lets you enter the backend URL, load an example request, and send predictions to the API.

### Free FL demo flow

1. Start the Flower server locally with `python -m src.fl.server --bind_address 0.0.0.0:8080`.
2. Start an Ngrok TCP tunnel for port `8080`.
3. Launch each Colab client with `python -m src.fl.client --server_address <ngrok-host:port> --hospital_id H1` and `H2`.
4. Save the final checkpoint to `results/federated_global_model_final.pt` and deploy `app.py` to Streamlit Cloud.

### Notes

- Training, federated learning, and explainability scripts remain local/batch jobs.
- The API expects feature-ready rows with the columns in `src/constants.py`.
- If you want to deploy on a platform other than Docker/Cloud Run/Railway/Fly.io, point it at the same `uvicorn api.app:app` entrypoint.

## Implemented Scope

- Phase 1: vitals/static extraction, HRV (SDNN, RMSSD), FFill + KNN imputation, federated partitioning
- Phase 2: Transformer-LSTM with hour-level self-attention and 6h sepsis risk output
- Phase 3: Flower FedAvg client/server + Opacus DP-SGD + SMPC placeholder interface
- Phase 4: SHAP plots + AUROC/sensitivity/specificity/lead-time calculation

## Recent Hardening

- Weighted BCE for class imbalance (sepsis-positive weighting)
- Sensitivity-targeted threshold tuning (default target: 0.80)
- Precision-Recall curve export from centralized training (`precision_recall_curve.png`)
- DP client metrics now include epsilon per round when `--use_dp` is enabled