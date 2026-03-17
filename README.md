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