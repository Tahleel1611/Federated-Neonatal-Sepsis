from __future__ import annotations

import streamlit as st

from src.constants import MODEL_FEATURE_COLUMNS
from src.deploy.service import load_model_bundle, predict_from_rows


st.set_page_config(page_title="Neonatal Sepsis FL Demo", page_icon="🩺", layout="centered")


@st.cache_resource
def get_bundle():
    return load_model_bundle()


def build_row() -> dict[str, float]:
    return {
        "HR": float(st.session_state.hr),
        "SpO2": float(st.session_state.spo2),
        "RR": float(st.session_state.rr),
        "Temp": float(st.session_state.temp),
        "WBC": float(st.session_state.wbc),
        "CRP": float(st.session_state.crp),
        "Platelets": float(st.session_state.platelets),
        "Lactate": float(st.session_state.lactate),
        "Birth_Weight": float(st.session_state.birth_weight),
        "Gestational_Age": float(st.session_state.gestational_age),
        "HRV_SDNN": float(st.session_state.hrv_sdnn),
        "HRV_RMSSD": float(st.session_state.hrv_rmssd),
        "WBC_measured_mask": 1.0,
        "CRP_measured_mask": 1.0,
        "Platelets_measured_mask": 1.0,
        "Lactate_measured_mask": 1.0,
        "WBC_hours_since_measured": 0.0,
        "CRP_hours_since_measured": 0.0,
        "Platelets_hours_since_measured": 0.0,
        "Lactate_hours_since_measured": 0.0,
    }


st.title("Neonatal Sepsis Early Warning System")
st.caption("Federated Transformer-LSTM demo for Streamlit Community Cloud.")

bundle = get_bundle()

with st.sidebar:
    st.header("Clinical inputs")
    st.slider("Heart rate (bpm)", 60, 220, 140, key="hr")
    st.slider("SpO2 (%)", 50, 100, 97, key="spo2")
    st.slider("Respiration rate", 10, 100, 42, key="rr")
    st.slider("Temperature (°C)", 34.0, 40.5, 36.8, 0.1, key="temp")
    st.slider("WBC", 0.0, 40.0, 8.0, 0.1, key="wbc")
    st.slider("CRP", 0.0, 200.0, 5.0, 0.1, key="crp")
    st.slider("Platelets", 0.0, 500.0, 220.0, 1.0, key="platelets")
    st.slider("Lactate", 0.0, 20.0, 1.5, 0.1, key="lactate")
    st.slider("Birth weight (kg)", 0.5, 6.0, 2.8, 0.1, key="birth_weight")
    st.slider("Gestational age (weeks)", 20.0, 45.0, 38.0, 0.5, key="gestational_age")
    st.slider("HRV_SDNN", 0.0, 200.0, 35.0, 1.0, key="hrv_sdnn")
    st.slider("HRV_RMSSD", 0.0, 200.0, 28.0, 1.0, key="hrv_rmssd")
    threshold = st.slider("Decision threshold", 0.0, 1.0, float(bundle.threshold), 0.01)

row = build_row()
rows = [dict(row) for _ in range(int(bundle.seq_len_steps))]

st.subheader("Model status")
st.write(
    {
        "checkpoint": str(bundle.checkpoint_path),
        "seq_len_steps": bundle.seq_len_steps,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "demo_mode": bundle.is_dummy,
    }
)

if st.button("Predict sepsis risk"):
    result = predict_from_rows(bundle, rows, threshold=threshold)
    st.metric("Risk probability", f"{result['probability']:.1%}")
    st.write(
        {
            "risk_level": result["risk_level"],
            "threshold": result["threshold"],
            "rows_used": result["rows_used"],
        }
    )
