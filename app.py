import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pedestrian–Vehicle Risk Prediction",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(
    r"D:\ML PROJECTS\Deep Learning-Based Pedestrian-Vehicle Interaction Risk Prediction for Proactive Road Safety"
)

MODELS_DIR = PROJECT_DIR / "models"

MODEL_CANDIDATES = [
    PROJECT_DIR / "jaad_lstm_risk_model.keras",
    MODELS_DIR / "jaad_lstm_risk_model.keras",
    PROJECT_DIR / "lstm_model.keras",
    MODELS_DIR / "lstm_model.keras",
]

SCALER_CANDIDATES = [
    MODELS_DIR / "jaad_feature_scaler.pkl",
    PROJECT_DIR / "jaad_feature_scaler.pkl",
]

CONFIG_CANDIDATES = [
    MODELS_DIR / "feature_config.json",
    PROJECT_DIR / "feature_config.json",
]

THRESHOLD_CANDIDATES = [
    MODELS_DIR / "risk_threshold.json",
    PROJECT_DIR / "risk_threshold.json",
]


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

FINAL_THRESHOLD = 0.40

EXPECTED_TIMESTEPS = 15
EXPECTED_FEATURES = 15


# ============================================================
# FINAL MODEL PERFORMANCE
# ============================================================

FINAL_MODEL_RESULTS = pd.DataFrame(
    [
        {
            "Model": "LSTM",
            "Threshold": 0.40,
            "Accuracy": 0.9002,
            "Precision": 0.7036,
            "Recall": 0.9280,
            "F1 Score": 0.8004,
            "ROC-AUC": 0.9602,
            "PR-AUC": 0.8267,
        },
        {
            "Model": "Logistic Regression",
            "Threshold": 0.80,
            "Accuracy": 0.8997,
            "Precision": 0.7069,
            "Recall": 0.9138,
            "F1 Score": 0.7972,
            "ROC-AUC": 0.9579,
            "PR-AUC": 0.8060,
        },
        {
            "Model": "Random Forest",
            "Threshold": 0.55,
            "Accuracy": 0.7923,
            "Precision": 0.9229,
            "Recall": 0.0400,
            "F1 Score": 0.0767,
            "ROC-AUC": 0.9441,
            "PR-AUC": 0.7420,
        },
        {
            "Model": "XGBoost",
            "Threshold": 0.35,
            "Accuracy": 0.8083,
            "Precision": 0.7811,
            "Recall": 0.1537,
            "F1 Score": 0.2569,
            "ROC-AUC": 0.9471,
            "PR-AUC": 0.7549,
        },
    ]
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(30, 64, 175, 0.15),
                transparent 35%
            ),
            #080d18;
        color: #f8fafc;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #94a3b8;
        margin-top: 5px;
    }

    .small-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 8px;
    }

    .section-title {
        font-size: 27px;
        font-weight: 750;
        color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .risk-high {
        background: linear-gradient(
            90deg,
            #ff4141,
            #c90000
        );
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .risk-low {
        background: linear-gradient(
            90deg,
            #16a34a,
            #15803d
        );
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .risk-info {
        background: #102a22;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 18px;
        color: #d1fae5;
        font-size: 16px;
        margin-top: 15px;
    }

    .risk-warning {
        background: #3a2811;
        border-left: 5px solid #f59e0b;
        border-radius: 12px;
        padding: 18px;
        color: #fde68a;
        font-size: 16px;
        margin-top: 15px;
    }

    .risk-danger {
        background: #3b1116;
        border-left: 5px solid #ef4444;
        border-radius: 12px;
        padding: 18px;
        color: #fecaca;
        font-size: 16px;
        margin-top: 15px;
    }

    .info-card {
        background: #101827;
        border: 1px solid #1e293b;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #243044;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 14px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 750;
        margin-top: 5px;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #273449;
        padding: 15px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
        background: #101827;
        border-radius: 10px;
    }

    .stTabs [aria-selected="true"] {
        background: #1d4ed8 !important;
        color: white !important;
    }

    .demo-warning {
        background: #263218;
        border-left: 5px solid #eab308;
        padding: 18px;
        border-radius: 12px;
        color: #fef08a;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def find_existing_file(candidates):
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def recursive_find(obj, keys):
    """
    Recursively search dictionaries/lists for likely feature-name keys.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in keys:
                if isinstance(value, list):
                    return value

            result = recursive_find(value, keys)

            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = recursive_find(item, keys)

            if result is not None:
                return result

    return None


# ============================================================
# LOAD FEATURE CONFIG
# ============================================================

def load_feature_names():

    config_path = find_existing_file(CONFIG_CANDIDATES)

    if config_path is None:
        return [f"Feature_{i+1}" for i in range(15)]

    try:

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        possible_keys = {
            "features",
            "feature_names",
            "model_features",
            "modelfeatures",
            "columns",
            "feature_columns",
            "input_features"
        }

        features = recursive_find(
            config,
            possible_keys
        )

        if features is not None:

            cleaned = [
                str(x)
                for x in features
                if x is not None
            ]

            if len(cleaned) == 15:
                return cleaned

    except Exception:
        pass

    return [f"Feature_{i+1}" for i in range(15)]


FEATURE_NAMES = load_feature_names()


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    path = find_existing_file(
        THRESHOLD_CANDIDATES
    )

    if path is None:
        return FINAL_THRESHOLD

    try:

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        possible_keys = [
            "threshold",
            "risk_threshold",
            "decision_threshold",
            "selected_threshold"
        ]

        for key in possible_keys:

            if key in config:

                value = float(config[key])

                if 0 < value < 1:
                    return value

    except Exception:
        pass

    return FINAL_THRESHOLD


MODEL_THRESHOLD = load_threshold()


# ============================================================
# ROBUST SCALER LOADER
# ============================================================

def load_scaler():

    scaler_path = find_existing_file(
        SCALER_CANDIDATES
    )

    if scaler_path is None:

        raise FileNotFoundError(
            "jaad_feature_scaler.pkl was not found.\n\n"
            "Expected one of:\n"
            + "\n".join(
                str(x)
                for x in SCALER_CANDIDATES
            )
        )

    errors = []

    # Try joblib first
    try:

        import joblib

        scaler = joblib.load(
            scaler_path
        )

        return scaler, scaler_path

    except Exception as e:

        errors.append(
            f"joblib: {e}"
        )

    # Try pickle
    try:

        with open(
            scaler_path,
            "rb"
        ) as f:

            scaler = pickle.load(f)

        return scaler, scaler_path

    except Exception as e:

        errors.append(
            f"pickle: {e}"
        )

    raise RuntimeError(
        "The scaler file exists but could not be loaded.\n\n"
        f"File:\n{scaler_path}\n\n"
        "Errors:\n"
        + "\n".join(errors)
    )


# ============================================================
# ROBUST MODEL LOADER
# ============================================================

@st.cache_resource
def load_model_and_scaler():

    model_path = find_existing_file(
        MODEL_CANDIDATES
    )

    if model_path is None:

        raise FileNotFoundError(
            "LSTM model not found.\n\n"
            "Expected one of:\n"
            + "\n".join(
                str(x)
                for x in MODEL_CANDIDATES
            )
        )

    try:

        from tensorflow.keras.models import load_model

        model = load_model(
            model_path,
            compile=False
        )

    except Exception as e:

        raise RuntimeError(
            f"Could not load LSTM model:\n{e}"
        )

    scaler, scaler_path = load_scaler()

    return (
        model,
        scaler,
        model_path,
        scaler_path
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

try:

    model, scaler, MODEL_PATH, SCALER_PATH = (
        load_model_and_scaler()
    )

    MODEL_LOAD_SUCCESS = True
    MODEL_ERROR = None

except Exception as e:

    MODEL_LOAD_SUCCESS = False
    MODEL_ERROR = str(e)

    model = None
    scaler = None
    MODEL_PATH = None
    SCALER_PATH = None


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "current_sequence": None,
    "current_source": None,
    "current_scenario": None,
    "risk_probability": None,
    "predicted_class": None,
    "analysis_complete": False,
    "session_history": [],
    "manual_sequence_df": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DEMO SCENARIOS
# ============================================================

DEMO_SCENARIOS = {

    "High-risk crossing": {
        "target": "high",
        "description": (
            "Synthetic demonstration representing a potentially "
            "dangerous pedestrian–vehicle interaction."
        )
    },

    "Moderate-risk interaction": {
        "target": "medium",
        "description": (
            "Synthetic demonstration representing an interaction "
            "near the decision boundary."
        )
    },

    "Low-risk interaction": {
        "target": "low",
        "description": (
            "Synthetic demonstration representing a comparatively "
            "safe interaction."
        )
    }
}


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_sequence(sequence):

    if model is None or scaler is None:

        raise RuntimeError(
            "Model or scaler is not loaded."
        )

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (
        EXPECTED_TIMESTEPS,
        EXPECTED_FEATURES
    ):

        raise ValueError(
            f"Expected sequence shape "
            f"(15, 15), got {sequence.shape}"
        )

    scaled = scaler.transform(
        sequence
    )

    scaled = scaled.reshape(
        1,
        EXPECTED_TIMESTEPS,
        EXPECTED_FEATURES
    )

    prediction = model.predict(
        scaled,
        verbose=0
    )

    probability = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    probability = float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )

    return probability


# ============================================================
# MODEL-AWARE DEMO GENERATOR
# ============================================================

def generate_demo_sequence(
    target,
    attempts=100,
    batch_size=128
):

    if model is None or scaler is None:
        raise RuntimeError(
            "Model/scaler unavailable."
        )

    best_sequence = None
    best_probability = None
    best_distance = float("inf")

    if target == "high":

        minimum_probability = 0.75

    elif target == "low":

        maximum_probability = 0.15

    else:

        minimum_probability = 0.35
        maximum_probability = 0.65

    for _ in range(attempts):

        scaled_batch = np.random.normal(
            0,
            1,
            size=(
                batch_size,
                EXPECTED_TIMESTEPS,
                EXPECTED_FEATURES
            )
        ).astype(np.float32)

        scaled_batch = np.clip(
            scaled_batch,
            -2.5,
            2.5
        )

        # Temporal smoothing
        for t in range(1, EXPECTED_TIMESTEPS):

            scaled_batch[:, t, :] = (
                0.75 * scaled_batch[:, t - 1, :]
                +
                0.25 * scaled_batch[:, t, :]
            )

        flat_scaled = scaled_batch.reshape(
            -1,
            EXPECTED_FEATURES
        )

        raw_batch = scaler.inverse_transform(
            flat_scaled
        ).reshape(
            batch_size,
            EXPECTED_TIMESTEPS,
            EXPECTED_FEATURES
        )

        predictions = model.predict(
            scaled_batch,
            verbose=0
        ).reshape(-1)

        if target == "high":

            valid = np.where(
                predictions >= minimum_probability
            )[0]

            if len(valid) > 0:

                idx = valid[
                    np.argmax(
                        predictions[valid]
                    )
                ]

                return (
                    raw_batch[idx],
                    float(predictions[idx])
                )

            distances = np.abs(
                predictions - 0.85
            )

        elif target == "low":

            valid = np.where(
                predictions <= maximum_probability
            )[0]

            if len(valid) > 0:

                idx = valid[
                    np.argmin(
                        predictions[valid]
                    )
                ]

                return (
                    raw_batch[idx],
                    float(predictions[idx])
                )

            distances = np.abs(
                predictions - 0.05
            )

        else:

            valid = np.where(
                (
                    predictions >= minimum_probability
                )
                &
                (
                    predictions <= maximum_probability
                )
            )[0]

            if len(valid) > 0:

                idx = valid[
                    np.argmin(
                        np.abs(
                            predictions[valid] - 0.50
                        )
                    )
                ]

                return (
                    raw_batch[idx],
                    float(predictions[idx])
                )

            distances = np.abs(
                predictions - 0.50
            )

        idx = np.argmin(
            distances
        )

        distance = float(
            distances[idx]
        )

        if distance < best_distance:

            best_distance = distance
            best_sequence = raw_batch[idx].copy()
            best_probability = float(
                predictions[idx]
            )

    return (
        best_sequence,
        best_probability
    )


# ============================================================
# MANUAL DATA PREPARATION
# ============================================================

def prepare_manual_sequence(df):

    df = df.copy()

    if df.shape != (15, 15):

        raise ValueError(
            f"Expected 15 × 15 table. "
            f"Received {df.shape}."
        )

    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    if df.isna().any().any():

        raise ValueError(
            "All 225 cells must contain numeric values."
        )

    return df.to_numpy(
        dtype=np.float32
    )


# ============================================================
# GAUGE
# ============================================================

def create_gauge(
    probability,
    threshold
):

    probability_percent = probability * 100
    threshold_percent = threshold * 100

    bar_color = (
        "#ef4444"
        if probability >= threshold
        else "#22c55e"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percent,
            number={
                "suffix": "%",
                "font": {
                    "size": 46,
                    "color": "#f8fafc"
                }
            },
            title={
                "text": "Predicted High-Risk Probability",
                "font": {
                    "color": "#cbd5e1",
                    "size": 15
                }
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickfont": {
                        "color": "#cbd5e1"
                    }
                },

                "bar": {
                    "color": bar_color,
                    "thickness": 0.35
                },

                "steps": [
                    {
                        "range": [
                            0,
                            threshold_percent
                        ],
                        "color": "#163b2a"
                    },
                    {
                        "range": [
                            threshold_percent,
                            100
                        ],
                        "color": "#431a20"
                    }
                ],

                "threshold": {
                    "line": {
                        "color": "#ffffff",
                        "width": 5
                    },
                    "thickness": 0.8,
                    "value": threshold_percent
                }
            }
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(
            l=25,
            r=25,
            t=60,
            b=15
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">
        🚨 Pedestrian–Vehicle Interaction Risk Prediction
    </div>

    <div class="subtitle">
        Deep Learning-Based Pedestrian–Vehicle Interaction Risk Prediction
    </div>

    <div class="small-subtitle">
        LSTM-powered proactive road safety analytics
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL STATUS
# ============================================================

if not MODEL_LOAD_SUCCESS:

    st.error(
        "Model initialization failed."
    )

    st.code(
        MODEL_ERROR
    )

    st.stop()


with st.sidebar:

    st.markdown(
        "## ⚙️ System Information"
    )

    st.success(
        "LSTM model loaded"
    )

    st.caption(
        f"Model:\n{MODEL_PATH}"
    )

    st.caption(
        f"Scaler:\n{SCALER_PATH}"
    )

    st.metric(
        "Decision Threshold",
        f"{MODEL_THRESHOLD:.2f}"
    )

    st.metric(
        "Input Timesteps",
        EXPECTED_TIMESTEPS
    )

    st.metric(
        "Input Features",
        EXPECTED_FEATURES
    )

    st.divider()

    st.markdown(
        """
        **Final LSTM test performance**

        Accuracy: **90.02%**

        Precision: **70.36%**

        Recall: **92.80%**

        F1 Score: **80.04%**

        ROC-AUC: **96.02%**

        PR-AUC: **82.67%**
        """
    )


# ============================================================
# MAIN TABS
# ============================================================

tabs = st.tabs(
    [
        "🚨 Risk Prediction",
        "🔬 Interaction Analysis",
        "📊 Final Model Performance",
        "📈 Session Analytics",
        "📋 Incident Reports",
    ]
)


# ============================================================
# TAB 1 — RISK PREDICTION
# ============================================================

with tabs[0]:

    left, right = st.columns(
        [1.05, 0.95],
        gap="large"
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="section-title">Prediction Input</div>',
            unsafe_allow_html=True
        )

        input_source = st.radio(
            "Input source",
            [
                "Demo sequence",
                "CSV sequence",
                "Manual sequence"
            ],
            horizontal=True,
            key="input_source"
        )

        # ====================================================
        # DEMO
        # ====================================================

        if input_source == "Demo sequence":

            st.markdown(
                "### Choose demo scenario"
            )

            scenario_name = st.selectbox(
                "Demo scenario",
                list(
                    DEMO_SCENARIOS.keys()
                ),
                label_visibility="collapsed"
            )

            scenario = DEMO_SCENARIOS[
                scenario_name
            ]

            st.markdown(
                f"""
                <div class="demo-warning">
                    <b>{scenario_name}</b><br><br>
                    {scenario["description"]}<br><br>
                    Demo sequences are synthetic UI demonstrations.
                    They must not be treated as scientific results.
                </div>
                """,
                unsafe_allow_html=True
            )

            if st.button(
                "Generate Selected Demo Scenario",
                use_container_width=True
            ):

                with st.spinner(
                    "Generating model-aware demo sequence..."
                ):

                    try:

                        sequence, demo_probability = (
                            generate_demo_sequence(
                                scenario["target"]
                            )
                        )

                        st.session_state[
                            "current_sequence"
                        ] = sequence

                        st.session_state[
                            "current_source"
                        ] = "Demo sequence"

                        st.session_state[
                            "current_scenario"
                        ] = scenario_name

                        st.session_state[
                            "analysis_complete"
                        ] = False

                        st.success(
                            f"Loaded demo scenario: "
                            f"{scenario_name}"
                        )

                        st.caption(
                            f"Generated sequence shape: "
                            f"{sequence.shape}"
                        )

                        st.caption(
                            f"Model-aware demonstration "
                            f"probability: "
                            f"{demo_probability:.2%}"
                        )

                    except Exception as e:

                        st.error(
                            f"Could not generate demo: {e}"
                        )

        # ====================================================
        # CSV
        # ====================================================

        elif input_source == "CSV sequence":

            st.markdown(
                "### Upload Sequence CSV"
            )

            uploaded_file = st.file_uploader(
                "Upload a 15 × 15 CSV sequence",
                type=["csv"],
                key="sequence_csv"
            )

            if uploaded_file is not None:

                try:

                    csv_df = pd.read_csv(
                        uploaded_file
                    )

                    if csv_df.shape != (
                        EXPECTED_TIMESTEPS,
                        EXPECTED_FEATURES
                    ):

                        st.error(
                            "CSV must contain exactly "
                            "15 rows × 15 columns. "
                            f"Detected: {csv_df.shape}"
                        )

                    else:

                        numeric_df = csv_df.apply(
                            pd.to_numeric,
                            errors="coerce"
                        )

                        if numeric_df.isna().any().any():

                            st.error(
                                "CSV contains missing or "
                                "non-numeric values."
                            )

                        else:

                            sequence = numeric_df.to_numpy(
                                dtype=np.float32
                            )

                            st.session_state[
                                "current_sequence"
                            ] = sequence

                            st.session_state[
                                "current_source"
                            ] = "CSV sequence"

                            st.session_state[
                                "current_scenario"
                            ] = uploaded_file.name

                            st.session_state[
                                "analysis_complete"
                            ] = False

                            st.success(
                                "CSV sequence loaded successfully."
                            )

                            st.dataframe(
                                numeric_df,
                                use_container_width=True,
                                height=400
                            )

                except Exception as e:

                    st.error(
                        f"Could not read CSV: {e}"
                    )

        # ====================================================
        # MANUAL
        # ====================================================

        elif input_source == "Manual sequence":

            st.markdown(
                "### Enter Manual Sequence"
            )

            st.info(
                "Enter exactly 15 timesteps × 15 features. "
                "Use the horizontal scrollbar if necessary."
            )

            if (
                st.session_state[
                    "manual_sequence_df"
                ] is None
            ):

                st.session_state[
                    "manual_sequence_df"
                ] = pd.DataFrame(
                    np.zeros(
                        (
                            EXPECTED_TIMESTEPS,
                            EXPECTED_FEATURES
                        )
                    ),
                    columns=FEATURE_NAMES,
                    index=[
                        f"T{i+1}"
                        for i in range(
                            EXPECTED_TIMESTEPS
                        )
                    ]
                )

            manual_df = st.data_editor(
                st.session_state[
                    "manual_sequence_df"
                ],
                num_rows="fixed",
                use_container_width=True,
                hide_index=False,
                height=650,
                key="manual_editor"
            )

            st.session_state[
                "manual_sequence_df"
            ] = manual_df

            if st.button(
                "Use Manual Sequence",
                use_container_width=True
            ):

                try:

                    sequence = prepare_manual_sequence(
                        manual_df
                    )

                    st.session_state[
                        "current_sequence"
                    ] = sequence

                    st.session_state[
                        "current_source"
                    ] = "Manual sequence"

                    st.session_state[
                        "current_scenario"
                    ] = "Manual sequence"

                    st.session_state[
                        "analysis_complete"
                    ] = False

                    st.success(
                        "Manual sequence loaded successfully."
                    )

                    st.caption(
                        f"Sequence shape: {sequence.shape}"
                    )

                except Exception as e:

                    st.error(
                        f"Invalid manual sequence: {e}"
                    )

        # ====================================================
        # CURRENT SEQUENCE
        # ====================================================

        if (
            st.session_state[
                "current_sequence"
            ] is not None
        ):

            st.markdown(
                "### Current Sequence"
            )

            current_sequence = (
                st.session_state[
                    "current_sequence"
                ]
            )

            sequence_df = pd.DataFrame(
                current_sequence,
                columns=FEATURE_NAMES,
                index=[
                    f"T{i+1}"
                    for i in range(15)
                ]
            )

            st.dataframe(
                sequence_df.round(4),
                use_container_width=True,
                height=350
            )

            st.caption(
                f"Prepared sequence shape: "
                f"{current_sequence.shape}"
            )

            if st.button(
                "🚨 ANALYZE INTERACTION",
                type="primary",
                use_container_width=True
            ):

                try:

                    probability = predict_sequence(
                        current_sequence
                    )

                    predicted_class = int(
                        probability >= MODEL_THRESHOLD
                    )

                    st.session_state[
                        "risk_probability"
                    ] = probability

                    st.session_state[
                        "predicted_class"
                    ] = predicted_class

                    st.session_state[
                        "analysis_complete"
                    ] = True

                    source = st.session_state.get(
                        "current_source",
                        "Unknown"
                    )

                    scenario = st.session_state.get(
                        "current_scenario",
                        "Unknown"
                    )

                    record = {
                        "Timestamp": pd.Timestamp.now(),
                        "Source": source,
                        "Scenario": scenario,
                        "Probability": probability,
                        "Threshold": MODEL_THRESHOLD,
                        "Prediction": (
                            "High Risk"
                            if predicted_class == 1
                            else "Low Risk"
                        ),
                        "Class": predicted_class
                    }

                    st.session_state[
                        "session_history"
                    ].append(record)

                    st.success(
                        "Risk analysis completed successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction failed: {e}"
                    )

    # --------------------------------------------------------
    # CURRENT RISK
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="section-title">Current Risk</div>',
            unsafe_allow_html=True
        )

        if (
            st.session_state[
                "analysis_complete"
            ]
            and
            st.session_state[
                "risk_probability"
            ] is not None
        ):

            probability = (
                st.session_state[
                    "risk_probability"
                ]
            )

            predicted_class = (
                st.session_state[
                    "predicted_class"
                ]
            )

            if predicted_class == 1:

                st.markdown(
                    """
                    <div class="risk-high">
                        🚨 HIGH RISK DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-low">
                        ✓ LOW RISK DETECTED
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            m1, m2, m3 = st.columns(3)

            with m1:

                st.metric(
                    "Risk Probability",
                    f"{probability:.2%}"
                )

            with m2:

                st.metric(
                    "Threshold",
                    f"{MODEL_THRESHOLD:.2f}"
                )

            with m3:

                st.metric(
                    "Decision Margin",
                    f"{probability - MODEL_THRESHOLD:+.3f}"
                )

            st.plotly_chart(
                create_gauge(
                    probability,
                    MODEL_THRESHOLD
                ),
                use_container_width=True
            )

            scenario = st.session_state.get(
                "current_scenario",
                "Current interaction"
            )

            st.caption(
                f"Scenario: {scenario}"
            )

            if predicted_class == 1:

                st.markdown(
                    """
                    <div class="risk-danger">
                        The predicted risk probability is above
                        the selected decision threshold.
                        The interaction is classified as
                        <b>High Risk</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-info">
                        The predicted risk probability is below
                        the selected decision threshold.
                        The interaction is classified as
                        <b>Low Risk</b>.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Select or upload a sequence and click "
                "'Analyze Interaction' to generate a prediction."
            )


# ============================================================
# TAB 2 — INTERACTION ANALYSIS
# ============================================================

with tabs[1]:

    st.markdown(
        '<div class="section-title">🔬 Interaction Analysis</div>',
        unsafe_allow_html=True
    )

    if not st.session_state[
        "analysis_complete"
    ]:

        st.info(
            "Run a prediction first to view interaction analysis."
        )

    else:

        probability = (
            st.session_state[
                "risk_probability"
            ]
        )

        sequence = (
            st.session_state[
                "current_sequence"
            ]
        )

        predicted_class = (
            st.session_state[
                "predicted_class"
            ]
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Risk Interpretation"
            )

            if predicted_class == 1:

                st.markdown(
                    """
                    <div class="risk-danger">
                        <b>HIGH-RISK INTERACTION</b><br><br>
                        The model probability exceeds the
                        0.40 decision threshold.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    """
                    <div class="risk-info">
                        <b>LOW-RISK INTERACTION</b><br><br>
                        The model probability is below the
                        0.40 decision threshold.
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with col2:

            st.markdown(
                "### Probability Details"
            )

            st.metric(
                "Predicted Probability",
                f"{probability:.2%}"
            )

            st.metric(
                "Threshold",
                f"{MODEL_THRESHOLD:.2%}"
            )

            st.metric(
                "Margin",
                f"{probability - MODEL_THRESHOLD:+.3f}"
            )

        st.markdown(
            "### Temporal Feature Overview"
        )

        sequence_df = pd.DataFrame(
            sequence,
            columns=FEATURE_NAMES
        )

        sequence_df["Timestep"] = [
            i + 1
            for i in range(
                len(sequence_df)
            )
        ]

        selected_features = st.multiselect(
            "Select features to visualize",
            FEATURE_NAMES,
            default=FEATURE_NAMES[:4]
        )

        if selected_features:

            chart_df = sequence_df[
                ["Timestep"]
                + selected_features
            ]

            chart_df = chart_df.set_index(
                "Timestep"
            )

            st.line_chart(
                chart_df,
                use_container_width=True
            )

        else:

            st.info(
                "Select at least one feature."
            )

        st.markdown(
            "### Sequence Statistics"
        )

        stats_df = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Mean": np.mean(
                    sequence,
                    axis=0
                ),
                "Std": np.std(
                    sequence,
                    axis=0
                ),
                "Minimum": np.min(
                    sequence,
                    axis=0
                ),
                "Maximum": np.max(
                    sequence,
                    axis=0
                )
            }
        )

        st.dataframe(
            stats_df.round(4),
            use_container_width=True
        )


# ============================================================
# TAB 3 — FINAL MODEL PERFORMANCE
# ============================================================

with tabs[2]:

    st.markdown(
        '<div class="section-title">📊 Final Model Performance</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        FINAL_MODEL_RESULTS.style.format(
            {
                "Threshold": "{:.2f}",
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
            }
        ),
        use_container_width=True
    )

    st.markdown(
        "### Performance Comparison"
    )

    metric = st.selectbox(
        "Select metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC"
        ]
    )

    fig = px.bar(
        FINAL_MODEL_RESULTS,
        x="Model",
        y=metric,
        text_auto=".3f",
        title=f"{metric} Comparison"
    )

    fig.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        font_color="#f8fafc"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        "### Best Performing Models"
    )

    best_accuracy = FINAL_MODEL_RESULTS.loc[
        FINAL_MODEL_RESULTS["Accuracy"].idxmax()
    ]

    best_precision = FINAL_MODEL_RESULTS.loc[
        FINAL_MODEL_RESULTS["Precision"].idxmax()
    ]

    best_recall = FINAL_MODEL_RESULTS.loc[
        FINAL_MODEL_RESULTS["Recall"].idxmax()
    ]

    best_f1 = FINAL_MODEL_RESULTS.loc[
        FINAL_MODEL_RESULTS["F1 Score"].idxmax()
    ]

    best_auc = FINAL_MODEL_RESULTS.loc[
        FINAL_MODEL_RESULTS["ROC-AUC"].idxmax()
    ]

    cols = st.columns(5)

    with cols[0]:
        st.metric(
            "Best Accuracy",
            best_accuracy["Model"],
            f'{best_accuracy["Accuracy"]:.4f}'
        )

    with cols[1]:
        st.metric(
            "Best Precision",
            best_precision["Model"],
            f'{best_precision["Precision"]:.4f}'
        )

    with cols[2]:
        st.metric(
            "Best Recall",
            best_recall["Model"],
            f'{best_recall["Recall"]:.4f}'
        )

    with cols[3]:
        st.metric(
            "Best F1",
            best_f1["Model"],
            f'{best_f1["F1 Score"]:.4f}'
        )

    with cols[4]:
        st.metric(
            "Best ROC-AUC",
            best_auc["Model"],
            f'{best_auc["ROC-AUC"]:.4f}'
        )

    st.markdown(
        "### Final LSTM Test Evaluation"
    )

    eval_cols = st.columns(6)

    metrics = [
        ("Accuracy", 0.9002),
        ("Precision", 0.7036),
        ("Recall", 0.9280),
        ("F1 Score", 0.8004),
        ("ROC-AUC", 0.9602),
        ("PR-AUC", 0.8267),
    ]

    for col, (name, value) in zip(
        eval_cols,
        metrics
    ):

        with col:

            st.metric(
                name,
                f"{value:.4f}"
            )

    st.markdown(
        "### Final LSTM Confusion Matrix"
    )

    cm = np.array(
        [
            [36922, 4444],
            [819, 10551]
        ]
    )

    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Low Risk",
            "Predicted High Risk"
        ],
        y=[
            "Actual Low Risk",
            "Actual High Risk"
        ],
        title="LSTM Confusion Matrix — Threshold 0.40"
    )

    cm_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        font_color="#f8fafc"
    )

    st.plotly_chart(
        cm_fig,
        use_container_width=True
    )

    st.markdown(
        "### Final LSTM Error Analysis"
    )

    error_df = pd.DataFrame(
        {
            "Category": [
                "True Negative",
                "False Positive",
                "False Negative",
                "True Positive"
            ],
            "Count": [
                36922,
                4444,
                819,
                10551
            ]
        }
    )

    error_fig = px.bar(
        error_df,
        x="Category",
        y="Count",
        text="Count",
        title="LSTM Prediction Error Distribution"
    )

    error_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0b1220",
        font_color="#f8fafc"
    )

    st.plotly_chart(
        error_fig,
        use_container_width=True
    )

    st.markdown(
        """
        **Final interpretation**

        The LSTM achieved the highest ROC-AUC, PR-AUC,
        recall and F1 score among the evaluated models.
        At the selected threshold of 0.40, the model prioritizes
        detection of high-risk interactions, achieving 92.80%
        recall while maintaining 90.02% accuracy.
        """
    )


# ============================================================
# TAB 4 — SESSION ANALYTICS
# ============================================================

with tabs[3]:

    st.markdown(
        '<div class="section-title">📈 Current Session Analytics</div>',
        unsafe_allow_html=True
    )

    history = st.session_state[
        "session_history"
    ]

    if len(history) == 0:

        st.info(
            "No predictions have been recorded in this session yet."
        )

    else:

        history_df = pd.DataFrame(
            history
        )

        # Explicitly guarantee required columns.
        required_columns = [
            "Timestamp",
            "Source",
            "Scenario",
            "Probability",
            "Threshold",
            "Prediction",
            "Class"
        ]

        for column in required_columns:

            if column not in history_df.columns:

                if column == "Prediction":

                    history_df[column] = np.where(
                        history_df["Class"] == 1,
                        "High Risk",
                        "Low Risk"
                    )

                elif column == "Threshold":

                    history_df[column] = MODEL_THRESHOLD

                else:

                    history_df[column] = "Unknown"

        # Never directly assume Prediction exists.
        prediction_series = history_df[
            "Prediction"
        ].astype(str)

        total_predictions = len(
            history_df
        )

        high_risk_count = int(
            (
                history_df["Class"] == 1
            ).sum()
        )

        low_risk_count = int(
            (
                history_df["Class"] == 0
            ).sum()
        )

        average_probability = float(
            history_df[
                "Probability"
            ].mean()
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Predictions",
                total_predictions
            )

        with c2:
            st.metric(
                "High Risk",
                high_risk_count
            )

        with c3:
            st.metric(
                "Low Risk",
                low_risk_count
            )

        with c4:
            st.metric(
                "Average Risk",
                f"{average_probability:.2%}"
            )

        st.markdown(
            "### Session Prediction History"
        )

        display_df = history_df[
            [
                "Timestamp",
                "Source",
                "Scenario",
                "Probability",
                "Threshold",
                "Prediction"
            ]
        ].copy()

        display_df["Probability"] = (
            display_df["Probability"]
            .map(
                lambda x: f"{x:.2%}"
            )
        )

        display_df["Threshold"] = (
            display_df["Threshold"]
            .map(
                lambda x: f"{x:.2f}"
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True
        )

        st.markdown(
            "### Session Risk Distribution"
        )

        distribution_df = pd.DataFrame(
            {
                "Prediction": [
                    "Low Risk",
                    "High Risk"
                ],
                "Count": [
                    low_risk_count,
                    high_risk_count
                ]
            }
        )

        fig = px.pie(
            distribution_df,
            names="Prediction",
            values="Count",
            hole=0.55
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown(
            "### Session Probability Trend"
        )

        trend_df = history_df.copy()

        trend_df[
            "Prediction Number"
        ] = range(
            1,
            len(trend_df) + 1
        )

        fig2 = px.line(
            trend_df,
            x="Prediction Number",
            y="Probability",
            markers=True,
            title="Risk Probability Across Predictions"
        )

        fig2.add_hline(
            y=MODEL_THRESHOLD,
            line_dash="dash",
            annotation_text="Threshold"
        )

        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0b1220",
            font_color="#f8fafc"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        csv_export = history_df.to_csv(
            index=False
        )

        st.download_button(
            "Download Session Analytics CSV",
            csv_export,
            file_name="risk_prediction_session.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button(
            "Clear Current Session"
        ):

            st.session_state[
                "session_history"
            ] = []

            st.session_state[
                "analysis_complete"
            ] = False

            st.session_state[
                "risk_probability"
            ] = None

            st.session_state[
                "predicted_class"
            ] = None

            st.rerun()


# ============================================================
# TAB 5 — INCIDENT REPORTS
# ============================================================

with tabs[4]:

    st.markdown(
        '<div class="section-title">📋 Incident Reports</div>',
        unsafe_allow_html=True
    )

    history = st.session_state[
        "session_history"
    ]

    if len(history) == 0:

        st.info(
            "No incidents have been generated in this session."
        )

    else:

        history_df = pd.DataFrame(
            history
        )

        high_risk_df = history_df[
            history_df["Class"] == 1
        ].copy()

        if len(high_risk_df) == 0:

            st.success(
                "No high-risk interactions have been "
                "detected in the current session."
            )

        else:

            st.warning(
                f"{len(high_risk_df)} high-risk "
                "interaction(s) detected."
            )

            for idx, row in high_risk_df.iterrows():

                probability = float(
                    row["Probability"]
                )

                with st.expander(
                    f"🚨 Incident {idx + 1} — "
                    f"{probability:.2%} risk"
                ):

                    col1, col2 = st.columns(2)

                    with col1:

                        st.write(
                            "**Timestamp:**"
                        )

                        st.write(
                            str(
                                row["Timestamp"]
                            )
                        )

                        st.write(
                            "**Source:**"
                        )

                        st.write(
                            str(
                                row["Source"]
                            )
                        )

                        st.write(
                            "**Scenario:**"
                        )

                        st.write(
                            str(
                                row["Scenario"]
                            )
                        )

                    with col2:

                        st.write(
                            "**Risk Probability:**"
                        )

                        st.write(
                            f"{probability:.2%}"
                        )

                        st.write(
                            "**Threshold:**"
                        )

                        st.write(
                            f"{float(row['Threshold']):.2f}"
                        )

                        st.write(
                            "**Classification:**"
                        )

                        st.write(
                            "HIGH RISK"
                        )

                    if probability >= 0.75:

                        severity = "CRITICAL"

                    elif probability >= 0.50:

                        severity = "HIGH"

                    else:

                        severity = "MODERATE"

                    st.markdown(
                        f"""
                        <div class="risk-danger">

                        <b>Severity: {severity}</b><br><br>

                        The LSTM model estimated a high-risk
                        interaction probability of
                        <b>{probability:.2%}</b>.

                        This result should be treated as a
                        decision-support signal rather than
                        a standalone safety determination.

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            incident_export = high_risk_df.to_csv(
                index=False
            )

            st.download_button(
                "Download Incident Report CSV",
                incident_export,
                file_name="high_risk_incident_report.csv",
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>
    <hr>

    <div style="
        text-align:center;
        color:#64748b;
        font-size:13px;
        padding:15px;">

        Deep Learning-Based Pedestrian–Vehicle
        Interaction Risk Prediction for Proactive Road Safety

        <br><br>

        LSTM model • 15 timesteps • 15 features •
        Decision threshold = 0.40

    </div>
    """,
    unsafe_allow_html=True
)