import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.DataStructs import ConvertToNumpyArray
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

APP_TITLE = "PhytoMyco-Pred"
APP_SUBTITLE = "AI-assisted antifungal prediction for plant-pathogenic fungi"
DEFAULT_MODEL_DIR = Path("models")
DEFAULT_SPECIES = [
    "Alternaria solani",
    "Aspergillus flavus",
    "Botrytis cinerea",
    "Colletotrichum gloeosporioides",
    "Fusarium oxysporum",
    "Magnaporthe oryzae",
    "Penicillium expansum",
    "Puccinia graminis f. sp. tritici",
    "Rhizoctonia solani",
    "Sclerotinia sclerotiorum",
]

GENERIC_THRESHOLD = 0.50
SPECIES_THRESHOLD = 0.50
NAV_OPTIONS = ["Home", "Prediction", "Tutorial", "Contact"]


# ------------------------------
# Utilities
# ------------------------------

def safe_slug(text: str) -> str:
    return (
        text.lower()
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
    )


def make_sample_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_id": ["CMPD_001", "CMPD_002", "CMPD_003"],
            "canonical_smiles": [
                "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
                "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
                "COc1ccccc1C=C",
            ],
        }
    )


def mol_from_smiles(smiles: str):
    if not RDKIT_AVAILABLE:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def fingerprint_from_smiles(smiles: str, radius: int = 2, nbits: int = 2048) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def descriptors_from_smiles(smiles: str) -> Optional[np.ndarray]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.HeavyAtomCount(mol),
        ],
        dtype=np.float32,
    )


def feature_vector(smiles: str) -> Optional[np.ndarray]:
    fp = fingerprint_from_smiles(smiles)
    desc = descriptors_from_smiles(smiles)
    if fp is None or desc is None:
        return None
    return np.hstack([fp, desc]).astype(np.float32)


def predict_probability(model, x: np.ndarray) -> float:
    x2 = x.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(x2)[:, 1][0])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(x2)[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    pred = model.predict(x2)
    return float(np.clip(pred[0], 0, 1))


def confidence_from_probability(prob: float) -> str:
    if prob >= 0.80:
        return "High"
    if prob >= 0.60:
        return "Medium"
    return "Low"


def generic_label(prob: float) -> str:
    return "Antifungal" if prob >= GENERIC_THRESHOLD else "Non-antifungal"


def species_label(prob: float) -> str:
    return "Likely target" if prob >= SPECIES_THRESHOLD else "Unlikely target"


def format_score(prob: Optional[float]) -> Optional[float]:
    if prob is None or pd.isna(prob):
        return None
    return round(float(prob), 4)


# ------------------------------
# Model loading
# ------------------------------
@st.cache_resource

def load_models() -> Tuple[Optional[object], Dict[str, object], Dict[str, dict]]:
    model_dir = DEFAULT_MODEL_DIR
    generic_model = None
    species_models: Dict[str, object] = {}
    metadata: Dict[str, dict] = {}

    generic_path = model_dir / "generic_antifungal_model.pkl"
    if generic_path.exists():
        generic_model = joblib.load(generic_path)

    generic_json = model_dir / "generic_antifungal_model.json"
    if generic_json.exists():
        try:
            metadata["generic"] = json.loads(generic_json.read_text())
        except Exception:
            metadata["generic"] = {}

    for species in DEFAULT_SPECIES:
        slug = safe_slug(species)
        pkl_path = model_dir / f"{slug}_model.pkl"
        json_path = model_dir / f"{slug}_model.json"
        if pkl_path.exists():
            species_models[species] = joblib.load(pkl_path)
        if json_path.exists():
            try:
                metadata[species] = json.loads(json_path.read_text())
            except Exception:
                metadata[species] = {}

    return generic_model, species_models, metadata


# ------------------------------
# Prediction engine
# ------------------------------

def run_predictions(df: pd.DataFrame, generic_model, species_models: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[dict] = []
    detail_rows: List[dict] = []

    for idx, row in df.iterrows():
        compound_id = str(row.get("compound_id", f"CMPD_{idx+1:03d}")).strip() or f"CMPD_{idx+1:03d}"
        raw_smiles = str(row.get("canonical_smiles", "")).strip()
        canonical_smiles = canonicalize_smiles(raw_smiles) if raw_smiles else None

        if not canonical_smiles:
            summary_rows.append(
                {
                    "compound_id": compound_id,
                    "input_smiles": raw_smiles,
                    "canonical_smiles": None,
                    "status": "Invalid SMILES",
                    "generic_antifungal_score": None,
                    "generic_prediction": None,
                    "generic_confidence": None,
                    "species_screening": "Not run",
                    "top_species": None,
                    "top_species_score": None,
                }
            )
            continue

        feats = feature_vector(canonical_smiles)
        if feats is None:
            summary_rows.append(
                {
                    "compound_id": compound_id,
                    "input_smiles": raw_smiles,
                    "canonical_smiles": canonical_smiles,
                    "status": "Feature generation failed",
                    "generic_antifungal_score": None,
                    "generic_prediction": None,
                    "generic_confidence": None,
                    "species_screening": "Not run",
                    "top_species": None,
                    "top_species_score": None,
                }
            )
            continue

        generic_score = predict_probability(generic_model, feats) if generic_model is not None else 0.0
        run_species = generic_score >= GENERIC_THRESHOLD and len(species_models) > 0
        species_scores = []

        if run_species:
            for species, model in species_models.items():
                try:
                    score = predict_probability(model, feats)
                except Exception:
                    score = np.nan
                species_scores.append((species, score))
                detail_rows.append(
                    {
                        "compound_id": compound_id,
                        "canonical_smiles": canonical_smiles,
                        "species": species,
                        "score": format_score(score),
                        "target_status": None if pd.isna(score) else species_label(float(score)),
                        "confidence": None if pd.isna(score) else confidence_from_probability(float(score)),
                    }
                )

        valid_species_scores = [(s, sc) for s, sc in species_scores if not pd.isna(sc)]
        top_species = None
        top_species_score = None
        if valid_species_scores:
            top_species, top_species_score = sorted(valid_species_scores, key=lambda x: x[1], reverse=True)[0]

        summary_rows.append(
            {
                "compound_id": compound_id,
                "input_smiles": raw_smiles,
                "canonical_smiles": canonical_smiles,
                "status": "OK",
                "generic_antifungal_score": format_score(generic_score),
                "generic_prediction": generic_label(generic_score),
                "generic_confidence": confidence_from_probability(generic_score),
                "species_screening": "Run" if run_species else "Skipped",
                "top_species": top_species,
                "top_species_score": format_score(top_species_score),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ------------------------------
# Styling
# ------------------------------

def set_page_config():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧫",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_css():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none !important;}
        #MainMenu, footer, header {visibility: hidden;}

        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            background: linear-gradient(135deg, #153b6f 0%, #1d5fa8 55%, #2a88c9 100%);
            padding: 2rem 2rem 1.7rem 2rem;
            border-radius: 24px;
            color: white;
            box-shadow: 0 16px 40px rgba(25, 80, 140, 0.20);
            margin-bottom: 1.3rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            line-height: 1.7;
            color: rgba(255,255,255,0.92);
            max-width: 850px;
        }
        .pill {
            display:inline-block;
            padding:0.35rem 0.75rem;
            background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 999px;
            font-size: 0.86rem;
            font-weight: 600;
            margin-bottom: 0.9rem;
        }
        .soft-card {
            background: #ffffff;
            border: 1px solid rgba(31, 78, 121, 0.10);
            border-radius: 20px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            box-shadow: 0 10px 28px rgba(17, 52, 86, 0.06);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(31, 78, 121, 0.10);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 20px rgba(17, 52, 86, 0.05);
        }
        .metric-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #58708f;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            color: #153b6f;
            line-height: 1.1;
        }
        .mini-note {
            font-size: 0.9rem;
            color: #5a6f85;
            line-height: 1.6;
        }
        .section-title {
            font-size: 1.5rem;
            font-weight: 800;
            color: #153b6f;
            margin-bottom: 0.35rem;
        }
        .section-subtitle {
            font-size: 0.95rem;
            color: #5a6f85;
            margin-bottom: 1rem;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: none;
            background: linear-gradient(135deg, #194d8b 0%, #2a88c9 100%);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            filter: brightness(1.02);
            box-shadow: 0 10px 24px rgba(25, 77, 139, 0.20);
        }
        .stTextInput > div > div, .stTextArea textarea, .stFileUploader, .stSelectbox > div > div {
            border-radius: 12px !important;
        }
        .nav-wrap {
            margin-bottom: 1rem;
        }
        .footer-note {
            font-size: 0.85rem;
            color: #6a7d92;
            text-align: center;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="mini-note">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        f"""
        <div class="hero">
            <div class="pill">Web server for plant-pathogenic fungi</div>
            <div class="hero-title">{APP_TITLE}</div>
            <div class="hero-subtitle">
                {APP_SUBTITLE}. Screen single compounds or batch libraries to identify likely antifungal molecules,
                then estimate the fungal species they are most likely to target.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_summary(summary_df: pd.DataFrame):
    if summary_df.empty:
        return
    row = summary_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Generic score", f"{row['generic_antifungal_score']:.3f}" if pd.notna(row['generic_antifungal_score']) else "NA")
    with c2:
        render_metric_card("Prediction", str(row["generic_prediction"]))
    with c3:
        render_metric_card("Confidence", str(row["generic_confidence"]))
    with c4:
        render_metric_card("Top fungal target", str(row["top_species"]) if pd.notna(row["top_species"]) else "Not applicable")


# ------------------------------
# UI
# ------------------------------

def render_navigation() -> str:
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    selected = st.radio(
        "Navigation",
        NAV_OPTIONS,
        key="nav",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    return selected


def render_home():
    render_hero()

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">About the platform</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">PhytoMycoPred first predicts whether a query compound is likely to possess antifungal potential. '
            'Only compounds predicted as antifungal are subsequently evaluated by species-wise models to estimate likely fungal targets.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            - **Generic antifungal screening** for rapid prioritization of candidate compounds.
            - **Species-wise targeting analysis** across major plant-pathogenic fungi.
            - **Single-compound** and **batch CSV** prediction modes.
            - **Downloadable output tables** for downstream analysis.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Expected input</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Upload a CSV containing a <code>canonical_smiles</code> column. '
            'You may optionally add a <code>compound_id</code> column.</div>',
            unsafe_allow_html=True,
        )
        st.code("compound_id,canonical_smiles\nCMPD_001,O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Supported fungal species</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Current species panel used by the species-wise prediction module.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    midpoint = int(np.ceil(len(DEFAULT_SPECIES) / 2))
    with col1:
        for species in DEFAULT_SPECIES[:midpoint]:
            st.markdown(f"- {species}")
    with col2:
        for species in DEFAULT_SPECIES[midpoint:]:
            st.markdown(f"- {species}")
    st.markdown('</div>', unsafe_allow_html=True)


def render_prediction(generic_model, species_models: Dict[str, object]):
    st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter a single compound or upload a batch CSV. '
        'Species-wise screening is performed only for compounds predicted as antifungal by the generic model.</div>',
        unsafe_allow_html=True,
    )

    if not RDKIT_AVAILABLE:
        st.error("RDKit is not available in this environment. Please install RDKit before running the app.")
        return

    if generic_model is None and not species_models:
        st.warning("No trained models were found in the default \'models\' directory.")
        return

    mode = st.radio("Choose input mode", ["Single compound", "Batch CSV upload"], horizontal=True, key="prediction_mode")

    if mode == "Single compound":
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        with st.form("single_prediction_form"):
            compound_id = st.text_input("Compound ID", value="CMPD_001")
            smiles = st.text_area(
                "Canonical SMILES",
                value="O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
                height=100,
            )
            submitted = st.form_submit_button("Run prediction")
        st.markdown('</div>', unsafe_allow_html=True)

        if submitted:
            input_df = pd.DataFrame({"compound_id": [compound_id], "canonical_smiles": [smiles]})
            summary_df, detail_df = run_predictions(input_df, generic_model, species_models)
            st.session_state["single_summary_df"] = summary_df
            st.session_state["single_detail_df"] = detail_df

        summary_df = st.session_state.get("single_summary_df", pd.DataFrame())
        detail_df = st.session_state.get("single_detail_df", pd.DataFrame())

        if not summary_df.empty:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Prediction summary</div>', unsafe_allow_html=True)
            render_status_summary(summary_df)
            display_summary = summary_df.copy()
            if not display_summary.empty:
                display_summary["generic_prediction"] = display_summary["generic_prediction"].fillna("-")
                display_summary["generic_confidence"] = display_summary["generic_confidence"].fillna("-")
                st.dataframe(display_summary, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if detail_df.empty:
                st.info("Species-wise analysis was not run because the compound was predicted as non-antifungal, or no species models are available.")
            else:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Species-wise targeting analysis</div>', unsafe_allow_html=True)
                st.markdown('<div class="section-subtitle">Higher scores indicate stronger predicted targeting likelihood for the corresponding fungal species.</div>', unsafe_allow_html=True)
                detail_df = detail_df.sort_values("score", ascending=False, na_position="last")
                st.dataframe(detail_df, use_container_width=True, hide_index=True)
                chart_df = detail_df.dropna(subset=["score"]).set_index("species")[["score"]]
                if not chart_df.empty:
                    st.bar_chart(chart_df)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.markdown("Upload a CSV containing `canonical_smiles` and an optional `compound_id` column.")
        sample_df = make_sample_input()
        c1, c2 = st.columns([1.1, 1])
        with c1:
            uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
        with c2:
            st.download_button(
                "Download sample input CSV",
                data=sample_df.to_csv(index=False).encode("utf-8"),
                file_name="sample_antifungal_input.csv",
                mime="text/csv",
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read the uploaded CSV: {e}")
                return

            if "canonical_smiles" not in input_df.columns:
                st.error("The uploaded file must contain a `canonical_smiles` column.")
                return

            if "compound_id" not in input_df.columns:
                input_df["compound_id"] = [f"CMPD_{i+1:03d}" for i in range(len(input_df))]

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Uploaded preview</div>', unsafe_allow_html=True)
            st.dataframe(input_df.head(20), use_container_width=True, hide_index=True)
            run_batch = st.button("Run batch prediction", key="run_batch_btn")
            st.markdown('</div>', unsafe_allow_html=True)

            if run_batch:
                summary_df, detail_df = run_predictions(input_df, generic_model, species_models)
                st.session_state["batch_summary_df"] = summary_df
                st.session_state["batch_detail_df"] = detail_df

            summary_df = st.session_state.get("batch_summary_df", pd.DataFrame())
            detail_df = st.session_state.get("batch_detail_df", pd.DataFrame())

            if not summary_df.empty:
                st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Summary results</div>', unsafe_allow_html=True)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download summary results CSV",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="antifungal_summary_predictions.csv",
                    mime="text/csv",
                )
                st.markdown('</div>', unsafe_allow_html=True)

                if not detail_df.empty:
                    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">Species-wise targeting analysis</div>', unsafe_allow_html=True)
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download detailed results CSV",
                        data=detail_df.to_csv(index=False).encode("utf-8"),
                        file_name="antifungal_species_predictions.csv",
                        mime="text/csv",
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Species-wise analysis was skipped for all uploaded compounds because they were predicted as non-antifungal, or no species models are available.")


def render_tutorial():
    st.markdown('<div class="section-title">Tutorial</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        """
        ### How to use PhytoMycoPred
        1. Open the **Prediction** section.
        2. Choose **Single compound** or **Batch CSV upload**.
        3. Enter a valid SMILES string or upload a CSV file.
        4. Review the **generic antifungal score**.
        5. If the compound is predicted as antifungal, inspect the **species-wise targeting analysis**.
        6. Download summary and detailed results for downstream interpretation.
        """
    )
    st.markdown(
        """
        ### Output interpretation
        - **Antifungal**: the compound is predicted to possess general antifungal potential against plant-pathogenic fungi.
        - **Non-antifungal**: the compound is not prioritized by the generic model.
        - **Likely target**: a fungal species with higher predicted targeting likelihood.
        - **Unlikely target**: a fungal species not prioritized by the species-wise model.
        """
    )
    st.info(
        "Predictions are computational and should be followed by experimental validation. "
        "This platform is intended for early-stage prioritization, not as a substitute for laboratory testing."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_contact():
    st.markdown('<div class="section-title">Contact</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown(
        """
        **Principal Investigator**  
        Dr. Sneha Murmu  
        Scientist (Bioinformatics)  
        Division of Agricultural Bioinformatics  
        ICAR-Indian Agricultural Statistics Research Institute, New Delhi, India  
        Email: murmu.sneha07@gmail.com
        """
    )
    st.markdown(
        "For feedback, collaboration queries, or technical issues related to the server, please contact the above email address."
    )
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    set_page_config()
    inject_css()

    if "nav" not in st.session_state:
        st.session_state["nav"] = "Home"

    generic_model, species_models, metadata = load_models()
    selected_nav = render_navigation()

    if selected_nav == "Home":
        render_home()
    elif selected_nav == "Prediction":
        render_prediction(generic_model, species_models)
    elif selected_nav == "Tutorial":
        render_tutorial()
    else:
        render_contact()

    st.markdown('<div class="footer-note">PhytoMycoPred • AI-assisted antifungal prediction for plant-pathogenic fungi</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
