
#!/usr/bin/env python3
"""
Train generic and species-wise antifungal models from merged ChEMBL-like CSV.

Expected columns:
- canonical_smiles
- target_organism
- binary_label
Optional columns used during aggregation:
- standard_value
- standard_type
- standard_units
- confidence_tier

Usage:
    python train_antifungal_models.py --input merged.csv --outdir model_output
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


DESC_NAMES = [
    "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "FractionCSP3", "HeavyAtomCount",
]


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonicalize_smiles(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def morgan_fp(smiles: str, radius: int = 2, nbits: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_desc(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
    ], dtype=float)


def featurize_smiles(smiles_list: List[str]) -> Tuple[np.ndarray, List[int]]:
    fps, descs, keep = [], [], []
    for i, smi in enumerate(smiles_list):
        fp = morgan_fp(smi)
        desc = rdkit_desc(smi)
        if fp is None or desc is None:
            continue
        fps.append(fp)
        descs.append(desc)
        keep.append(i)
    if not fps:
        return np.empty((0, 2057)), []
    X = np.hstack([np.asarray(fps), np.asarray(descs)])
    return X, keep


def confidence_weight(x: str) -> float:
    if pd.isna(x):
        return 0.7
    x = str(x).lower()
    if "tier-1" in x:
        return 1.0
    if "tier-2" in x:
        return 0.85
    if "tier-3" in x:
        return 0.65
    if "tier-4" in x:
        return 0.40
    return 0.70


def choose_model(random_state: int = 42):
    if HAS_LGBM:
        # Conservative LightGBM settings for small species-wise datasets.
        # verbose=-1 suppresses repeated warnings during training.
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=2,
            min_data_in_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    return RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def compute_metrics(y_true, prob, pred):
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    return metrics


def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    req = ["canonical_smiles", "target_organism", "binary_label"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.dropna(subset=req)
    df["binary_label"] = df["binary_label"].astype(int)
    df["canonical_smiles"] = df["canonical_smiles"].map(canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df["sample_weight"] = df.get("confidence_tier", "").map(confidence_weight)

    # Remove conflicting duplicates at compound-species level
    grp = df.groupby(["canonical_smiles", "target_organism"])["binary_label"].nunique().reset_index(name="n_labels")
    clean_keys = grp.loc[grp["n_labels"] == 1, ["canonical_smiles", "target_organism"]]
    df = df.merge(clean_keys, on=["canonical_smiles", "target_organism"], how="inner")

    agg_dict = {
        "binary_label": "first",
        "sample_weight": "max",
    }
    for col in ["compound_name", "compound_class", "standard_type", "standard_units", "activity_comment", "confidence_tier"]:
        if col in df.columns:
            agg_dict[col] = "first"
    if "standard_value" in df.columns:
        agg_dict["standard_value"] = "median"

    agg = (
        df.groupby(["canonical_smiles", "target_organism"], as_index=False)
          .agg(agg_dict)
    )
    agg["scaffold"] = agg["canonical_smiles"].map(get_scaffold)
    return agg


def build_generic_dataset(df_agg: pd.DataFrame) -> pd.DataFrame:
    # compound positive if active against at least one selected species
    out = (
        df_agg.groupby("canonical_smiles", as_index=False)
        .agg({
            "binary_label": "max",
            "sample_weight": "max",
            "compound_name": "first" if "compound_name" in df_agg.columns else "count",
            "compound_class": "first" if "compound_class" in df_agg.columns else "count",
            "scaffold": "first",
        })
        .rename(columns={"binary_label": "generic_label"})
    )
    return out


def split_by_scaffold(X, y, groups, test_size=0.2, random_state=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_train, idx_test = next(gss.split(X, y, groups=groups))
    return idx_train, idx_test


def train_one_model(df_model: pd.DataFrame, label_col: str, out_prefix: Path):
    X, keep = featurize_smiles(df_model["canonical_smiles"].tolist())
    df_model = df_model.iloc[keep].reset_index(drop=True)
    y = df_model[label_col].values
    w = df_model.get("sample_weight", pd.Series(np.ones(len(df_model)))).values
    groups = df_model.get("scaffold", pd.Series([""] * len(df_model))).values

    idx_train, idx_test = split_by_scaffold(X, y, groups)
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    w_train = w[idx_train]

    base = choose_model()
    base.fit(X_train, y_train, sample_weight=w_train)

    # Calibrate on a holdout slice of training data if enough data
    if len(np.unique(y_train)) == 2 and len(y_train) >= 40:
        X_subtr, X_cal, y_subtr, y_cal, w_subtr, w_cal = train_test_split(
            X_train, y_train, w_train,
            test_size=0.2, stratify=y_train, random_state=42
        )
        base = choose_model()
        base.fit(X_subtr, y_subtr, sample_weight=w_subtr)

        # scikit-learn >=1.6 deprecates/removes cv="prefit".
        # Use FrozenEstimator when available; fall back to cv="prefit" for older versions.
        if FrozenEstimator is not None:
            clf = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        else:
            clf = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")

        clf.fit(X_cal, y_cal, sample_weight=w_cal)
    else:
        clf = base

    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, prob, pred)

    joblib.dump(clf, out_prefix.with_suffix(".pkl"))
    meta = {
        "n_rows": int(len(df_model)),
        "n_train": int(len(idx_train)),
        "n_test": int(len(idx_test)),
        "n_pos": int(np.sum(y)),
        "n_neg": int(np.sum(1 - y)),
        "metrics": metrics,
        "descriptor_names": DESC_NAMES,
        "fingerprint": {"type": "Morgan", "radius": 2, "nBits": 2048},
    }
    out_prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_rows_species", type=int, default=30)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    agg = clean_and_aggregate(raw)
    agg.to_csv(outdir / "aggregated_compound_species.csv", index=False)

    summary = {
        "raw_rows": int(len(raw)),
        "aggregated_rows": int(len(agg)),
        "n_species": int(agg["target_organism"].nunique()),
        "species_counts": agg["target_organism"].value_counts().to_dict(),
    }

    generic = build_generic_dataset(agg)
    generic.to_csv(outdir / "generic_compound_dataset.csv", index=False)
    summary["generic_model"] = train_one_model(
        generic.rename(columns={"generic_label": "label"}),
        label_col="label",
        out_prefix=outdir / "generic_antifungal_model",
    )

    species_metrics = {}
    for species, sdf in agg.groupby("target_organism"):
        if len(sdf) < args.min_rows_species or sdf["binary_label"].nunique() < 2:
            continue
        safe_name = species.lower().replace(" ", "_").replace(".", "").replace("/", "_")
        species_metrics[species] = train_one_model(
            sdf,
            label_col="binary_label",
            out_prefix=outdir / f"{safe_name}_model",
        )

    summary["species_models"] = species_metrics
    (outdir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
