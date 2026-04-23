#!/usr/bin/env python3
"""
download_yeast_arff.py

Downloads the Yeast multi-label dataset from OpenML and saves it in ARFF format
as:
  data/yeast/yeast-train.arff
  data/yeast/yeast-test.arff
(optionally writes a simple yeast.xml listing the label names)

These files are compatible with a loader that expects:
- last 14 columns to be binary label indicators {0,1}
- the rest to be numeric features

Usage:
  python download_yeast_arff.py
  python download_yeast_arff.py --outdir data/yeast --test-size 0.2 --seed 42

Dependencies:
  - openml
  - numpy
  - pandas
  - scikit-learn
"""

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import openml  # pip install openml
except ImportError as e:
    print("ERROR: The 'openml' package is required. Install it via: pip install openml")
    sys.exit(1)

from sklearn.model_selection import train_test_split


OPENML_DATASET_ID = 40597  # Yeast (multi-label) on OpenML


def sanitize_names(names: List[str]) -> List[str]:
    """
    Make attribute names ARFF-friendly and unique:
    - Replace non-alphanumeric characters with underscores
    - Ensure names start with a letter (prefix with 'f_' if needed)
    - De-duplicate by appending _i suffix if necessary
    """
    def sanitize_one(n: str) -> str:
        if n is None:
            n = ""
        n2 = re.sub(r"[^0-9A-Za-z_]", "_", str(n))
        if not n2:
            n2 = "attr"
        if not re.match(r"^[A-Za-z_]", n2):
            n2 = "f_" + n2
        return n2

    out = []
    seen = set()
    for n in names:
        base = sanitize_one(n)
        candidate = base
        i = 1
        while candidate in seen:
            candidate = f"{base}_{i}"
            i += 1
        seen.add(candidate)
        out.append(candidate)
    return out


def write_arff(
    path: str,
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: List[str],
    label_names: List[str],
    relation_name: str = "yeast_multilabel",
) -> None:
    """
    Write ARFF file with all features first (numeric), then labels as {0,1}.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Ensure proper dtypes
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)
    assert X.ndim == 2 and Y.ndim == 2
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of rows"

    feature_names = sanitize_names(feature_names)
    label_names = sanitize_names(label_names)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@RELATION {relation_name}\n\n")
        for name in feature_names:
            f.write(f"@ATTRIBUTE {name} NUMERIC\n")
        for name in label_names:
            f.write(f"@ATTRIBUTE {name} {{0,1}}\n")
        f.write("\n@DATA\n")
        # Write row-wise: features..., labels...
        for i in range(X.shape[0]):
            feats = ",".join(f"{v:.6g}" for v in X[i])
            labels = ",".join(str(int(v)) for v in Y[i])
            f.write(f"{feats},{labels}\n")


def write_mulan_xml(path: str, label_names: List[str]) -> None:
    """
    Write a minimal Mulan-compatible labels XML file (optional).
    Not required by scikit-multilearn's load_from_arff if label_count is provided,
    but useful for other tools.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    label_names = sanitize_names(label_names)
    xml = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<labels xmlns="http://mulan.sourceforge.net/labels">']
    for name in label_names:
        xml.append(f'    <label name="{name}" />')
    xml.append("</labels>\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml))


def load_yeast_openml() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the Yeast multi-label dataset from OpenML and return a DataFrame where:
    - All columns (features + labels) are present
    - The label column names are provided separately
    """
    print(f"Downloading Yeast (OpenML dataset id {OPENML_DATASET_ID}) ...")
    ds = openml.datasets.get_dataset(OPENML_DATASET_ID, download_all_files=True)
    df, _, _, _ = ds.get_data(dataset_format="dataframe", target=None)

    # default_target_attribute is a comma-separated list of label columns for multi-label datasets
    label_cols = [c.strip() for c in str(ds.default_target_attribute).split(",") if c.strip()]
    if not label_cols:
        raise RuntimeError("OpenML dataset did not expose multi-label targets as expected.")

    print(f"Identified {len(label_cols)} label columns from OpenML metadata.")
    return df, label_cols


def main():
    parser = argparse.ArgumentParser(description="Download Yeast dataset and save ARFF train/test splits.")
    parser.add_argument("--outdir", type=str, default="data/yeast", help="Output directory (default: data/yeast)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--write-xml", action="store_true", help="Also write yeast.xml with label names")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # 1) Load from OpenML
    df, label_cols = load_yeast_openml()

    # 2) Split into X (features) and Y (labels)
    # Ensure labels are {0,1}
    Y = df[label_cols].copy()
    for c in label_cols:
        # Cast truthy/Falsey to {0,1}
        Y[c] = Y[c].astype(int)
        # If values are not 0/1, coerce to 0/1
        Y[c] = (Y[c] > 0).astype(int)

    X = df.drop(columns=label_cols)
    # Ensure numeric features
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Replace any NaNs with column means (shouldn't happen on Yeast, but for safety)
    X = X.fillna(X.mean())

    X_np = X.to_numpy(dtype=np.float32)
    Y_np = Y.to_numpy(dtype=np.int32)

    n_samples, n_features = X_np.shape
    n_labels = Y_np.shape[1]
    print(f"Data shape: X={X_np.shape}, Y={Y_np.shape} (labels={n_labels})")

    # 3) Train/test split (random; OpenML does not ship a canonical split here)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_np, Y_np, test_size=args.test_size, random_state=args.seed, shuffle=True, stratify=None
    )
    print(f"Split: train={X_tr.shape[0]} rows, test={X_te.shape[0]} rows")

    # 4) Write ARFF files (features first, then labels)
    feature_names = list(X.columns)
    label_names = label_cols

    train_path = os.path.join(outdir, "yeast-train.arff")
    test_path = os.path.join(outdir, "yeast-test.arff")
    write_arff(train_path, X_tr, y_tr, feature_names, label_names, relation_name="yeast_multilabel_train")
    write_arff(test_path, X_te, y_te, feature_names, label_names, relation_name="yeast_multilabel_test")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {test_path}")

    # 5) Optionally write a minimal Mulan labels XML
    if args.write_xml:
        xml_path = os.path.join(outdir, "yeast.xml")
        write_mulan_xml(xml_path, label_names)
        print(f"Wrote: {xml_path}")

    print("\nDone. You can now load with:")
    print("  from skmultilearn.dataset import load_from_arff")
    print("  X_tr, y_tr, _, _ = load_from_arff('data/yeast/yeast-train.arff', label_count=%d, return_attribute_names=True)" % n_labels)
    print("  X_te, y_te, _, _ = load_from_arff('data/yeast/yeast-test.arff', label_count=%d, return_attribute_names=True)" % n_labels)


if __name__ == "__main__":
    main()

