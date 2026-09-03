#!/usr/bin/env python3
"""Evaluate any Fei-Ma-format dense HRTF prediction on subject 40."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.hutubs import load_hutubs_subject
from src.utils.hrtf_metrics import evaluate_hrtf_prediction, summarize_unknown_metrics


DEFAULT_REFERENCE = REPOSITORY_ROOT.parent / "PINN-for-HRTF-upsampling" / "40.mat"
FIELDS = (
    "frequency_index",
    "frequency_hz",
    "split",
    "directions",
    "paper_error_db",
    "complex_nmse_db",
    "magnitude_nmse_db",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--prediction-key", default="total_est")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject = load_hutubs_subject(args.reference)
    prediction_mat = sio.loadmat(args.prediction)
    if args.prediction_key not in prediction_mat:
        raise KeyError(
            f"{args.prediction} does not contain key {args.prediction_key!r}"
        )
    prediction = np.asarray(prediction_mat[args.prediction_key])
    rows = evaluate_hrtf_prediction(
        subject.total_hrtf,
        prediction,
        subject.frequencies_hz,
        subject.known_direction_mask,
    )
    summary = summarize_unknown_metrics(rows)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.prediction.expanduser().resolve().parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "reference": str(args.reference.expanduser().resolve()),
        "prediction": str(args.prediction.expanduser().resolve()),
        "prediction_key": args.prediction_key,
        "primary_metric": "paper_error_db",
        "primary_split": "unknown",
        "unknown_metrics": summary,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for row in rows:
        if row["split"] == "unknown":
            print(
                f"{float(row['frequency_hz']):8.1f} Hz  "
                f"paper={float(row['paper_error_db']):8.3f} dB  "
                f"complex_nmse={float(row['complex_nmse_db']):8.3f} dB  "
                f"magnitude_nmse={float(row['magnitude_nmse_db']):8.3f} dB"
            )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
