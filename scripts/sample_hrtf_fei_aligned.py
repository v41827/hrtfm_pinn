#!/usr/bin/env python3
"""Sample and evaluate the 28 independent Fei-aligned HRTF field flows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.hutubs import HUTUBSFieldDataset, load_hutubs_subject
from src.fm.functional_prior import HelmholtzGaussianField
from src.fm.hrtf_integrators import heun_integrate_independent_hrtf
from src.models.hrtf_flow import IndependentHRTFFieldFlow, IndependentHRTFFlowConfig
from src.utils.hrtf_metrics import evaluate_hrtf_prediction, summarize_unknown_metrics


DEFAULT_INPUT = REPOSITORY_ROOT.parent / "PINN-for-HRTF-upsampling" / "40.mat"
METRIC_FIELDS = (
    "sample",
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--integration-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.num_samples < 1 or args.integration_steps < 1:
        parser.error("sample count and integration steps must be positive")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_field_checkpoints(training_dir: Path) -> list[Path]:
    checkpoints = sorted((training_dir / "fields").glob("field_*.pt"))
    if len(checkpoints) != 28:
        raise RuntimeError(
            f"Expected 28 completed field checkpoints under {training_dir / 'fields'}, "
            f"found {len(checkpoints)}"
        )
    return checkpoints


def sample_one_dense_hrtf(
    checkpoints: list[Path],
    fields: HUTUBSFieldDataset,
    *,
    device: torch.device,
    integration_steps: int,
    generator: torch.Generator,
) -> np.ndarray:
    subject = fields.subject
    prediction = np.full_like(subject.total_hrtf, np.nan, dtype=np.float32)
    for field_index, checkpoint_path in enumerate(checkpoints):
        item = fields[field_index]
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        expected = {
            "field_index": field_index,
            "frequency_index": int(item["frequency_index"]),
            "frequency_hz": float(item["frequency_hz"]),
            "component": "real" if int(item["component"]) == 0 else "imaginary",
            "hemisphere": "positive_y"
            if int(item["hemisphere"]) == 0
            else "negative_y",
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(f"{checkpoint_path} has inconsistent {key}")
        if not np.isclose(float(checkpoint["radius_m"]), subject.radius_m):
            raise ValueError(f"{checkpoint_path} uses a different measurement radius")

        model = IndependentHRTFFieldFlow(
            IndependentHRTFFlowConfig(**checkpoint["model_config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        prior = HelmholtzGaussianField(**checkpoint["prior_config"]).to(device)
        prior.load_state_dict(checkpoint["prior"])

        observed_xyz_m = item["observed_xyz_m"].unsqueeze(0).to(device)
        observed_values = item["observed_values"].unsqueeze(0).to(device)
        query_xyz_m = item["collocation_xyz_m"].unsqueeze(0).to(device)
        frequency_hz = item["frequency_hz"].reshape(1).to(device)
        latent = prior.sample_latent(
            1,
            device=device,
            dtype=observed_values.dtype,
            generator=generator,
        )
        initial = prior(query_xyz_m / subject.radius_m, latent, frequency_hz)
        known_indices = item["observed_local_indices"].to(device)
        observed_source = initial[:, known_indices]
        final = heun_integrate_independent_hrtf(
            model,
            initial,
            query_xyz_m,
            steps=integration_steps,
            observed_indices=known_indices,
            observed_source=observed_source,
            observed_target=observed_values,
        )
        global_indices = item["total_global_indices"].numpy()
        prediction[
            int(item["frequency_index"]), int(item["component"]), global_indices
        ] = final.squeeze(0).cpu().numpy()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Sampling did not fill every HRTF field location")
    return prediction


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    subject = load_hutubs_subject(args.input)
    fields = HUTUBSFieldDataset(subject)
    training_dir = args.training_dir.expanduser().resolve()
    checkpoints = load_field_checkpoints(training_dir)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    samples = []
    metric_rows: list[dict[str, int | float | str]] = []
    for sample_index in range(args.num_samples):
        prediction = sample_one_dense_hrtf(
            checkpoints,
            fields,
            device=device,
            integration_steps=args.integration_steps,
            generator=generator,
        )
        samples.append(prediction)
        rows = evaluate_hrtf_prediction(
            subject.total_hrtf,
            prediction,
            subject.frequencies_hz,
            subject.known_direction_mask,
        )
        metric_rows.extend({"sample": sample_index, **row} for row in rows)
        summary = summarize_unknown_metrics(rows)
        print(
            f"sample={sample_index} unknown paper error="
            f"{summary['paper_error_db_mean_across_frequencies']:.3f} dB"
        )

    stacked_samples = np.stack(samples)
    mean_prediction = stacked_samples.mean(axis=0)
    mean_rows = evaluate_hrtf_prediction(
        subject.total_hrtf,
        mean_prediction,
        subject.frequencies_hz,
        subject.known_direction_mask,
    )
    metric_rows.extend({"sample": "mean", **row} for row in mean_rows)
    mean_summary = summarize_unknown_metrics(mean_rows)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_mat = sio.loadmat(subject.path)
    sio.savemat(
        output_dir / "prediction.mat",
        {
            "total_est": mean_prediction,
            "total_samples": stacked_samples,
            "total_hrtf": subject.total_hrtf,
            "total_coor": np.asarray(source_mat["total_coor"], dtype=np.float64),
            "train_coor": np.asarray(source_mat["train_coor"], dtype=np.float64),
            "freq_bins": subject.frequencies_hz[None, :],
        },
    )
    with (output_dir / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = {
        "protocol": "fei_ma_aligned_independent_field_flow",
        "training_dir": str(training_dir),
        "input": str(subject.path),
        "num_samples": args.num_samples,
        "integration_steps": args.integration_steps,
        "seed": args.seed,
        "primary_metric": "paper_error_db",
        "primary_split": "unknown",
        "mean_prediction_unknown_metrics": mean_summary,
        "known_values_are_hard_constrained": True,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(mean_summary, indent=2, sort_keys=True))
    print(f"Wrote {output_dir / 'prediction.mat'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
