#!/usr/bin/env python3
"""Sample and evaluate dense subject-40 HRTFs from a trained field flow."""

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
from src.fm.hrtf_integrators import heun_integrate_hrtf
from src.models.hrtf_flow import ConditionalHRTFFieldFlow, HRTFFlowConfig
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "runs"
            / "hutubs_subject40"
            / "hrtfm_pinn_seed2026"
            / "training"
            / "checkpoint_final.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "runs"
            / "hutubs_subject40"
            / "hrtfm_pinn_seed2026"
            / "evaluation"
            / "heun40_samples5"
        ),
    )
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


def sample_one_dense_hrtf(
    model: ConditionalHRTFFieldFlow,
    prior: HelmholtzGaussianField,
    fields: HUTUBSFieldDataset,
    *,
    device: torch.device,
    integration_steps: int,
    generator: torch.Generator,
) -> np.ndarray:
    subject = fields.subject
    prediction = np.full_like(subject.total_hrtf, np.nan, dtype=np.float32)
    model.eval()
    for field_index in range(len(fields)):
        item = fields[field_index]
        observed_xyz = item["observed_xyz_m"].unsqueeze(0).to(device)
        observed_values = item["observed_values"].unsqueeze(0).to(device)
        query_xyz = item["collocation_xyz_m"].unsqueeze(0).to(device)
        observed_unit_xyz = observed_xyz / subject.radius_m
        query_unit_xyz = query_xyz / subject.radius_m
        frequency = item["frequency_hz"].reshape(1).to(device)
        component = item["component"].reshape(1).to(device)
        hemisphere = item["hemisphere"].reshape(1).to(device)
        with torch.no_grad():
            context = model.encode_condition(
                observed_unit_xyz,
                observed_values,
                frequency,
                component,
                hemisphere,
            )
            latent = prior.sample_latent(
                1,
                device=device,
                dtype=observed_values.dtype,
                generator=generator,
            )
            initial = prior(query_unit_xyz, latent, frequency)
            known_indices = item["observed_local_indices"].to(device)
            observed_source = initial[:, known_indices]
            final = heun_integrate_hrtf(
                model,
                initial,
                query_unit_xyz,
                context,
                steps=integration_steps,
                observed_indices=known_indices,
                observed_source=observed_source,
                observed_target=observed_values,
            )
        frequency_index = int(item["frequency_index"])
        component_index = int(item["component"])
        global_indices = item["total_global_indices"].numpy()
        prediction[frequency_index, component_index, global_indices] = (
            final.squeeze(0).cpu().numpy()
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("Sampling did not fill every HRTF field location")
    return prediction


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)
    subject = load_hutubs_subject(args.input)
    fields = HUTUBSFieldDataset(subject)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if not np.allclose(checkpoint["frequencies_hz"], subject.frequencies_hz):
        raise ValueError("Checkpoint frequencies do not match the input MAT file")
    if not np.isclose(float(checkpoint["radius_m"]), subject.radius_m):
        raise ValueError("Checkpoint measurement radius does not match the input MAT file")

    model = ConditionalHRTFFieldFlow(
        HRTFFlowConfig(**checkpoint["model_config"])
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    prior = HelmholtzGaussianField(**checkpoint["prior_config"]).to(device)
    prior.load_state_dict(checkpoint["prior"])

    generator = torch.Generator(device=device).manual_seed(args.seed)
    samples = []
    metric_rows: list[dict[str, int | float | str]] = []
    for sample_index in range(args.num_samples):
        prediction = sample_one_dense_hrtf(
            model,
            prior,
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
    with (output_dir / "evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    metadata = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
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
