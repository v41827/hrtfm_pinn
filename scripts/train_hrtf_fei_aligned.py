#!/usr/bin/env python3
"""Train 28 independent HRTF flows under a Fei-Ma-aligned protocol.

This comparison runner deliberately matches Ma et al.'s subject-40 training
structure wherever the different objective permits: one model per scalar
field, three tanh hidden layers, frequency-dependent width, Adam at 1e-3,
all 630 hemisphere coordinates in every physics loss, equal FM/physics loss
weights, up to five attempts, and training-only attempt selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.hutubs import HUTUBSFieldDataset, load_hutubs_subject
from src.fm.functional_prior import HelmholtzGaussianField
from src.fm.hrtf_integrators import (
    euler_unroll_independent_to_endpoint,
    heun_integrate_independent_hrtf,
)
from src.fm.hrtf_objectives import (
    clean_endpoint_from_velocity,
    constant_field_velocity,
    linear_field_path,
)
from src.models.hrtf_flow import IndependentHRTFFieldFlow, IndependentHRTFFlowConfig
from src.physics.helmholtz import physical_helmholtz_loss
from src.utils.seed import set_seed


DEFAULT_INPUT = REPOSITORY_ROOT.parent / "PINN-for-HRTF-upsampling" / "40.mat"
PROGRESS_FIELDS = (
    "field_index",
    "frequency_index",
    "frequency_hz",
    "component",
    "hemisphere",
    "attempt",
    "step",
    "width",
    "total_loss",
    "fm_loss",
    "measured_endpoint_loss",
    "physics_loss",
    "physics_residual_rms",
    "elapsed_seconds",
)
ATTEMPT_FIELDS = (
    "field_index",
    "frequency_index",
    "frequency_hz",
    "component",
    "hemisphere",
    "attempt",
    "seed",
    "steps",
    "width",
    "selection_data_loss_db",
    "duration_seconds",
    "selected",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT
        / "runs"
        / "hutubs_subject40"
        / "hrtfm_pinn_fei_aligned_seed2026"
        / "training",
    )
    parser.add_argument("--steps-per-attempt", type=int, default=1_000_000)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--physics-unroll-steps", type=int, default=4)
    parser.add_argument("--prior-modes", type=int, default=16)
    parser.add_argument("--prior-seed", type=int, default=1729)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    parser.add_argument("--loss-threshold-db", type=float, default=-29.0)
    parser.add_argument("--selection-samples", type=int, default=5)
    parser.add_argument("--selection-integration-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--save-every", type=int, default=10_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-fields",
        type=int,
        default=28,
        help="Limit fields only for smoke testing; paper runs must use 28.",
    )
    args = parser.parse_args()
    positive = (
        "steps_per_attempt",
        "attempts",
        "depth",
        "physics_unroll_steps",
        "prior_modes",
        "selection_samples",
        "selection_integration_steps",
        "log_every",
        "save_every",
        "max_fields",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_fields > 28:
        parser.error("--max-fields cannot exceed the 28 subject-40 scalar fields")
    if args.learning_rate <= 0 or args.physics_weight < 0:
        parser.error("learning rate must be positive and physics weight non-negative")
    if args.speed_of_sound <= 0:
        parser.error("speed of sound must be positive")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def network_width(frequency_hz: float) -> int:
    """Exactly reproduce the frequency-width rule in Fei Ma's runner."""

    if frequency_hz < 3000.0:
        return int(math.ceil(frequency_hz / 500.0))
    if frequency_hz > 6000.0:
        return int(math.ceil(frequency_hz / 1000.0))
    return 6


def field_metadata(
    field_index: int, item: dict[str, torch.Tensor]
) -> dict[str, object]:
    return {
        "field_index": field_index,
        "frequency_index": int(item["frequency_index"]),
        "frequency_hz": float(item["frequency_hz"]),
        "component": "real" if int(item["component"]) == 0 else "imaginary",
        "hemisphere": "positive_y" if int(item["hemisphere"]) == 0 else "negative_y",
    }


def field_checkpoint_path(
    fields_dir: Path, field_index: int, item: dict[str, torch.Tensor]
) -> Path:
    component = "real" if int(item["component"]) == 0 else "imag"
    hemisphere = "posy" if int(item["hemisphere"]) == 0 else "negy"
    frequency = int(round(float(item["frequency_hz"])))
    name = f"field_{field_index:02d}_f{frequency}_{component}_{hemisphere}.pt"
    return fields_dir / name


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def append_csv(path: Path, fields: tuple[str, ...], row: dict[str, object]) -> None:
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value.expanduser().resolve())
    return result


def resume_signature(args: argparse.Namespace, subject_path: Path) -> dict[str, object]:
    return {
        "subject_file": str(subject_path),
        "steps_per_attempt": args.steps_per_attempt,
        "attempts": args.attempts,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "physics_weight": args.physics_weight,
        "physics_unroll_steps": args.physics_unroll_steps,
        "prior_modes": args.prior_modes,
        "prior_seed": args.prior_seed,
        "speed_of_sound": args.speed_of_sound,
        "loss_threshold_db": args.loss_threshold_db,
        "selection_samples": args.selection_samples,
        "selection_integration_steps": args.selection_integration_steps,
        "seed": args.seed,
        "max_fields": args.max_fields,
    }


@torch.no_grad()
def measured_reconstruction_db(
    model: IndependentHRTFFieldFlow,
    prior: HelmholtzGaussianField,
    observed_xyz_m: torch.Tensor,
    observed_values: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    samples: int,
    integration_steps: int,
    seed: int,
    radius_m: float,
) -> float:
    """Training-only selection score analogous to Ma's normalized data MSE."""

    generator = torch.Generator(device=observed_values.device).manual_seed(seed)
    predictions = []
    for _ in range(samples):
        latent = prior.sample_latent(
            1,
            device=observed_values.device,
            dtype=observed_values.dtype,
            generator=generator,
        )
        initial = prior(observed_xyz_m / radius_m, latent, frequency_hz)
        predictions.append(
            heun_integrate_independent_hrtf(
                model,
                initial,
                observed_xyz_m,
                steps=integration_steps,
            )
        )
    prediction = torch.stack(predictions).mean(dim=0)
    mse = F.mse_loss(prediction, observed_values)
    signal = observed_values.square().mean().clamp_min(torch.finfo(mse.dtype).tiny)
    return float(10.0 * torch.log10(mse / signal))


def save_resume_state(
    path: Path,
    *,
    signature: dict[str, object],
    field_index: int,
    attempt: int,
    step: int,
    model: IndependentHRTFFieldFlow | None,
    optimizer: torch.optim.Optimizer | None,
    best_score_db: float,
    best_state: dict[str, torch.Tensor] | None,
    attempt_rows: list[dict[str, object]],
) -> None:
    state: dict[str, object] = {
        "format_version": 1,
        "signature": signature,
        "field_index": field_index,
        "attempt": attempt,
        "step": step,
        "best_score_db": best_score_db,
        "best_state": best_state,
        "attempt_rows": attempt_rows,
        "cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if model is not None and optimizer is not None:
        state["model"] = model.state_dict()
        state["model_config"] = model.config.to_dict()
        state["optimizer"] = optimizer.state_dict()
    atomic_torch_save(state, path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    fields_dir = output_dir / "fields"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields_dir.mkdir(parents=True, exist_ok=True)

    subject = load_hutubs_subject(args.input)
    fields = HUTUBSFieldDataset(subject)
    signature = resume_signature(args, subject.path)
    prior = HelmholtzGaussianField(
        modes=args.prior_modes,
        radius_m=subject.radius_m,
        speed_of_sound_m_s=args.speed_of_sound,
        seed=args.prior_seed,
    ).to(device)

    run_config = {
        "status": "running",
        "protocol": "fei_ma_aligned_independent_field_flow",
        "arguments": serializable_args(args),
        "device": str(device),
        "subject_file": str(subject.path),
        "radius_m": subject.radius_m,
        "known_directions": len(subject.train_xyz_m),
        "held_out_directions": len(subject.total_xyz_m) - len(subject.train_xyz_m),
        "field_tasks": args.max_fields,
        "comparison_alignment": {
            "independent_scalar_models": True,
            "hidden_layers": args.depth,
            "frequency_dependent_width": True,
            "optimizer": "Adam",
            "all_hemisphere_physics_coordinates_each_step": True,
            "physics_warmup": False,
            "attempt_selection_uses_only_measured_values": True,
            "held_out_total_hrtf_used_for_training": False,
        },
    }
    config_path = output_dir / "config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    resume_path = (
        args.resume.expanduser().resolve() if args.resume else output_dir / "resume.pt"
    )
    resume: dict[str, object] | None = None
    if args.resume:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume.get("signature") != signature:
            raise ValueError("Resume checkpoint configuration does not match this run")

    progress_path = output_dir / "training_progress.csv"
    attempts_path = output_dir / "training_attempts.csv"
    whole_run_started = time.monotonic()
    for field_index in range(args.max_fields):
        item = fields[field_index]
        final_path = field_checkpoint_path(fields_dir, field_index, item)
        if final_path.exists():
            print(
                f"field={field_index:02d} already complete; "
                f"skipping {final_path.name}"
            )
            continue

        metadata = field_metadata(field_index, item)
        frequency_hz = item["frequency_hz"].reshape(1).to(device)
        width = network_width(float(frequency_hz.item()))
        observed_xyz_m = item["observed_xyz_m"].unsqueeze(0).to(device)
        observed_values = item["observed_values"].unsqueeze(0).to(device)
        collocation_base = item["collocation_xyz_m"].unsqueeze(0).to(device)
        field_started = time.monotonic()

        same_resume_field = (
            resume is not None and int(resume["field_index"]) == field_index
        )
        starting_attempt = int(resume["attempt"]) if same_resume_field else 0
        best_score_db = (
            float(resume["best_score_db"]) if same_resume_field else float("inf")
        )
        best_state = resume.get("best_state") if same_resume_field else None
        attempt_rows = list(resume.get("attempt_rows", [])) if same_resume_field else []

        for attempt in range(starting_attempt, args.attempts):
            attempt_seed = args.seed + int(item["frequency_index"]) * 100 + int(
                item["component"]
            ) * 10 + int(item["hemisphere"]) * 5 + attempt
            model = IndependentHRTFFieldFlow(
                IndependentHRTFFlowConfig(width=width, depth=args.depth)
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            starting_step = 0
            if same_resume_field and attempt == starting_attempt and "model" in resume:
                if resume["model_config"] != model.config.to_dict():
                    raise ValueError("Resume model configuration does not match")
                model.load_state_dict(resume["model"])
                optimizer.load_state_dict(resume["optimizer"])
                starting_step = int(resume["step"])
                torch.set_rng_state(resume["cpu_rng_state"].cpu())
                if device.type == "cuda" and "cuda_rng_state_all" in resume:
                    torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
                print(
                    f"resuming field={field_index:02d} attempt={attempt + 1} "
                    f"at step={starting_step}"
                )
            else:
                set_seed(attempt_seed)

            model.train()
            attempt_started = time.monotonic()
            for step in range(starting_step + 1, args.steps_per_attempt + 1):
                latent = prior.sample_latent(
                    1, device=device, dtype=observed_values.dtype
                )
                source = prior(observed_xyz_m / subject.radius_m, latent, frequency_hz)
                time_batch = torch.rand((1,), device=device).clamp_(1e-4, 1.0 - 1e-4)
                state = linear_field_path(source, observed_values, time_batch)
                target_velocity = constant_field_velocity(source, observed_values)
                predicted_velocity = model(state, time_batch, observed_xyz_m)
                fm_loss = F.mse_loss(predicted_velocity, target_velocity)
                endpoint_estimate = clean_endpoint_from_velocity(
                    state, predicted_velocity, time_batch
                )
                measured_endpoint_loss = F.mse_loss(endpoint_estimate, observed_values)

                collocation_xyz_m = (
                    collocation_base.detach().clone().requires_grad_(True)
                )
                physics_source = prior(
                    collocation_xyz_m / subject.radius_m, latent, frequency_hz
                )
                clean_endpoint = euler_unroll_independent_to_endpoint(
                    model,
                    physics_source,
                    collocation_xyz_m,
                    steps=args.physics_unroll_steps,
                )
                physics_loss, residual = physical_helmholtz_loss(
                    clean_endpoint,
                    collocation_xyz_m,
                    frequency_hz,
                    speed_of_sound_m_s=args.speed_of_sound,
                )
                total_loss = fm_loss + args.physics_weight * physics_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at field {field_index}, "
                        f"attempt {attempt}, step {step}"
                    )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

                should_log = (
                    step == 1
                    or step % args.log_every == 0
                    or step == args.steps_per_attempt
                )
                if should_log:
                    row = {
                        **metadata,
                        "attempt": attempt + 1,
                        "step": step,
                        "width": width,
                        "total_loss": float(total_loss.detach()),
                        "fm_loss": float(fm_loss.detach()),
                        "measured_endpoint_loss": float(measured_endpoint_loss.detach()),
                        "physics_loss": float(physics_loss.detach()),
                        "physics_residual_rms": float(
                            residual.detach().square().mean().sqrt()
                        ),
                        "elapsed_seconds": time.monotonic() - attempt_started,
                    }
                    append_csv(progress_path, PROGRESS_FIELDS, row)
                    print(
                        f"field={field_index:02d}/27 f={float(frequency_hz.item()):.1f} "
                        f"{metadata['component']}/{metadata['hemisphere']} "
                        f"attempt={attempt + 1}/{args.attempts} "
                        f"step={step}/{args.steps_per_attempt} total={row['total_loss']:.6g} "
                        f"fm={row['fm_loss']:.6g} physics={row['physics_loss']:.6g}"
                    )
                if step % args.save_every == 0 or step == args.steps_per_attempt:
                    save_resume_state(
                        resume_path,
                        signature=signature,
                        field_index=field_index,
                        attempt=attempt,
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        best_score_db=best_score_db,
                        best_state=best_state,
                        attempt_rows=attempt_rows,
                    )

            model.eval()
            score_db = measured_reconstruction_db(
                model,
                prior,
                observed_xyz_m,
                observed_values,
                frequency_hz,
                samples=args.selection_samples,
                integration_steps=args.selection_integration_steps,
                seed=attempt_seed + 100_000,
                radius_m=subject.radius_m,
            )
            attempt_row = {
                **metadata,
                "attempt": attempt + 1,
                "seed": attempt_seed,
                "steps": args.steps_per_attempt,
                "width": width,
                "selection_data_loss_db": score_db,
                "duration_seconds": time.monotonic() - attempt_started,
                "selected": False,
            }
            attempt_rows.append(attempt_row)
            if score_db < best_score_db:
                best_score_db = score_db
                best_state = cpu_state_dict(model)
            print(
                f"field={field_index:02d} attempt={attempt + 1} "
                f"selection_data_loss_db={score_db:.3f} best={best_score_db:.3f}"
            )

            next_attempt = attempt + 1
            save_resume_state(
                resume_path,
                signature=signature,
                field_index=field_index,
                attempt=next_attempt,
                step=0,
                model=None,
                optimizer=None,
                best_score_db=best_score_db,
                best_state=best_state,
                attempt_rows=attempt_rows,
            )
            same_resume_field = False
            if best_score_db < args.loss_threshold_db:
                break

        if best_state is None:
            raise RuntimeError(f"No trained model was produced for field {field_index}")
        selected_index = int(
            np.argmin([float(row["selection_data_loss_db"]) for row in attempt_rows])
        )
        attempt_rows[selected_index]["selected"] = True
        for row in attempt_rows:
            append_csv(attempts_path, ATTEMPT_FIELDS, row)

        atomic_torch_save(
            {
                "format_version": 1,
                "protocol": "fei_ma_aligned_independent_field_flow",
                **metadata,
                "model": best_state,
                "model_config": IndependentHRTFFlowConfig(
                    width=width, depth=args.depth
                ).to_dict(),
                "prior": prior.state_dict(),
                "prior_config": prior.config_dict(),
                "best_selection_data_loss_db": best_score_db,
                "attempts_completed": len(attempt_rows),
                "radius_m": subject.radius_m,
                "subject_file": str(subject.path),
                "training_config": serializable_args(args),
            },
            final_path,
        )
        print(
            f"completed field={field_index:02d} best={best_score_db:.3f} dB "
            f"duration={time.monotonic() - field_started:.1f}s -> {final_path.name}"
        )
        resume = None
        save_resume_state(
            resume_path,
            signature=signature,
            field_index=field_index + 1,
            attempt=0,
            step=0,
            model=None,
            optimizer=None,
            best_score_db=float("inf"),
            best_state=None,
            attempt_rows=[],
        )

    run_config["status"] = "completed"
    run_config["duration_seconds"] = time.monotonic() - whole_run_started
    run_config["completed_field_checkpoints"] = len(list(fields_dir.glob("field_*.pt")))
    config_path.write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "TRAINING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(f"Completed {args.max_fields} independent field flows in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
