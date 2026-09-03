#!/usr/bin/env python3
"""Train a subject-40 HUTUBS physics-informed conditional field flow.

This pilot follows Fei Ma's interpolation protocol: ``train_hrtf`` supplies
the only supervised targets, while ``total_coor`` supplies unlabeled physics
collocation points.  ``total_hrtf`` is never used by this training script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.hutubs import HUTUBSFieldDataset, load_hutubs_subject
from src.fm.functional_prior import HelmholtzGaussianField
from src.fm.hrtf_integrators import euler_unroll_to_endpoint
from src.fm.hrtf_objectives import (
    clean_endpoint_from_velocity,
    constant_field_velocity,
    linear_field_path,
)
from src.models.hrtf_flow import ConditionalHRTFFieldFlow, HRTFFlowConfig
from src.physics.helmholtz import helmholtz_loss
from src.utils.seed import set_seed


DEFAULT_INPUT = REPOSITORY_ROOT.parent / "PINN-for-HRTF-upsampling" / "40.mat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "runs"
            / "hutubs_subject40"
            / "hrtfm_pinn_seed2026"
            / "training"
        ),
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--resume", type=Path)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--context-dim", type=int, default=128)
    parser.add_argument("--observation-width", type=int, default=96)
    parser.add_argument("--coordinate-bands", type=int, default=4)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--frequency-scale-hz", type=float, default=15000.0)

    parser.add_argument("--prior-modes", type=int, default=16)
    parser.add_argument("--prior-seed", type=int, default=1729)

    parser.add_argument("--physics-weight", type=float, default=0.1)
    parser.add_argument("--physics-warmup-steps", type=int, default=500)
    parser.add_argument("--physics-ramp-steps", type=int, default=2000)
    parser.add_argument("--physics-every", type=int, default=1)
    parser.add_argument("--physics-points", type=int, default=64)
    parser.add_argument("--physics-unroll-steps", type=int, default=4)
    parser.add_argument("--speed-of-sound", type=float, default=343.0)
    args = parser.parse_args()

    positive_integer_names = (
        "steps",
        "batch_size",
        "log_every",
        "save_every",
        "width",
        "depth",
        "context_dim",
        "observation_width",
        "time_dim",
        "prior_modes",
        "physics_every",
        "physics_points",
        "physics_unroll_steps",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.coordinate_bands < 0 or args.num_workers < 0:
        parser.error("coordinate bands and worker count cannot be negative")
    if args.physics_weight < 0 or args.physics_warmup_steps < 0:
        parser.error("physics weight and warmup cannot be negative")
    if args.physics_ramp_steps < 1:
        parser.error("--physics-ramp-steps must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        parser.error("learning rate must be positive; decay and clipping cannot be negative")
    if args.frequency_scale_hz <= 0 or args.speed_of_sound <= 0:
        parser.error("frequency scale and speed of sound must be positive")
    return args


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def physics_weight_at_step(args: argparse.Namespace, step: int) -> float:
    if args.physics_weight == 0 or step <= args.physics_warmup_steps:
        return 0.0
    progress = min(
        1.0, (step - args.physics_warmup_steps) / float(args.physics_ramp_steps)
    )
    return args.physics_weight * progress


def random_collocation_subset(
    coordinates: torch.Tensor, points: int
) -> torch.Tensor:
    batch_size, available, dimensions = coordinates.shape
    if points >= available:
        return coordinates
    indices = torch.stack(
        [
            torch.randperm(available, device=coordinates.device)[:points]
            for _ in range(batch_size)
        ]
    )
    return torch.gather(coordinates, 1, indices[..., None].expand(-1, -1, dimensions))


def make_model_config(args: argparse.Namespace) -> HRTFFlowConfig:
    return HRTFFlowConfig(
        width=args.width,
        depth=args.depth,
        context_dim=args.context_dim,
        observation_width=args.observation_width,
        coordinate_bands=args.coordinate_bands,
        time_dim=args.time_dim,
        frequency_scale_hz=args.frequency_scale_hz,
    )


def make_prior(args: argparse.Namespace, radius_m: float) -> HelmholtzGaussianField:
    return HelmholtzGaussianField(
        modes=args.prior_modes,
        radius_m=radius_m,
        speed_of_sound_m_s=args.speed_of_sound,
        seed=args.prior_seed,
    )


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    values = vars(args).copy()
    for key, value in values.items():
        if isinstance(value, Path):
            values[key] = str(value.expanduser().resolve())
    return values


def save_checkpoint(
    path: Path,
    *,
    model: ConditionalHRTFFieldFlow,
    prior: HelmholtzGaussianField,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    radius_m: float,
    frequencies_hz: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "step": step,
            "model": model.state_dict(),
            "model_config": model.config.to_dict(),
            "prior": prior.state_dict(),
            "prior_config": prior.config_dict(),
            "optimizer": optimizer.state_dict(),
            "train_config": serializable_args(args),
            "radius_m": radius_m,
            "frequencies_hz": frequencies_hz,
            "protocol": {
                "supervised_values": "train_hrtf only",
                "physics_coordinates": "total_coor",
                "held_out_values": "total_hrtf",
                "ear": "single ear supplied by Fei Ma 40.mat",
            },
        },
        temporary,
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    subject = load_hutubs_subject(args.input)
    fields = HUTUBSFieldDataset(subject)
    loader = DataLoader(
        fields,
        batch_size=min(args.batch_size, len(fields)),
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    iterator = iter(loader)

    model = ConditionalHRTFFieldFlow(make_model_config(args)).to(device)
    prior = make_prior(args, subject.radius_m).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    starting_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        prior.load_state_dict(checkpoint["prior"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        starting_step = int(checkpoint["step"])

    run_config = {
        "arguments": serializable_args(args),
        "device": str(device),
        "subject_file": str(subject.path),
        "radius_m": subject.radius_m,
        "frequencies_hz": subject.frequencies_hz.tolist(),
        "field_tasks": len(fields),
        "known_directions": len(subject.train_xyz_m),
        "held_out_directions": len(subject.total_xyz_m) - len(subject.train_xyz_m),
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = output_dir / "training.jsonl"

    print(json.dumps(run_config, indent=2, sort_keys=True))
    print("Training uses train_hrtf only; dense total_hrtf is held out.")
    model.train()
    started = time.monotonic()
    for step in range(starting_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        observed_unit_xyz = batch["observed_xyz_m"] / subject.radius_m
        observed_values = batch["observed_values"]
        frequency_hz = batch["frequency_hz"]
        context = model.encode_condition(
            observed_unit_xyz,
            observed_values,
            frequency_hz,
            batch["component"],
            batch["hemisphere"],
        )
        latent = prior.sample_latent(
            observed_values.shape[0], device=device, dtype=observed_values.dtype
        )
        source = prior(observed_unit_xyz, latent, frequency_hz)
        time_batch = torch.rand((observed_values.shape[0],), device=device).clamp_(
            1e-4, 1.0 - 1e-4
        )
        state = linear_field_path(source, observed_values, time_batch)
        target_velocity = constant_field_velocity(source, observed_values)
        predicted_velocity = model.velocity_with_context(
            state, time_batch, observed_unit_xyz, context
        )
        flow_loss = F.mse_loss(predicted_velocity, target_velocity)
        endpoint_estimate = clean_endpoint_from_velocity(
            state, predicted_velocity, time_batch
        )
        measured_endpoint_loss = F.mse_loss(endpoint_estimate, observed_values)

        active_physics_weight = physics_weight_at_step(args, step)
        physics_loss = torch.zeros((), device=device)
        residual_rms = torch.zeros((), device=device)
        if active_physics_weight > 0 and step % args.physics_every == 0:
            collocation = random_collocation_subset(
                batch["collocation_xyz_m"], args.physics_points
            )
            collocation_unit_xyz = (collocation / subject.radius_m).detach()
            collocation_unit_xyz.requires_grad_(True)
            physics_source = prior(collocation_unit_xyz, latent, frequency_hz)
            clean_endpoint = euler_unroll_to_endpoint(
                model,
                physics_source,
                collocation_unit_xyz,
                context,
                steps=args.physics_unroll_steps,
            )
            physics_loss, residual = helmholtz_loss(
                clean_endpoint,
                collocation_unit_xyz,
                frequency_hz,
                radius_m=subject.radius_m,
                speed_of_sound_m_s=args.speed_of_sound,
            )
            residual_rms = residual.detach().square().mean().sqrt()

        total_loss = flow_loss + active_physics_weight * physics_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite loss at step {step}: FM={flow_loss.item()}, "
                f"physics={physics_loss.item()}"
            )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.grad_clip > 0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
        else:
            gradient_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.monotonic() - started
            record = {
                "step": step,
                "total_loss": float(total_loss.detach()),
                "flow_loss": float(flow_loss.detach()),
                "measured_endpoint_loss": float(measured_endpoint_loss.detach()),
                "physics_loss": float(physics_loss.detach()),
                "physics_residual_rms": float(residual_rms),
                "physics_weight": active_physics_weight,
                "gradient_norm": float(gradient_norm),
                "steps_per_second": (step - starting_step) / max(elapsed, 1e-9),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(
                f"[{step:6d}/{args.steps}] total={record['total_loss']:.6g} "
                f"fm={record['flow_loss']:.6g} physics={record['physics_loss']:.6g} "
                f"w={active_physics_weight:.4g} {record['steps_per_second']:.2f} step/s"
            )

        if step % args.save_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_{step}.pt",
                model=model,
                prior=prior,
                optimizer=optimizer,
                step=step,
                args=args,
                radius_m=subject.radius_m,
                frequencies_hz=subject.frequencies_hz.tolist(),
            )

    save_checkpoint(
        output_dir / "checkpoint_final.pt",
        model=model,
        prior=prior,
        optimizer=optimizer,
        step=args.steps,
        args=args,
        radius_m=subject.radius_m,
        frequencies_hz=subject.frequencies_hz.tolist(),
    )
    print(f"Wrote {output_dir / 'checkpoint_final.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
