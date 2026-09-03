"""Evaluation metrics for complex HRTF interpolation."""

from __future__ import annotations

import math

import numpy as np


def _db_ratio(numerator: float, denominator: float, factor: float) -> float:
    tiny = np.finfo(np.float64).tiny
    return float(factor * math.log10(max(numerator, tiny) / max(denominator, tiny)))


def evaluate_hrtf_prediction(
    reference: np.ndarray,
    prediction: np.ndarray,
    frequencies_hz: np.ndarray,
    known_mask: np.ndarray,
) -> list[dict[str, int | float | str]]:
    """Evaluate the Ma-paper error and conventional complex/magnitude NMSEs."""

    reference = np.asarray(reference)
    prediction = np.asarray(prediction)
    frequencies = np.ravel(frequencies_hz)
    known_mask = np.asarray(known_mask, dtype=bool)
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes must match")
    if reference.ndim != 3 or reference.shape[1] != 2:
        raise ValueError("HRTFs must have shape [frequency, real_imag, direction]")
    if reference.shape[0] != len(frequencies):
        raise ValueError("frequency count does not match HRTF data")
    if reference.shape[-1] != len(known_mask):
        raise ValueError("known mask does not match direction count")

    rows: list[dict[str, int | float | str]] = []
    for frequency_index, frequency_hz in enumerate(frequencies):
        truth = reference[frequency_index, 0] + 1j * reference[frequency_index, 1]
        estimate = prediction[frequency_index, 0] + 1j * prediction[frequency_index, 1]
        for split, mask in (
            ("unknown", ~known_mask),
            ("known", known_mask),
            ("all", np.ones_like(known_mask)),
        ):
            truth_split = truth[mask]
            estimate_split = estimate[mask]
            complex_error = np.abs(truth_split - estimate_split)
            magnitude_error = np.abs(truth_split) - np.abs(estimate_split)
            truth_absolute = np.abs(truth_split)
            truth_energy = float(np.square(truth_absolute).sum())
            rows.append(
                {
                    "frequency_index": frequency_index,
                    "frequency_hz": float(frequency_hz),
                    "split": split,
                    "directions": int(mask.sum()),
                    # Equation (25) in Ma et al.: normalized absolute complex error.
                    "paper_error_db": _db_ratio(
                        float(complex_error.sum()), float(truth_absolute.sum()), 20.0
                    ),
                    "complex_nmse_db": _db_ratio(
                        float(np.square(complex_error).sum()), truth_energy, 10.0
                    ),
                    "magnitude_nmse_db": _db_ratio(
                        float(np.square(magnitude_error).sum()), truth_energy, 10.0
                    ),
                }
            )
    return rows


def summarize_unknown_metrics(
    rows: list[dict[str, int | float | str]],
) -> dict[str, float | int]:
    unknown = [row for row in rows if row["split"] == "unknown"]
    if not unknown:
        raise ValueError("No unknown-direction metric rows were provided")
    return {
        "unknown_direction_count": int(unknown[0]["directions"]),
        "paper_error_db_mean_across_frequencies": float(
            np.mean([float(row["paper_error_db"]) for row in unknown])
        ),
        "complex_nmse_db_mean_across_frequencies": float(
            np.mean([float(row["complex_nmse_db"]) for row in unknown])
        ),
        "magnitude_nmse_db_mean_across_frequencies": float(
            np.mean([float(row["magnitude_nmse_db"]) for row in unknown])
        ),
    }
