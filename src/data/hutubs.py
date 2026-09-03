"""HUTUBS subject data used by the Fei Ma HRTF interpolation example.

The ``40.mat`` fixture contains a *single-ear* complex HRTF.  Its direction
axis covers both positive- and negative-y hemispheres; those hemispheres must
not be confused with left and right ears.  Ma et al. train the two hemispheres
separately, so this loader exposes the same 7 frequencies x 2 complex
components x 2 hemispheres = 28 scalar fields.

Only ``train_hrtf`` is returned by :class:`HUTUBSFieldDataset`.  The dense
``total_hrtf`` target remains available on :class:`HUTUBSSubjectData` solely
for post-training evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


COMPONENT_NAMES = ("real", "imag")
HEMISPHERE_NAMES = ("positive_y", "negative_y")


def _coordinate_keys(xyz: np.ndarray, decimals: int = 10) -> list[tuple[float, ...]]:
    return [tuple(row) for row in np.round(np.asarray(xyz), decimals=decimals)]


def match_coordinate_indices(
    query_xyz: np.ndarray, reference_xyz: np.ndarray
) -> np.ndarray:
    """Map every query coordinate to an exactly matching reference row."""

    lookup = {key: index for index, key in enumerate(_coordinate_keys(reference_xyz))}
    if len(lookup) != len(reference_xyz):
        raise ValueError("Reference coordinates contain duplicates")
    try:
        indices = np.asarray(
            [lookup[key] for key in _coordinate_keys(query_xyz)], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(
            "A sparse HRTF coordinate is absent from the dense grid"
        ) from error
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Sparse coordinates did not map one-to-one to the dense grid")
    return indices


@dataclass(frozen=True)
class HUTUBSSubjectData:
    """Validated arrays from one Fei-Ma-format HUTUBS ``.mat`` file."""

    path: Path
    frequencies_hz: np.ndarray
    train_xyz_m: np.ndarray
    total_xyz_m: np.ndarray
    train_hrtf: np.ndarray
    total_hrtf: np.ndarray
    sparse_to_total: np.ndarray
    radius_m: float

    @property
    def known_direction_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.total_xyz_m), dtype=bool)
        mask[self.sparse_to_total] = True
        return mask


def load_hutubs_subject(path: str | Path) -> HUTUBSSubjectData:
    """Load and strictly validate the MAT schema used in Ma et al.'s example."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HUTUBS MAT file not found: {path}")
    raw = sio.loadmat(path)
    required = {
        "freq_bins",
        "train_coor",
        "total_coor",
        "train_hrtf",
        "total_hrtf",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"{path} is missing variables: {', '.join(missing)}")

    frequencies = np.ravel(raw["freq_bins"]).astype(np.float64)
    train_coordinates = np.asarray(raw["train_coor"], dtype=np.float64)
    total_coordinates = np.asarray(raw["total_coor"], dtype=np.float64)
    train_hrtf = np.asarray(raw["train_hrtf"], dtype=np.float64)
    total_hrtf = np.asarray(raw["total_hrtf"], dtype=np.float64)

    if train_coordinates.ndim != 2 or train_coordinates.shape[1] < 3:
        raise ValueError("train_coor must have shape [known_direction, >=3]")
    if total_coordinates.ndim != 2 or total_coordinates.shape[1] < 3:
        raise ValueError("total_coor must have shape [direction, >=3]")
    expected_train = (len(frequencies), 2, len(train_coordinates))
    expected_total = (len(frequencies), 2, len(total_coordinates))
    if train_hrtf.shape != expected_train:
        raise ValueError(
            f"train_hrtf has shape {train_hrtf.shape}, expected {expected_train}"
        )
    if total_hrtf.shape != expected_total:
        raise ValueError(
            f"total_hrtf has shape {total_hrtf.shape}, expected {expected_total}"
        )
    if not all(
        np.isfinite(array).all()
        for array in (
            frequencies,
            train_coordinates,
            total_coordinates,
            train_hrtf,
            total_hrtf,
        )
    ):
        raise ValueError("HUTUBS arrays contain NaN or infinity")
    if np.any(frequencies <= 0):
        raise ValueError("All frequency bins must be positive")

    train_xyz = train_coordinates[:, :3]
    total_xyz = total_coordinates[:, :3]
    train_radii = np.linalg.norm(train_xyz, axis=1)
    total_radii = np.linalg.norm(total_xyz, axis=1)
    radius = float(np.median(total_radii))
    if radius <= 0 or not np.allclose(total_radii, radius, rtol=1e-6, atol=1e-9):
        raise ValueError("Dense coordinates are not on one non-zero measurement sphere")
    if not np.allclose(train_radii, radius, rtol=1e-6, atol=1e-9):
        raise ValueError("Sparse and dense coordinates use different measurement radii")

    sparse_to_total = match_coordinate_indices(train_xyz, total_xyz)
    if not np.allclose(train_hrtf, total_hrtf[:, :, sparse_to_total], atol=1e-10):
        raise ValueError("train_hrtf values disagree with total_hrtf at matching coordinates")

    for name, xyz in (("sparse", train_xyz), ("dense", total_xyz)):
        positive = int(np.count_nonzero(xyz[:, 1] > 0))
        negative = int(np.count_nonzero(xyz[:, 1] < 0))
        if positive + negative != len(xyz) or positive != negative:
            raise ValueError(
                f"{name} coordinates must split evenly into non-zero y hemispheres"
            )

    return HUTUBSSubjectData(
        path=path,
        frequencies_hz=frequencies,
        train_xyz_m=train_xyz,
        total_xyz_m=total_xyz,
        train_hrtf=train_hrtf,
        total_hrtf=total_hrtf,
        sparse_to_total=sparse_to_total,
        radius_m=radius,
    )


class HUTUBSFieldDataset(Dataset[dict[str, torch.Tensor]]):
    """The 28 scalar fields in the subject-40 interpolation experiment.

    Each item contains 165 supervised HRTF measurements and the 630
    collocation coordinates in the corresponding y hemisphere.  It never
    returns held-out dense HRTF values.
    """

    def __init__(self, subject: HUTUBSSubjectData):
        self.subject = subject
        self._fields = [
            (frequency_index, component, hemisphere)
            for frequency_index in range(len(subject.frequencies_hz))
            for component in range(2)
            for hemisphere in range(2)
        ]

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        frequency_index, component, hemisphere = self._fields[index]
        positive_y = hemisphere == 0
        train_mask = (
            self.subject.train_xyz_m[:, 1] > 0
            if positive_y
            else self.subject.train_xyz_m[:, 1] < 0
        )
        total_mask = (
            self.subject.total_xyz_m[:, 1] > 0
            if positive_y
            else self.subject.total_xyz_m[:, 1] < 0
        )
        total_indices = np.flatnonzero(total_mask)
        total_global_to_local = {
            int(global_index): local_index
            for local_index, global_index in enumerate(total_indices)
        }
        observed_global = self.subject.sparse_to_total[train_mask]
        observed_local = np.asarray(
            [total_global_to_local[int(i)] for i in observed_global], dtype=np.int64
        )

        return {
            "observed_xyz_m": torch.from_numpy(
                self.subject.train_xyz_m[train_mask].astype(np.float32)
            ),
            "observed_values": torch.from_numpy(
                self.subject.train_hrtf[frequency_index, component, train_mask].astype(
                    np.float32
                )
            ),
            "collocation_xyz_m": torch.from_numpy(
                self.subject.total_xyz_m[total_mask].astype(np.float32)
            ),
            "observed_local_indices": torch.from_numpy(observed_local),
            "total_global_indices": torch.from_numpy(total_indices.astype(np.int64)),
            "frequency_hz": torch.tensor(
                self.subject.frequencies_hz[frequency_index], dtype=torch.float32
            ),
            "frequency_index": torch.tensor(frequency_index, dtype=torch.long),
            "component": torch.tensor(component, dtype=torch.long),
            "hemisphere": torch.tensor(hemisphere, dtype=torch.long),
        }
