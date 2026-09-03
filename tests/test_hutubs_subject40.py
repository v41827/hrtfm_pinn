from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.data.hutubs import HUTUBSFieldDataset, load_hutubs_subject


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_40 = REPOSITORY_ROOT.parent / "PINN-for-HRTF-upsampling" / "40.mat"


@unittest.skipUnless(SUBJECT_40.is_file(), "Fei Ma subject-40 MAT file is unavailable")
class HUTUBSSubject40Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = load_hutubs_subject(SUBJECT_40)
        cls.fields = HUTUBSFieldDataset(cls.subject)

    def test_protocol_shapes_and_holdout(self) -> None:
        self.assertEqual(self.subject.train_hrtf.shape, (7, 2, 330))
        self.assertEqual(self.subject.total_hrtf.shape, (7, 2, 1260))
        self.assertEqual(int(self.subject.known_direction_mask.sum()), 330)
        self.assertEqual(int((~self.subject.known_direction_mask).sum()), 930)
        self.assertEqual(len(self.fields), 28)

    def test_field_contains_no_dense_target(self) -> None:
        field = self.fields[0]
        self.assertNotIn("total_hrtf", field)
        self.assertEqual(field["observed_xyz_m"].shape, (165, 3))
        self.assertEqual(field["observed_values"].shape, (165,))
        self.assertEqual(field["collocation_xyz_m"].shape, (630, 3))
        reconstructed_observed_xyz = field["collocation_xyz_m"][
            field["observed_local_indices"]
        ]
        torch.testing.assert_close(reconstructed_observed_xyz, field["observed_xyz_m"])


if __name__ == "__main__":
    unittest.main()
