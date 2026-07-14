import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets import DatasetForRADRec


class CouplingEstimationTest(unittest.TestCase):
    def test_single_interest_group_has_zero_entropy_and_coupling(self):
        embeddings = torch.tensor(
            [[[1.0, 0.0], [0.98, 0.1], [0.96, 0.12]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.long)

        entropy, coupling = DatasetForRADRec._interest_structure_from_embeddings(
            embeddings=embeddings,
            item_seq_len=lengths,
            theta=0.5,
        )

        self.assertAlmostEqual(float(entropy[0]), 0.0, places=6)
        self.assertAlmostEqual(float(coupling[0]), 0.0, places=6)

    def test_coupling_uses_weighted_centroid_similarity_between_components(self):
        embeddings = torch.tensor(
            [[[1.0, 0.0], [0.96, 0.1], [-1.0, 0.0]]],
            dtype=torch.float32,
        )
        lengths = torch.tensor([3], dtype=torch.long)

        entropy, coupling = DatasetForRADRec._interest_structure_from_embeddings(
            embeddings=embeddings,
            item_seq_len=lengths,
            theta=0.9,
        )

        expected_entropy = -(
            (2 / 3) * np.log2(2 / 3) + (1 / 3) * np.log2(1 / 3)
        )
        centroid_a = torch.tensor([0.98, 0.05], dtype=torch.float32)
        centroid_b = torch.tensor([-1.0, 0.0], dtype=torch.float32)
        expected_coupling = torch.nn.functional.cosine_similarity(
            centroid_a.view(1, -1),
            centroid_b.view(1, -1),
        ).item()

        self.assertAlmostEqual(float(entropy[0]), expected_entropy, places=6)
        self.assertAlmostEqual(float(coupling[0]), expected_coupling, places=6)

    def test_partition_extracts_low_coupling_mid_entropy_before_entropy_split(self):
        entropy_scores = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        coupling_scores = np.asarray([0.9, 0.9, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.9, 0.9], dtype=np.float32)

        groups, metadata = DatasetForRADRec._assign_entropy_groups(
            entropy_scores,
            low_ratio=0.2,
            high_ratio=0.2,
            coupling_scores=coupling_scores,
            low_coupling_ratio=0.4,
        )

        self.assertEqual(groups.tolist(), [0, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        self.assertEqual(metadata["vulnerable_count"], 4)
        self.assertEqual(metadata["low_count"], 1)
        self.assertEqual(metadata["high_count"], 1)
        self.assertEqual(metadata["mid_count"], 8)

    def test_saved_cache_keeps_entropy_only_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.npz"
            DatasetForRADRec._save_entropy_cache_to_file(
                cache_path,
                entropy_scores=np.asarray([0.0, 1.0], dtype=np.float32),
                entropy_groups=np.asarray([0, 2], dtype=np.int64),
                metadata={"partition_strategy": "se_ce_then_entropy"},
            )

            with np.load(cache_path, allow_pickle=False) as cached_data:
                self.assertEqual(
                    set(cached_data.files),
                    {"entropy_scores", "entropy_groups", "metadata_json"},
                )
            self.assertIsNotNone(DatasetForRADRec._load_entropy_cache_from_file(cache_path))


if __name__ == "__main__":
    unittest.main()
