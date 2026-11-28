"""Exemplar fitting reads the whole feature store without materializing it."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.fit_av2_exemplars import fit_store


class Av2FitTests(unittest.TestCase):
    def test_observed_exemplars_from_entire_store(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store, output = root / "store", root / "model"
            store.mkdir()
            positions = np.zeros((24, 110, 2), dtype="<f4")
            features = np.zeros((24, 275), dtype="<f4")
            for i in range(24):
                positions[i, :, 0] = np.arange(110) * (2 + i / 24) / 10
                features[i, :] = i / 4
            positions.tofile(store / "positions.f32")
            features.tofile(store / "features.f32")
            (store / "store.json").write_text(
                json.dumps({"count": 24, "frames": 110, "feature_dim": 275})
            )
            picked = fit_store(store, output, clusters=3, batch_size=12, seed=7)
            self.assertEqual(len(picked), 3)
            meta = json.loads((output / "model-meta.json").read_text())
            self.assertEqual(meta["training_windows"], 24)
            self.assertEqual(meta["backend"], "faiss-kmeans")
            self.assertEqual(meta["passes"], 10)
            observed = np.frombuffer((output / "exemplars.bin").read_bytes(), dtype="<i2")
            self.assertEqual(observed.size, 3 * 110 * 2)
            model = json.loads((output / "model.json").read_text())
            self.assertEqual(model["features"], features[picked].tolist())

    def test_auto_counts_density_clusters_then_fits_all_windows(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            store, output = root / "store", root / "model"
            store.mkdir()
            rng = np.random.default_rng(17)
            centers = np.tile([[0, 0], [10, 0], [0, 10]], (40, 1))
            features = (centers + rng.normal(0, 0.05, centers.shape)).astype("<f4")
            positions = np.zeros((len(features), 2, 2), dtype="<f4")
            positions[:, 1, :] = features
            positions.tofile(store / "positions.f32")
            features.tofile(store / "features.f32")
            (store / "store.json").write_text(
                json.dumps({"count": len(features), "frames": 2, "feature_dim": 2})
            )

            picked = fit_store(
                store,
                output,
                clusters="auto",
                batch_size=24,
                passes=1,
                seed=7,
                density_sample_size=90,
            )
            meta = json.loads((output / "model-meta.json").read_text())
            self.assertEqual(len(picked), 3)
            self.assertEqual(meta["count"], 3)
            self.assertEqual(meta["training_windows"], len(features))
            self.assertEqual(sum(meta["cluster_sizes"]), len(features))
            selection = meta["count_selection"]
            self.assertEqual(selection["sample_windows"], 90)
            self.assertEqual(selection["method"], "hdbscan")
            self.assertFalse(selection["uses_full_store"])
            self.assertEqual(
                sum(selection["sample_cluster_sizes"]) + selection["noise_windows"], 90
            )
            model = json.loads((output / "model.json").read_text())
            self.assertEqual(model["features"], features[picked].tolist())

            picked = fit_store(store, output, clusters="auto", batch_size=24, passes=1, seed=7)
            selection = json.loads((output / "model-meta.json").read_text())["count_selection"]
            self.assertEqual(len(picked), 3)
            self.assertEqual(selection["sample_windows"], len(features))
            self.assertTrue(selection["uses_full_store"])

            features.fill(0)
            features.tofile(store / "features.f32")
            with self.assertRaisesRegex(ValueError, "HDBSCAN found 0 clusters"):
                fit_store(
                    store, root / "no-model", clusters="auto", batch_size=24, passes=1, seed=7
                )
            self.assertFalse((root / "no-model").exists())


if __name__ == "__main__":
    unittest.main()
