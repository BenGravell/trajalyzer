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
            picked = fit_store(store, output, clusters=3, batch_size=12, passes=1, seed=7)
            self.assertEqual(len(picked), 3)
            self.assertEqual(
                json.loads((output / "model-meta.json").read_text())["training_windows"], 24
            )
            observed = np.frombuffer((output / "exemplars.bin").read_bytes(), dtype="<i2")
            self.assertEqual(observed.size, 3 * 110 * 2)
            model = json.loads((output / "model.json").read_text())
            self.assertEqual(model["features"], features[picked].tolist())


if __name__ == "__main__":
    unittest.main()
