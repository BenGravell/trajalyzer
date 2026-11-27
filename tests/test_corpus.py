"""Static shard protocol and bounded flush behavior."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from trajectory_manifold.corpus import CorpusWriter


class CorpusTests(unittest.TestCase):
    def test_shards_and_action_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            writer = CorpusWriter(root, shard_size=2)
            paths = np.zeros((3, 50, 2), dtype=np.float32)
            paths[:, :, 0] = np.arange(50) / 10 * 4
            actions = np.zeros_like(paths)
            actions[:, :, 0] = np.array([1, 2, 3])[:, None]
            writer.add(paths, actions, np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))
            manifest = writer.finish()
            self.assertEqual(manifest["count"], 3)
            self.assertEqual([s["count"] for s in manifest["shards"]], [2, 1])
            self.assertEqual(json.loads((root / "manifest.json").read_text())["count"], 3)
            payload = (root / manifest["shards"][0]["path"]).read_bytes()
            self.assertEqual(len(payload), 416)
            self.assertEqual(int(np.frombuffer(payload[200:204], dtype="<u4")[0]), 0)
            self.assertEqual(int(np.frombuffer(payload[408:412], dtype="<u4")[0]), 1)
            detail = (
                root / manifest["shards"][0]["path"].replace(".bin", ".actions.bin")
            ).read_bytes()
            self.assertEqual(len(detail), 800)
            self.assertEqual(
                np.frombuffer(detail, dtype="<f4").reshape(2, 50, 2)[:, 0, 0].tolist(), [1, 2]
            )


if __name__ == "__main__":
    unittest.main()
