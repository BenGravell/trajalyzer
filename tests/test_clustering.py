import unittest

import numpy as np

from trajectory_manifold.clustering import assign_exemplars, fit_exemplars


class ExemplarTests(unittest.TestCase):
    def test_streamed_exemplars_and_test_assignment(self):
        t = np.arange(50) / 10
        trajectories = np.stack([
            np.column_stack(((2.0 + .02 * i) * t, a * t**2))
            for i, a in enumerate(np.linspace(-.2, .2, 60))
        ]).astype(np.float32)

        def batches():
            yield trajectories[:30]
            yield trajectories[30:]

        model = fit_exemplars(batches, n_exemplars=3, passes=2,
                              batch_size=32, seed=7)
        self.assertEqual(model.trajectories.shape, (3, 50, 2))
        self.assertEqual(int(model.cluster_sizes.sum()), 60)
        labels, distance = assign_exemplars(model, model.trajectories)
        np.testing.assert_array_equal(labels, np.arange(3))
        np.testing.assert_allclose(distance, 0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
