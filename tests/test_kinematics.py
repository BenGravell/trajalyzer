import unittest

import numpy as np

from trajectory_manifold.kinematics import infer_kinematics


class KinematicsTests(unittest.TestCase):
    def test_constant_acceleration(self):
        t = np.arange(50) / 10
        positions = np.column_stack((4 * t + .5 * 1.2 * t**2, np.zeros_like(t)))
        states, actions, valid = infer_kinematics(positions)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(states[:, 3], 4 + 1.2 * t, atol=1e-10)
        np.testing.assert_allclose(actions[:, 0], 1.2, atol=1e-10)
        np.testing.assert_allclose(actions[:, 1], 0, atol=1e-10)

    def test_constant_curvature(self):
        t = np.arange(50) / 10
        radius, angular_rate = 10., .4
        positions = np.column_stack((radius * np.sin(angular_rate * t),
                                     radius * (1 - np.cos(angular_rate * t))))
        _, actions, valid = infer_kinematics(positions)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(actions[3:-3, 1], 1 / radius, atol=.001)
        np.testing.assert_allclose(actions[3:-3, 0], 0, atol=.01)

    def test_stationary_is_unidentifiable(self):
        states, actions, valid = infer_kinematics(np.zeros((50, 2)))
        self.assertFalse(valid.any())
        self.assertTrue(np.isnan(states[:, 2]).all())
        self.assertTrue(np.isnan(actions).all())


if __name__ == "__main__":
    unittest.main()
