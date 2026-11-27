"""Infer a planar kinematic state and actions from sampled flat outputs (x, y).

Model: x_dot = v cos(yaw), y_dot = v sin(yaw), yaw_dot = v curvature,
v_dot = acceleration. The flatness formulas require nonzero forward speed.
Local least-squares cubic fits estimate derivatives without an extra dependency.
"""

from __future__ import annotations

import numpy as np


def infer_kinematics(
    positions: np.ndarray,
    sample_rate: float = 10.0,
    window: int = 7,
    degree: int = 3,
    min_speed: float = 0.5,
):
    """Return ``(states, actions, valid)`` for (N,T,2) or (T,2) positions.

    states are [x, y, yaw, speed] in m, rad, m/s; actions are [accel,
    curvature] in m/s², 1/m. Derivatives use centered local polynomial fits
    with shifted one-sided windows at the boundaries. Position state retains
    the observed coordinates; derivative-derived channels are smoothed.

    At speeds below ``min_speed``, yaw and both actions are NaN and valid is
    False. This is an identifiability mask, not a claim of zero steering.
    Results should be treated as estimates, especially at window boundaries.
    """
    data = np.asarray(positions, dtype=np.float64)
    single = data.ndim == 2
    if single:
        data = data[None]
    if data.ndim != 3 or data.shape[-1] != 2 or data.shape[1] < 4:
        raise ValueError("positions must have shape (N,T,2) or (T,2), T >= 4")
    if not np.isfinite(data).all():
        raise ValueError("positions must be finite")
    if (
        not np.isfinite(sample_rate)
        or sample_rate <= 0
        or not np.isfinite(min_speed)
        or min_speed <= 0
    ):
        raise ValueError("sample_rate and min_speed must be positive finite values")
    if window < 4 or window % 2 != 1 or degree < 2 or degree >= window:
        raise ValueError("window must be odd and >= 5; require 2 <= degree < window")
    n, frames, _ = data.shape
    width = min(window, frames)
    velocity = np.empty_like(data)
    second = np.empty_like(data)
    for i in range(frames):
        start = min(max(0, i - width // 2), frames - width)
        offsets = (np.arange(start, start + width) - i) / sample_rate
        fit = np.linalg.pinv(np.vander(offsets, degree + 1, increasing=True))
        chunk = data[:, start : start + width, :]
        velocity[:, i, :] = np.einsum("j,njd->nd", fit[1], chunk)
        second[:, i, :] = np.einsum("j,njd->nd", 2 * fit[2], chunk)

    vx, vy = velocity[..., 0], velocity[..., 1]
    ax, ay = second[..., 0], second[..., 1]
    speed = np.hypot(vx, vy)
    valid = speed >= min_speed
    yaw = np.unwrap(np.arctan2(vy, vx), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = (vx * ax + vy * ay) / speed
        curvature = (vx * ay - vy * ax) / speed**3
    yaw = np.where(valid, yaw, np.nan)
    actions = np.stack(
        (np.where(valid, accel, np.nan), np.where(valid, curvature, np.nan)), axis=-1
    )
    states = np.concatenate((data, yaw[..., None], speed[..., None]), axis=-1)
    return (states[0], actions[0], valid[0]) if single else (states, actions, valid)
