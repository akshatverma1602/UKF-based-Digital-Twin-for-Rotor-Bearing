"""
degradation.py
==============
Fault-progression profiles. A developing bearing fault (loss of preload, wear,
incipient spalling) is represented as a gradual change of an effective bearing
coefficient over time. A smooth sigmoid is used rather than a linear ramp so the
onset is slow, then accelerates, then saturates -- the qualitative signature of
many real degradation processes.

    k(t) = k0 * [1 - D_max * s(t)]

where s(t) in [0, 1] is the normalised damage curve.
"""

import numpy as np


def sigmoid_damage(t, t_onset, rate):
    """Normalised damage s(t) in [0, 1] following a logistic curve.

    t_onset : time of the inflection point (50 % damage)   [s]
    rate    : steepness of the transition                  [1/s]
    """
    return 1.0 / (1.0 + np.exp(-rate * (t - t_onset)))


def degraded_stiffness(t, k0, d_max, t_onset, rate):
    """Time-varying stiffness k(t) = k0 * (1 - d_max * s(t)).

    d_max : fractional stiffness loss at full damage (e.g. 0.35 = -35 %)
    """
    return k0 * (1.0 - d_max * sigmoid_damage(t, t_onset, rate))


def damage_fraction(t, d_max, t_onset, rate):
    """Convenience: the true fractional stiffness loss at time t (for plotting
    the ground-truth severity the estimator is trying to recover)."""
    return d_max * sigmoid_damage(t, t_onset, rate)
