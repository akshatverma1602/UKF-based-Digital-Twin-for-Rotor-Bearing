"""
simulator.py
============
The *truth plant* -- a stand-in for the real machine. It integrates the rotor
with a time-varying (degrading) horizontal bearing stiffness kxx(t) while the
remaining coefficients stay healthy, then returns noisy displacement
measurements as a proximity probe would.

The digital twin (ukf_twin.py) never sees the true parameters or the true
state; it only receives the measurement stream produced here. This separation
is what makes the exercise a digital twin rather than a curve fit.
"""

import numpy as np

from .rotor_model import RotorParams, rotor_rhs, rk4_step
from .degradation import degraded_stiffness


def simulate(p: RotorParams, dt, t_end, meas_noise_std,
             fault=None, seed=0):
    """Run the truth plant and produce a measurement record.

    Parameters
    ----------
    p             : healthy RotorParams (nominal values)
    dt            : integration / sampling step                [s]
    t_end         : total record length                        [s]
    meas_noise_std: proximity-probe noise std (per axis)       [m]
    fault         : None for a healthy run, or a dict with keys
                    {d_max, t_onset, rate} describing kxx degradation
    seed          : RNG seed for repeatable noise

    Returns
    -------
    dict with time vector, true states, true kxx(t), clean and noisy
    measurements [x, y].
    """
    rng = np.random.default_rng(seed)
    n = int(round(t_end / dt))
    t = np.arange(n) * dt

    z = np.zeros(4)                      # start from rest
    Z = np.zeros((n, 4))                 # true full state history
    kxx_true = np.full(n, p.kxx)         # true (possibly degrading) stiffness

    for i in range(n):
        if fault is not None:
            kxx_true[i] = degraded_stiffness(
                t[i], p.kxx, fault["d_max"], fault["t_onset"], fault["rate"])
        Z[i] = z
        z = rk4_step(rotor_rhs, z, t[i], dt, p=p, kxx=kxx_true[i])

    # measurement = horizontal & vertical displacement + Gaussian noise
    meas_clean = Z[:, [0, 2]]
    noise = rng.normal(0.0, meas_noise_std, size=meas_clean.shape)
    meas = meas_clean + noise

    return {
        "t": t, "dt": dt,
        "Z_true": Z,
        "kxx_true": kxx_true,
        "meas_clean": meas_clean,
        "meas": meas,
        "meas_noise_std": meas_noise_std,
    }
