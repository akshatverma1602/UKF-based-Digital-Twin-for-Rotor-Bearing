"""
indicators.py
=============
Two complementary condition indicators derived from the filter output.

1. Innovation / NIS  -- anomaly detection.
   The normalised innovation squared is chi-square distributed with dof equal
   to the measurement dimension when the model matches reality. A NIS that
   leaves the chi-square confidence band signals that the twin no longer
   explains the data -- an early fault flag that fires before the parameter
   estimate has fully re-converged.

2. Fault severity index (FSI) -- severity / diagnosis.
   Once the filter tracks the degrading stiffness, its departure from the
   healthy nominal quantifies how bad the fault is, reported both as an
   interpretable percentage and as an uncertainty-normalised (Mahalanobis)
   distance.
"""

import numpy as np
from scipy.stats import chi2

PARAM_NAMES = ["kxx", "kyy", "cxx", "cyy"]


def moving_average(x, w):
    """Centred-ish moving average over a window of w samples."""
    w = max(1, int(w))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def nis_bounds(dof=2, window=1, conf=0.95):
    """Two-sided confidence bounds for (windowed-average) NIS.

    A single NIS sample ~ chi2(dof). The average of `window` independent
    samples times `window` ~ chi2(dof*window), so the bounds on the *average*
    tighten as the window grows.
    """
    lo = (1 - conf) / 2
    hi = 1 - lo
    d = dof * window
    return chi2.ppf(lo, d) / window, chi2.ppf(hi, d) / window


def nis_consistency(nis, dof=2, conf=0.95):
    """Fraction of NIS samples inside the single-sample chi-square band
    (should be ~conf when the filter is consistent)."""
    lo, hi = nis_bounds(dof=dof, window=1, conf=conf)
    return float(np.mean((nis >= lo) & (nis <= hi)))


def fsi_physical(theta_hat, theta0, param="kxx"):
    """Percent loss of a chosen coefficient: 100*(theta0 - theta_hat)/theta0."""
    j = PARAM_NAMES.index(param)
    return 100.0 * (theta0[j] - theta_hat[:, j]) / theta0[j]


def fsi_mahalanobis(theta_hat, theta0, P_theta, est_idx):
    """Uncertainty-normalised severity over the *estimated* coefficients.

        FSI_stat = sqrt( d^T P^{-1} d ),   d = theta_hat - theta0

    Only the estimated sub-vector/sub-covariance is used, because frozen
    coefficients carry no variance (their covariance block is singular).
    """
    est_idx = list(est_idx)
    N = theta_hat.shape[0]
    out = np.zeros(N)
    d_all = theta_hat - theta0[None, :]
    for i in range(N):
        d = d_all[i, est_idx]
        Pi = P_theta[i][np.ix_(est_idx, est_idx)]
        out[i] = np.sqrt(d @ np.linalg.solve(Pi + 1e-18 * np.eye(len(est_idx)), d))
    return out


def detect_onset(indicator, threshold, t, persist=0.2, dt=None):
    """First time the indicator exceeds `threshold` and stays above it for at
    least `persist` seconds. Returns the onset time, or None."""
    if dt is None:
        dt = t[1] - t[0]
    need = int(persist / dt)
    above = indicator > threshold
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= need:
            return t[i - need + 1]
    return None
