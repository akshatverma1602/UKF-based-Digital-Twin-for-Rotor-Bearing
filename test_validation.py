"""
test_validation.py
==================
Automated versions of the three validation checks. Run with:

    pytest -q          (from the project root)
    python -m pytest   (equivalent)

These assert the physics core and the estimator behave correctly, so a reviewer
can confirm the results without reading every plot.
"""

import numpy as np
import pytest

from dtwin.rotor_model import RotorParams, steady_state_amplitude, rotor_rhs, rk4_step
from dtwin.simulator import simulate
from dtwin.ukf_twin import JointUKF
from dtwin.ekf_twin import JointEKF
from dtwin.validation import free_decay_frequency
from dtwin import indicators as ind

DT = 1 / 2000.0
NOISE = 5e-6


# --------------------------------------------------------------------------
def test_integrator_matches_analytic_amplitude():
    """The time-domain RK4 response must match the closed-form synchronous
    amplitude (validates the physics core / integrator)."""
    p = RotorParams(gravity=False)
    z = np.zeros(4); t = 0.0
    xs = []
    for _ in range(int(6.0 / DT)):
        z = rk4_step(rotor_rhs, z, t, DT, p=p); t += DT
        if t > 4.0:
            xs.append(z[0])
    amp_sim = (max(xs) - min(xs)) / 2
    amp_ana = steady_state_amplitude(p, "x")
    assert abs(amp_sim - amp_ana) / amp_ana < 0.01     # within 1 %


def test_natural_frequency_check():
    """Free-decay ring-down frequency must match sqrt(k/m)/(2*pi)."""
    p = RotorParams()
    f_fft, f_ana = free_decay_frequency(p)
    assert abs(f_fft - f_ana) / f_ana < 0.01           # within 1 %


def test_constant_parameter_convergence_and_consistency():
    """With no fault the UKF must recover the true stiffnesses and stay
    statistically consistent (NIS within the chi-square band)."""
    p = RotorParams()
    sim = simulate(p, DT, 6.0, NOISE, fault=None, seed=11)
    ukf = JointUKF(p, DT, NOISE, estimate=("kxx", "kyy"))
    res = ukf.run(sim["t"], sim["meas"])
    ss = sim["t"] > 4.0
    kxx_err = abs(res["theta_hat"][ss, 0].mean() - p.kxx) / p.kxx
    kyy_err = abs(res["theta_hat"][ss, 1].mean() - p.kyy) / p.kyy
    assert kxx_err < 0.05 and kyy_err < 0.05           # within 5 %
    consist = ind.nis_consistency(res["nis"][ss])
    assert 0.80 <= consist <= 1.0                      # most samples in band


def test_fault_tracking():
    """The twin must track a 35 % stiffness loss to within a few percent."""
    p = RotorParams()
    fault = dict(d_max=0.35, t_onset=4.0, rate=2.5)
    sim = simulate(p, DT, 8.0, NOISE, fault=fault, seed=3)
    ukf = JointUKF(p, DT, NOISE, estimate=("kxx", "kyy"))
    res = ukf.run(sim["t"], sim["meas"])
    est_final = res["theta_hat"][-1, 0]
    true_final = sim["kxx_true"][-1]
    assert abs(est_final - true_final) / true_final < 0.05


def test_ukf_ekf_agreement():
    """UKF and EKF must agree on the constant-parameter case."""
    p = RotorParams()
    sim = simulate(p, DT, 6.0, NOISE, fault=None, seed=11)
    ukf = JointUKF(p, DT, NOISE, estimate=("kxx", "kyy"))
    ekf = JointEKF(p, DT, NOISE, estimate=("kxx", "kyy"))
    ru = ukf.run(sim["t"], sim["meas"])
    re = ekf.run(sim["t"], sim["meas"])
    ss = sim["t"] > 4.0
    for j in (0, 1):
        du = ru["theta_hat"][ss, j].mean()
        de = re["theta_hat"][ss, j].mean()
        assert abs(du - de) / du < 0.05


def test_frozen_damping_stays_at_nominal():
    """Coefficients excluded from `estimate` must not drift from nominal."""
    p = RotorParams()
    sim = simulate(p, DT, 3.0, NOISE, fault=None, seed=5)
    ukf = JointUKF(p, DT, NOISE, estimate=("kxx", "kyy"))   # damping frozen
    res = ukf.run(sim["t"], sim["meas"])
    assert np.allclose(res["theta_hat"][:, 2], p.cxx, rtol=1e-6)
    assert np.allclose(res["theta_hat"][:, 3], p.cyy, rtol=1e-6)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
