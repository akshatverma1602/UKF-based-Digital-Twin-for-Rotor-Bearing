"""
ekf_twin.py
===========
Extended Kalman Filter companion to the UKF. It is *not* the primary estimator
-- it exists to (a) cross-validate the UKF on a constant-parameter case and
(b) make the physics-estimation coupling explicit through the analytic
Jacobian, which is the pedagogical heart of the method.

The augmented state and log-parameterisation are identical to ukf_twin.py:
    zeta = [ x, x_dot, y, y_dot, ln kxx, ln kyy, ln cxx, ln cyy ]

The Jacobian's parameter-sensitivity entries depend on the live state, e.g.

    d(x_ddot)/d(ln kxx) = -kxx * x / m
    d(x_ddot)/d(ln cxx) = -cxx * x_dot / m

These products of state and parameter are exactly what make the joint problem
nonlinear and what a linear Kalman filter cannot represent.
"""

import numpy as np

from .rotor_model import rotor_rhs, rk4_step

PARAM_NAMES = ["kxx", "kyy", "cxx", "cyy"]


def _aug_rhs(zeta, t, p):
    z = zeta[:4]
    kxx, kyy, cxx, cyy = np.exp(zeta[4:8])
    zdot = rotor_rhs(z, t, p, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy)
    return np.concatenate([zdot, np.zeros(4)])


def _jacobian(zeta, p):
    """Continuous-time Jacobian d(zeta_dot)/d(zeta) of the augmented (log)
    dynamics, baseline (cross-coupling off)."""
    x, vx, y, vy = zeta[:4]
    kxx, kyy, cxx, cyy = np.exp(zeta[4:8])
    m = p.m
    F = np.zeros((8, 8))
    # x_dot = vx ; y_dot = vy
    F[0, 1] = 1.0
    F[2, 3] = 1.0
    # x_ddot row
    F[1, 0] = -kxx / m            # d/dx
    F[1, 1] = -cxx / m            # d/dx_dot
    F[1, 4] = -kxx * x / m        # d/d(ln kxx)   <-- state x parameter coupling
    F[1, 6] = -cxx * vx / m       # d/d(ln cxx)
    # y_ddot row
    F[3, 2] = -kyy / m            # d/dy
    F[3, 3] = -cyy / m            # d/dy_dot
    F[3, 5] = -kyy * y / m        # d/d(ln kyy)
    F[3, 7] = -cyy * vy / m       # d/d(ln cyy)
    return F


class JointEKF:
    """Joint state + bearing-parameter Extended Kalman Filter."""

    def __init__(self, p, dt, meas_noise_std,
                 estimate=("kxx", "kyy", "cxx", "cyy"),
                 q_logk=0.05, q_logc=0.01, q_pos=1e-10, q_vel=1e-3,
                 P0_logk=0.10, P0_logc=0.20, P0_pos=1e-6, P0_vel=1e-2):
        self.p = p
        self.dt = dt
        self.n = 8
        self.estimate = tuple(estimate)
        on = np.array([nm in set(estimate) for nm in PARAM_NAMES], dtype=float)

        self.theta0 = np.array([p.kxx, p.kyy, p.cxx, p.cyy])
        self.zeta = np.zeros(self.n)
        self.zeta[4:8] = np.log(self.theta0)

        q_log = np.array([q_logk, q_logk, q_logc, q_logc]) * on
        P0_log = np.array([P0_logk, P0_logk, P0_logc, P0_logc]) * on
        self.P = np.diag(np.concatenate([
            [P0_pos**2, P0_vel**2, P0_pos**2, P0_vel**2], P0_log**2]))
        self.Q = np.diag(np.concatenate([
            [q_pos*dt, q_vel*dt, q_pos*dt, q_vel*dt], q_log*dt]))

        self.H = np.zeros((2, 8)); self.H[0, 0] = 1.0; self.H[1, 2] = 1.0
        self.R = (meas_noise_std**2) * np.eye(2)

    def step(self, y, t):
        # ---- predict: mean by RK4, covariance by linearised (Euler) Jacobian
        zeta_pred = rk4_step(_aug_rhs, self.zeta, t, self.dt, p=self.p)
        Fd = np.eye(self.n) + _jacobian(self.zeta, self.p) * self.dt
        P_pred = Fd @ self.P @ Fd.T + self.Q

        # ---- update (linear measurement) ----------------------------------
        S = self.H @ P_pred @ self.H.T + self.R
        Sinv = np.linalg.inv(S)
        K = P_pred @ self.H.T @ Sinv
        innov = y - self.H @ zeta_pred
        self.zeta = zeta_pred + K @ innov
        self.P = (np.eye(self.n) - K @ self.H) @ P_pred
        self.P = 0.5 * (self.P + self.P.T)

        theta = np.exp(self.zeta[4:8])
        s_var = np.clip(np.diag(self.P)[4:8], 0.0, None)
        return {"theta": theta, "theta_std": theta * np.sqrt(s_var),
                "P_theta": self.P[4:8, 4:8], "innov": innov,
                "nis": float(innov @ Sinv @ innov), "state": self.zeta[:4].copy()}

    def run(self, t, meas):
        N = len(t)
        theta_hat = np.zeros((N, 4)); theta_std = np.zeros((N, 4))
        innov = np.zeros((N, 2)); nis = np.zeros(N)
        state_hat = np.zeros((N, 4)); P_theta = np.zeros((N, 4, 4))
        for i in range(N):
            r = self.step(meas[i], t[i])
            theta_hat[i] = r["theta"]; theta_std[i] = r["theta_std"]
            innov[i] = r["innov"]; nis[i] = r["nis"]
            state_hat[i] = r["state"]; P_theta[i] = r["P_theta"]
        return {"t": t, "theta_hat": theta_hat, "theta_std": theta_std,
                "innov": innov, "nis": nis, "state_hat": state_hat,
                "P_theta": P_theta, "theta0": self.theta0}
