"""
ukf_twin.py
===========
The digital twin's estimator: a joint (augmented-state) Unscented Kalman Filter.

Because the bearing coefficients sit inside the system matrix and multiply the
state, estimating them together with the states is a *nonlinear* problem. The
UKF handles this without analytic Jacobians by propagating sigma points through
the true nonlinear dynamics.

Augmented state (n = 8):
    zeta = [ x, x_dot, y, y_dot, ln kxx, ln kyy, ln cxx, ln cyy ]

The four bearing coefficients are carried in log-space so the filter can never
produce a non-physical (negative) stiffness or damping, and so that a 30 %
change is the same "size" of step whatever the absolute value.

The parameters follow a random-walk model: their continuous-time process-noise
spectral densities (q_logk, q_logc) set how quickly the twin is allowed to
adapt -- the single most important tuning choice in the project.
"""

import numpy as np

from .rotor_model import rotor_rhs, rk4_step

PARAM_NAMES = ["kxx", "kyy", "cxx", "cyy"]


def _aug_rhs(zeta, t, p):
    """Augmented dynamics: rotor physics for the states, zero drift for the
    (log) parameters. Bearing coefficients are exp() of the log-states."""
    z = zeta[:4]
    kxx, kyy, cxx, cyy = np.exp(zeta[4:8])
    zdot = rotor_rhs(z, t, p, kxx=kxx, kyy=kyy, cxx=cxx, cyy=cyy)
    return np.concatenate([zdot, np.zeros(4)])


class JointUKF:
    """Joint state + bearing-parameter Unscented Kalman Filter."""

    def __init__(self, p, dt, meas_noise_std,
                 estimate=("kxx", "kyy", "cxx", "cyy"),
                 q_logk=0.05, q_logc=0.01,
                 q_pos=1e-10, q_vel=1e-3,
                 P0_logk=0.10, P0_logc=0.20,
                 P0_pos=1e-6, P0_vel=1e-2,
                 alpha=1.0, beta=2.0, kappa=0.0):
        """
        p              : RotorParams holding the *nominal* (healthy) values and
                         the known, fixed quantities (m, e, Omega, g, cross terms)
        dt             : filter / sampling step                          [s]
        meas_noise_std : proximity-probe noise std per axis              [m]
        estimate       : which bearing coefficients are treated as unknown.
                         Any coefficient left out is frozen at its nominal
                         value (zero process noise, zero prior variance) --
                         this is how the principled "estimate stiffness, treat
                         damping as known" fallback is realised when damping is
                         not observable at the operating speed.
        q_logk, q_logc : random-walk spectral densities for log stiffness
                         and log damping  (adaptation rate)              [1/s]
        q_pos, q_vel   : small process-noise spectral densities on the
                         physical states (model-discretisation slack)
        P0_*           : initial standard deviations for the covariance
        alpha,beta,kappa: unscented-transform scaling parameters
        """
        self.p = p
        self.dt = dt
        self.n = 8
        self.estimate = tuple(estimate)
        est = set(estimate)
        on = np.array([nm in est for nm in PARAM_NAMES], dtype=float)  # 1/0 mask

        # ----- nominal parameters (the twin's prior and reference) -----------
        self.theta0 = np.array([p.kxx, p.kyy, p.cxx, p.cyy])
        s0 = np.log(self.theta0)

        # ----- per-parameter process-noise density & prior std -------------
        q_log = np.array([q_logk, q_logk, q_logc, q_logc]) * on
        P0_log = np.array([P0_logk, P0_logk, P0_logc, P0_logc]) * on

        # ----- initial mean and covariance ----------------------------------
        self.zeta = np.zeros(self.n)
        self.zeta[4:8] = s0
        self.P = np.diag(np.concatenate([
            [P0_pos**2, P0_vel**2, P0_pos**2, P0_vel**2], P0_log**2]))

        # ----- process-noise covariance (continuous density * dt) -----------
        self.Q = np.diag(np.concatenate([
            [q_pos*dt, q_vel*dt, q_pos*dt, q_vel*dt], q_log*dt]))

        # ----- measurement model: y = [x, y] (linear selection) -------------
        self.H = np.zeros((2, self.n)); self.H[0, 0] = 1.0; self.H[1, 2] = 1.0
        self.R = (meas_noise_std**2) * np.eye(2)

        # ----- unscented weights --------------------------------------------
        self.lam = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lam)
        self.Wm = np.full(2*self.n + 1, 1.0 / (2.0*(self.n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.Wm[0] + (1.0 - alpha**2 + beta)

    # ------------------------------------------------------------------------
    def _sigma_points(self):
        """Generate 2n+1 sigma points from the current (zeta, P)."""
        P = 0.5 * (self.P + self.P.T) + 1e-15 * np.eye(self.n)  # symmetrise
        try:
            S = np.linalg.cholesky(P)
        except np.linalg.LinAlgError:           # repair a near-singular P
            w, V = np.linalg.eigh(P)
            w = np.clip(w, 1e-15, None)
            S = V @ np.diag(np.sqrt(w))
        pts = np.zeros((2*self.n + 1, self.n))
        pts[0] = self.zeta
        for i in range(self.n):
            pts[1 + i]          = self.zeta + self.gamma * S[:, i]
            pts[1 + self.n + i] = self.zeta - self.gamma * S[:, i]
        return pts

    def _propagate(self, pts, t):
        """Push every sigma point one step through the nonlinear dynamics."""
        out = np.empty_like(pts)
        for i, x in enumerate(pts):
            out[i] = rk4_step(_aug_rhs, x, t, self.dt, p=self.p)
        return out

    # ------------------------------------------------------------------------
    def step(self, y, t):
        """One predict + update cycle for a single measurement y at time t.

        Returns a dict with the posterior estimate, parameter standard
        deviations, the innovation and the normalised innovation squared.
        """
        # ---- predict --------------------------------------------------------
        pts = self._sigma_points()
        pp = self._propagate(pts, t)
        zeta_pred = self.Wm @ pp
        dP = pp - zeta_pred
        P_pred = (dP.T * self.Wc) @ dP + self.Q

        # ---- update (linear measurement; sigma points reused) --------------
        yp = pp @ self.H.T                       # measurement sigma points
        y_pred = self.Wm @ yp
        dy = yp - y_pred
        S = (dy.T * self.Wc) @ dy + self.R       # innovation covariance
        Pxy = (dP.T * self.Wc) @ dy              # state-measurement cross cov
        Sinv = np.linalg.inv(S)
        K = Pxy @ Sinv

        innov = y - y_pred
        self.zeta = zeta_pred + K @ innov
        self.P = P_pred - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # ---- package outputs -----------------------------------------------
        theta = np.exp(self.zeta[4:8])           # back to physical units
        # delta-method std of each parameter from the log-state variance
        s_var = np.clip(np.diag(self.P)[4:8], 0.0, None)
        theta_std = theta * np.sqrt(s_var)
        nis = float(innov @ Sinv @ innov)
        return {
            "theta": theta,
            "theta_std": theta_std,
            "P_theta": self.P[4:8, 4:8],
            "innov": innov,
            "nis": nis,
            "state": self.zeta[:4].copy(),
        }

    # ------------------------------------------------------------------------
    def run(self, t, meas):
        """Filter an entire measurement record.

        Returns arrays: theta_hat (N,4), theta_std (N,4), innovation (N,2),
        nis (N,), state_hat (N,4), and the nominal theta0.
        """
        N = len(t)
        theta_hat = np.zeros((N, 4)); theta_std = np.zeros((N, 4))
        innov = np.zeros((N, 2)); nis = np.zeros(N)
        state_hat = np.zeros((N, 4)); P_theta = np.zeros((N, 4, 4))
        for i in range(N):
            r = self.step(meas[i], t[i])
            theta_hat[i] = r["theta"]; theta_std[i] = r["theta_std"]
            innov[i] = r["innov"]; nis[i] = r["nis"]
            state_hat[i] = r["state"]; P_theta[i] = r["P_theta"]
        return {
            "t": t, "theta_hat": theta_hat, "theta_std": theta_std,
            "innov": innov, "nis": nis, "state_hat": state_hat,
            "P_theta": P_theta, "theta0": self.theta0,
        }
