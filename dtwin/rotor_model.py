"""
rotor_model.py
==============
Physics core of the digital twin: a 2-DOF Jeffcott (Laval) rotor supported on
bearings whose stiffness and damping are the quantities we later estimate.

State (physical):      z = [x, x_dot, y, y_dot]
DOFs:                  lateral disk displacements x (horizontal), y (vertical)
Excitation:            rotating mass unbalance + gravity
Bearing coefficients:  direct terms kxx, kyy, cxx, cyy (anisotropic) and,
                       optionally, cross-coupled terms kxy, kyx, cxy, cyx.

Everything here is *pure physics*. There is no estimation logic in this module
so that the same equations of motion serve both the truth plant (simulator.py)
and the digital twin's internal model (ukf_twin.py / ekf_twin.py).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RotorParams:
    """Physical parameters of the rotor-bearing system (SI units)."""
    m: float = 5.0          # disk mass                       [kg]
    e: float = 1.0e-4       # mass eccentricity (unbalance)   [m]
    Omega: float = 300.0    # running speed                   [rad/s]
    g: float = 9.81         # gravitational acceleration      [m/s^2]
    gravity: bool = True    # include static gravity load
    # direct bearing coefficients
    kxx: float = 1.0e6      # horizontal stiffness            [N/m]
    kyy: float = 1.2e6      # vertical stiffness              [N/m]
    cxx: float = 250.0      # horizontal damping              [N.s/m]
    cyy: float = 300.0      # vertical damping                [N.s/m]
    # cross-coupled coefficients (off in the baseline configuration)
    kxy: float = 0.0
    kyx: float = 0.0
    cxy: float = 0.0
    cyx: float = 0.0

    def natural_frequencies_hz(self):
        """Undamped natural frequencies of the two direct axes [Hz].

        For the decoupled direct model each axis behaves like an SDOF
        oscillator with f_n = (1/2*pi) * sqrt(k/m).
        """
        fx = np.sqrt(self.kxx / self.m) / (2.0 * np.pi)
        fy = np.sqrt(self.kyy / self.m) / (2.0 * np.pi)
        return fx, fy


def unbalance_force(t, p: RotorParams):
    """Synchronous unbalance force plus gravity, returned as [Fx, Fy] in N.

    The rotating unbalance produces a force of magnitude m*e*Omega^2 that spins
    with the shaft; gravity is a constant downward load on the vertical DOF.
    """
    amp = p.m * p.e * p.Omega ** 2
    Fx = amp * np.cos(p.Omega * t)
    Fy = amp * np.sin(p.Omega * t)
    if p.gravity:
        Fy = Fy - p.m * p.g
    return Fx, Fy


def rotor_rhs(z, t, p: RotorParams,
              kxx=None, kyy=None, cxx=None, cyy=None):
    """Right-hand side of the equations of motion: returns z_dot.

    The four bearing coefficients can be overridden (this is how the truth
    plant injects a *time-varying* fault while the rest of the physics stays
    fixed). Cross-coupled terms are taken from ``p`` unchanged.

        m*x_ddot = Fx - cxx*x_dot - cxy*y_dot - kxx*x - kxy*y
        m*y_ddot = Fy - cyx*x_dot - cyy*y_dot - kyx*x - kyy*y
    """
    kxx = p.kxx if kxx is None else kxx
    kyy = p.kyy if kyy is None else kyy
    cxx = p.cxx if cxx is None else cxx
    cyy = p.cyy if cyy is None else cyy

    x, vx, y, vy = z
    Fx, Fy = unbalance_force(t, p)

    ax = (Fx - cxx * vx - p.cxy * vy - kxx * x - p.kxy * y) / p.m
    ay = (Fy - p.cyx * vx - cyy * vy - p.kyx * x - kyy * y) / p.m
    return np.array([vx, ax, vy, ay])


def rk4_step(rhs, z, t, dt, **kwargs):
    """One classical fourth-order Runge-Kutta step of an autonomous-in-state,
    time-dependent ODE z_dot = rhs(z, t, ...)."""
    k1 = rhs(z, t, **kwargs)
    k2 = rhs(z + 0.5 * dt * k1, t + 0.5 * dt, **kwargs)
    k3 = rhs(z + 0.5 * dt * k2, t + 0.5 * dt, **kwargs)
    k4 = rhs(z + dt * k3, t + dt, **kwargs)
    return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def steady_state_amplitude(p: RotorParams, axis="x"):
    """Analytic synchronous response amplitude of one direct axis [m].

    Used as a validation reference for the time-domain integrator:
        X = m*e*Omega^2 / sqrt((k - m*Omega^2)^2 + (c*Omega)^2)
    (valid for the decoupled SDOF approximation, i.e. cross terms = 0).
    """
    k = p.kxx if axis == "x" else p.kyy
    c = p.cxx if axis == "x" else p.cyy
    F = p.m * p.e * p.Omega ** 2
    denom = np.sqrt((k - p.m * p.Omega ** 2) ** 2 + (c * p.Omega) ** 2)
    return F / denom
