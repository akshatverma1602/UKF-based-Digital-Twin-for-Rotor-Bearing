"""
visualize.py
============
Publication-quality figures. Each function takes already-computed data and
writes one figure; run_demo.py wires them together.
"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_style import apply_style, finish, COLORS
from .indicators import moving_average, nis_bounds

apply_style()
UM = 1e6   # metres -> micrometres


def fig_stiffness_tracking(sim, res, save):
    """Estimated vs. true horizontal stiffness with a +/-2 sigma band, plus the
    healthy vertical stiffness for contrast."""
    t = sim["t"]
    kxx_hat = res["theta_hat"][:, 0]; kxx_sd = res["theta_std"][:, 0]
    kyy_hat = res["theta_hat"][:, 1]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, sim["kxx_true"] / 1e6, color=COLORS["true"], lw=2.2,
            label="true $k_{xx}(t)$")
    ax.plot(t, kxx_hat / 1e6, color=COLORS["est"], label="estimated $\\hat{k}_{xx}$")
    ax.fill_between(t, (kxx_hat - 2*kxx_sd)/1e6, (kxx_hat + 2*kxx_sd)/1e6,
                    color=COLORS["band"], alpha=0.7, label="$\\pm 2\\sigma$")
    ax.axhline(res["theta0"][0]/1e6, color=COLORS["nominal"], ls="--", lw=1.0,
               label="nominal $k_{xx,0}$")
    ax.plot(t, kyy_hat / 1e6, color=COLORS["accent"], lw=1.1, alpha=0.8,
            label="estimated $\\hat{k}_{yy}$ (healthy axis)")
    finish(ax, "Digital twin tracks the developing bearing fault",
           "time [s]", "bearing stiffness [MN/m]")
    fig.savefig(save); plt.close(fig)


def fig_innovation_nis(res, dt, save, window_s=0.1):
    """Innovation sequence and normalised innovation squared with chi-square
    consistency bounds."""
    t = res["t"]; innov = res["innov"]; nis = res["nis"]
    w = int(window_s / dt)
    lo1, hi1 = nis_bounds(dof=2, window=1)
    loW, hiW = nis_bounds(dof=2, window=w)
    nis_ma = moving_average(nis, w)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    a1.plot(t, innov[:, 0]*UM, color=COLORS["accent"], lw=0.7, label="$\\nu_x$")
    a1.plot(t, innov[:, 1]*UM, color=COLORS["warn"], lw=0.7, alpha=0.8, label="$\\nu_y$")
    finish(a1, "Innovation sequence (measurement residual)",
           None, "innovation [$\\mu$m]")

    a2.plot(t, nis, color="#bbbbbb", lw=0.5, label="NIS (per sample)")
    a2.plot(t, nis_ma, color=COLORS["est"], lw=1.8,
            label=f"NIS moving avg ({window_s*1000:.0f} ms)")
    a2.axhline(hiW, color=COLORS["warn"], ls="--", lw=1.0,
               label="95% bound (windowed)")
    a2.axhline(loW, color=COLORS["warn"], ls="--", lw=1.0)
    a2.set_ylim(0, max(hi1*1.1, np.percentile(nis, 99)))
    finish(a2, "NIS stays within the consistency band (filter well-matched)",
           "time [s]", "NIS")
    fig.savefig(save); plt.close(fig)


def fig_fsi(t, fsi_phys, true_damage, fsi_stat, onset, save):
    """Fault-severity-index trend with the detected onset marker; physical
    percentage on the left axis, statistical (Mahalanobis) distance on right."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, true_damage, color=COLORS["true"], lw=2.0, label="true stiffness loss")
    ax.plot(t, fsi_phys, color=COLORS["est"], label="$FSI_{phys}$ (estimated)")
    ax.set_ylabel("stiffness loss [%]")
    if onset is not None:
        ax.axvline(onset, color=COLORS["warn"], ls=":", lw=1.6,
                   label=f"detected onset @ {onset:.2f}s")
    ax2 = ax.twinx()
    ax2.plot(t, fsi_stat, color=COLORS["purple"], lw=1.1, alpha=0.8,
             label="$FSI_{stat}$ (Mahalanobis)")
    ax2.set_ylabel("Mahalanobis distance [-]", color=COLORS["purple"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    ax.set_title("Fault severity index"); ax.set_xlabel("time [s]")
    fig.savefig(save); plt.close(fig)


def _orbit(ax, x, y, color, label, scatter=None):
    ax.plot(x*UM, y*UM, color=color, lw=1.3, label=label)
    if scatter is not None:
        ax.scatter(scatter[0]*UM, scatter[1]*UM, s=3, color="#cccccc",
                   alpha=0.5, zorder=0)
    ax.set_aspect("equal", "datalim")


def fig_orbits(sim, save, healthy=(1.0, 1.25), degraded=(7.4, 7.65)):
    """Shaft-centre orbit, healthy vs. degraded state."""
    t = sim["t"]; Z = sim["Z_true"]; M = sim["meas"]
    def win(lo, hi):
        m = (t >= lo) & (t <= hi)
        return Z[m, 0], Z[m, 2], (M[m, 0], M[m, 1])
    hx, hy, hs = win(*healthy)
    dx, dy, ds = win(*degraded)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.9))
    _orbit(a1, hx, hy, COLORS["est"], "true orbit", hs)
    finish(a1, f"Healthy ({healthy[0]:.1f}-{healthy[1]:.1f}s)",
           "x [$\\mu$m]", "y [$\\mu$m]")
    _orbit(a2, dx, dy, COLORS["warn"], "true orbit", ds)
    finish(a2, f"Degraded ({degraded[0]:.1f}-{degraded[1]:.1f}s)",
           "x [$\\mu$m]", "y [$\\mu$m]")
    fig.suptitle("Orbit grows as horizontal stiffness drops", fontweight="bold")
    fig.savefig(save); plt.close(fig)


def fig_reconstruction(sim, res, save, window=(7.4, 7.65)):
    """Measured orbit vs. the twin's reconstructed (estimated-state) orbit."""
    t = sim["t"]
    m = (t >= window[0]) & (t <= window[1])
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.scatter(sim["meas"][m, 0]*UM, sim["meas"][m, 1]*UM, s=6,
               color=COLORS["nominal"], alpha=0.6, label="measured")
    ax.plot(res["state_hat"][m, 0]*UM, res["state_hat"][m, 2]*UM,
            color=COLORS["est"], lw=1.6, label="twin reconstruction")
    ax.set_aspect("equal", "datalim")
    finish(ax, "Measured vs. reconstructed orbit (degraded)",
           "x [$\\mu$m]", "y [$\\mu$m]")
    fig.savefig(save); plt.close(fig)


def fig_param_convergence(sim_c, res_c, save):
    """Constant-truth validation: estimates converge to the true values."""
    t = sim_c["t"]; th = res_c["theta_hat"]; sd = res_c["theta_std"]
    th0 = res_c["theta0"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for j, (nm, c) in enumerate([("k_{xx}", COLORS["est"]),
                                 ("k_{yy}", COLORS["accent"])]):
        ax.plot(t, th[:, j]/1e6, color=c, label=f"$\\hat{{{nm}}}$")
        ax.fill_between(t, (th[:, j]-2*sd[:, j])/1e6, (th[:, j]+2*sd[:, j])/1e6,
                        color=c, alpha=0.15)
        ax.axhline(th0[j]/1e6, color=c, ls="--", lw=1.0)
    finish(ax, "Validation: constant-parameter convergence (truth dashed)",
           "time [s]", "stiffness [MN/m]")
    fig.savefig(save); plt.close(fig)


def fig_observability(omegas_hz, err_cxx, err_kxx, crit_hz, save):
    """Why damping is frozen at the service speed: damping-estimate error
    collapses only as the running speed approaches the critical speed."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(omegas_hz, np.abs(err_cxx), "o-", color=COLORS["warn"],
            label="|error| in $\\hat{c}_{xx}$ (damping)")
    ax.plot(omegas_hz, np.abs(err_kxx), "s-", color=COLORS["est"],
            label="|error| in $\\hat{k}_{xx}$ (stiffness)")
    ax.axvline(crit_hz, color=COLORS["nominal"], ls="--", lw=1.2,
               label=f"critical speed {crit_hz:.0f} Hz")
    ax.set_yscale("log")
    finish(ax, "Damping is observable only near the critical speed",
           "running speed [Hz]", "parameter error [%]")
    fig.savefig(save); plt.close(fig)
