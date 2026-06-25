"""
run_demo.py
===========
End-to-end demonstration of the digital twin.

Pipeline:
  1. Load the scenario from config.yaml.
  2. Run the truth plant with a developing horizontal-stiffness fault.
  3. Run the joint UKF (primary) to track the bearing coefficients.
  4. Compute the condition indicators (NIS, FSI).
  5. Run the validation scenarios (constant-parameter convergence,
     free-decay natural-frequency check, UKF-vs-EKF agreement, observability
     sweep that motivates freezing damping).
  6. Write every figure to ./figures and print a summary.

Run:  python run_demo.py
"""

import os
import numpy as np
import yaml

from dtwin.rotor_model import RotorParams
from dtwin.simulator import simulate
from dtwin.ukf_twin import JointUKF, PARAM_NAMES
from dtwin.ekf_twin import JointEKF
from dtwin import indicators as ind
from dtwin import visualize as viz
from dtwin.validation import free_decay_frequency

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def load_config():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    r = cfg["rotor"]; s = cfg["simulation"]; flt = cfg["fault"]; est = cfg["estimator"]
    r = {k: (bool(v) if k == "gravity" else float(v)) for k, v in r.items()}
    flt = {k: float(v) for k, v in flt.items()}
    p = RotorParams(**r)
    dt = float(s["dt"]); T = float(s["t_end"]); noise = float(s["meas_noise_std"])
    est_set = tuple(est["estimate"])
    crit_x, crit_y = p.natural_frequencies_hz()

    print("=" * 64)
    print("MP-3 digital twin -- demo run")
    print("=" * 64)
    print(f"running speed   : {p.Omega:.0f} rad/s ({p.Omega/2/np.pi:.1f} Hz)")
    print(f"critical speeds : x {crit_x:.1f} Hz | y {crit_y:.1f} Hz")
    print(f"estimating      : {', '.join(est_set)}  (others frozen at nominal)")

    # ---- 1) fault scenario + UKF ---------------------------------------
    sim = simulate(p, dt, T, noise, fault=flt, seed=int(s["seed"]))
    ukf = JointUKF(p, dt, noise, estimate=est_set,
                   q_logk=float(est["q_logk"]), q_logc=float(est["q_logc"]))
    res = ukf.run(sim["t"], sim["meas"])

    # ---- 2) indicators -------------------------------------------------
    fsi_phys = ind.fsi_physical(res["theta_hat"], res["theta0"], "kxx")
    true_damage = 100.0 * (1.0 - sim["kxx_true"] / p.kxx)
    est_idx = [PARAM_NAMES.index(nm) for nm in est_set]
    fsi_stat = ind.fsi_mahalanobis(res["theta_hat"], res["theta0"],
                                   res["P_theta"], est_idx)
    onset = ind.detect_onset(fsi_phys, float(cfg["detection"]["fsi_threshold_pct"]),
                             sim["t"], persist=float(cfg["detection"]["persist_s"]),
                             dt=dt)
    final_loss_true = true_damage[-1]
    final_loss_est = fsi_phys[-1]
    print("-" * 64)
    print(f"true final stiffness loss : {final_loss_true:5.1f} %")
    print(f"estimated final loss      : {final_loss_est:5.1f} %")
    print(f"fault onset detected at   : "
          f"{('%.2f s' % onset) if onset else 'not detected'}")

    # ---- 3) validation: constant-parameter convergence -----------------
    sim_c = simulate(p, dt, 6.0, noise, fault=None, seed=11)
    ukf_c = JointUKF(p, dt, noise, estimate=est_set,
                     q_logk=float(est["q_logk"]))
    res_c = ukf_c.run(sim_c["t"], sim_c["meas"])
    ss = sim_c["t"] > 4.0
    conv_err = [100*(res_c["theta_hat"][ss, j].mean() - res_c["theta0"][j])
                / res_c["theta0"][j] for j in est_idx]
    nis_consist = ind.nis_consistency(res_c["nis"][ss])
    print("-" * 64)
    print("validation 1  constant-parameter convergence:")
    for nm, e in zip(est_set, conv_err):
        print(f"    {nm:4s} steady-state error : {e:+.2f} %")
    print(f"    mean NIS = {res_c['nis'][ss].mean():.2f} (target 2.0), "
          f"{100*nis_consist:.0f}% of samples inside 95% band")

    # ---- 4) validation: free-decay natural frequency -------------------
    f_fft, f_ana = free_decay_frequency(p)
    print("validation 2  natural-frequency check:")
    print(f"    FFT peak {f_fft:.2f} Hz vs analytic {f_ana:.2f} Hz "
          f"({100*abs(f_fft-f_ana)/f_ana:.2f}% error)")

    # ---- 5) validation: UKF vs EKF on constant case --------------------
    ekf_c = JointEKF(p, dt, noise, estimate=est_set, q_logk=float(est["q_logk"]))
    res_e = ekf_c.run(sim_c["t"], sim_c["meas"])
    diff = [100*abs(res_e["theta_hat"][ss, j].mean() - res_c["theta_hat"][ss, j].mean())
            / res_c["theta_hat"][ss, j].mean() for j in est_idx]
    print("validation 3  UKF vs EKF agreement (constant case):")
    for nm, d in zip(est_set, diff):
        print(f"    {nm:4s} difference : {d:.3f} %")

    # ---- 6) observability sweep (motivates freezing damping) -----------
    omegas = [300, 360, 400, 430, 450]
    err_c, err_k, om_hz = [], [], []
    for Om in omegas:
        pp = RotorParams(**{**r, "Omega": float(Om)})
        sc = simulate(pp, dt, 4.0, noise, fault=None, seed=7)
        uk = JointUKF(pp, dt, noise, estimate=("kxx", "kyy", "cxx", "cyy"),
                      q_logc=0.01)
        rr = uk.run(sc["t"], sc["meas"])
        mm = sc["t"] > 3.0
        th = rr["theta_hat"][mm].mean(axis=0)
        err_c.append(100*(th[2]-pp.cxx)/pp.cxx)
        err_k.append(100*(th[0]-pp.kxx)/pp.kxx)
        om_hz.append(Om/2/np.pi)

    # ---- figures -------------------------------------------------------
    viz.fig_stiffness_tracking(sim, res, os.path.join(FIG, "01_stiffness_tracking.png"))
    viz.fig_innovation_nis(res, dt, os.path.join(FIG, "02_innovation_nis.png"))
    viz.fig_fsi(sim["t"], fsi_phys, true_damage, fsi_stat, onset,
                os.path.join(FIG, "03_fault_severity.png"))
    viz.fig_orbits(sim, os.path.join(FIG, "04_orbits.png"))
    viz.fig_param_convergence(sim_c, res_c, os.path.join(FIG, "05_param_convergence.png"))
    viz.fig_reconstruction(sim, res, os.path.join(FIG, "06_reconstruction.png"))
    viz.fig_observability(om_hz, err_c, err_k, crit_x,
                          os.path.join(FIG, "07_observability.png"))
    print("-" * 64)
    print(f"figures written to {FIG}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
