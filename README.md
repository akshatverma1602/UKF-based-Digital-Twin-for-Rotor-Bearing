##MP-3 — Digital Twin of a Rotor–Bearing System
A physics-based digital twin of a rotating shaft on instrumented bearings.
A model of the rotor runs alongside a simulated machine, and a recursive
Bayesian estimator continuously updates the model's bearing stiffness and
damping from noisy displacement measurements. As a seeded bearing fault
develops, the estimated coefficients diverge from their healthy nominal values —
and that divergence, together with the filter's innovation statistics, becomes a
physically interpretable condition indicator.
The project is to couple state estimation, physics-based
modelling, and condition monitoring: the Kalman filter updates actual
physical parameters of the rotor, not abstract signal features.
---
1. Physical basis
The plant is a 2-DOF Jeffcott (Laval) rotor: a disk of mass `m` on a
flexible massless shaft supported on bearings, with lateral displacements
`x` (horizontal) and `y` (vertical).
```
M q̈ + C q̇ + K q = f(t) ,      q = \[x, y]ᵀ
M = diag(m, m)
K = \[\[kxx, kxy], \[kyx, kyy]]      C = \[\[cxx, cxy], \[cyx, cyy]]
```
Excitation is rotating mass unbalance plus gravity:
```
f(t) = m e Ω² \[cos Ωt, sin Ωt]ᵀ + \[0, −mg]ᵀ
```
In first-order state-space form with state `z = \[x, ẋ, y, ẏ]ᵀ`:
```
ż = A(θ) z + b(t) ,      A(θ) = \[\[0, I], \[−M⁻¹K, −M⁻¹C]]
```
Proximity probes measure the two displacements only: `y\_k = H z\_k + v\_k`,
`H = \[\[1,0,0,0],\[0,0,1,0]]`.
The default rotor sits at a subcritical service speed (Ω = 300 rad/s ≈
47.7 Hz; critical speeds 71.2 Hz / 78.0 Hz), the operating regime of most
production machinery.
2. Digital-twin concept
The twin holds the same physics model but treats the four bearing coefficients
as unknown and re-estimates them at every measurement step. Healthy →
estimates sit at nominal and the model predicts the data. Fault grows → reality
drifts from nominal, and the estimator tracks that drift in the physical
parameters themselves, which is what makes the output diagnosable rather than
merely anomalous.
```
 truth plant ──► measurements ──► joint UKF ──► innovation / NIS (anomaly)
 (degrading)      (probe+noise)    (twin)    └─► estimated k,c  ──► fault severity
                                   nominal model ─────────────────►   (vs nominal)
```
3. Estimator (the maths)
Because the bearing coefficients sit inside `A(θ)` and multiply the state,
joint state+parameter estimation is intrinsically nonlinear — a linear
Kalman filter cannot do it. The state is augmented with the (log) parameters:
```
ζ = \[x, ẋ, y, ẏ, ln kxx, ln kyy, ln cxx, ln cyy] ,   θ̇ = 0 + w\_θ   (random walk)
```
Log-parameterisation guarantees positivity and makes a 30 % change the
same "size" of step at any absolute value.
Primary estimator — joint UKF (`ukf\_twin.py`): propagates sigma points
through the true nonlinear dynamics, no Jacobian needed.
EKF companion (`ekf\_twin.py`): analytic Jacobian whose parameter-
sensitivity terms are the coupling core, e.g.
```
  ∂ẍ/∂(ln kxx) = −kxx·x/m ,   ∂ẍ/∂(ln cxx) = −cxx·ẋ/m
  ```
Used to cross-validate the UKF and to expose the physics explicitly.
Adaptation rate is set by the random-walk spectral densities `q\_logk`,
`q\_logc` (config). They are scaled by `dt` so tuning is sample-rate
independent.
Observability caveat (important). Damping is only weakly observable from
displacement measurements away from a critical speed. The included sweep
(`07\_observability.png`) shows the damping-estimate error collapsing from ~110 %
at the service speed to ~5 % near the critical speed, while stiffness stays
accurate throughout. The baseline therefore estimates stiffness (`kxx`,
`kyy`) and freezes damping at nominal — the principled fallback. Any subset
can be estimated via the `estimate=` argument; nothing else changes.
4. Condition indicators (`indicators.py`)
NIS (normalized innovation squared) — `\~χ²(2)` when the model matches
reality. A windowed NIS leaving the chi-square band is an early anomaly
flag, independent of the parameter estimate.
Fault severity index — once the filter tracks the degrading stiffness,
its departure from nominal is reported as an interpretable percentage loss
(`FSI\_phys`) and as an uncertainty-normalized Mahalanobis distance
(`FSI\_stat`).
5. Repository layout
```
dtwin/
dtwin/
rotor_model.py   physics core: EOM, forcing, RK4, analytic references
degradation.py   sigmoid fault-progression profiles k(t)
simulator.py     truth plant: integrates the degrading rotor, emits noisy data
ukf_twin.py      joint UKF (primary estimator)        ← start here
ekf_twin.py      joint EKF companion + analytic Jacobian
indicators.py    NIS chi-square test, both FSI forms
validation.py    free-decay natural-frequency check
plot_style.py    shared publication figure style
visualize.py     all seven figures
run_demo.py        end-to-end scenario + validation + figures   ← run this
config.yaml        rotor, simulation, fault, estimator, detection settings
tests/             the validation checks as automated unit tests
figures/           generated output (created on run)
```

  ## 6\. Usage

  ```bash
pip install -r requirements.txt
python run\_demo.py        # runs everything, writes figures/ and a summary
pytest -q                 # runs the validation test suite
```
Edit `config.yaml` to change the rotor, the fault profile, the running speed, or
which coefficients are estimated.
7. Results (default scenario)
check	result
Integrator vs. analytic amplitude	0.00 % error
Natural frequency (free-decay FFT vs √(k/m)/2π)	71.11 vs 71.18 Hz (0.10 %)
Constant-parameter convergence	kxx −1.6 %, kyy −1.1 %
Filter consistency	mean NIS 2.4 (target 2.0), 93 % inside 95 % band
UKF vs EKF agreement	≤ 1.2 %
Fault tracking	true 35.0 % loss → estimated 36.0 %; onset flagged at 3.24 s
Figures: `01` stiffness tracking with ±2σ band · `02` innovation + NIS ·
`03` fault-severity trend · `04` healthy vs degraded orbits ·
`05` constant-parameter convergence · `06` measured vs reconstructed orbit ·
`07` observability sweep.
8. Scope and extensions
Implemented: anisotropic direct coefficients, joint UKF + EKF, sigmoid
degradation, NIS + FSI, observability study, validation suite. Documented
extension paths: cross-coupled coefficients (already in the formulation, off by
default), combined stiffness+damping faults, and RUL extrapolation from the FSI
trend.
---
