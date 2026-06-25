"""
validation.py
=============
Physics-core and filter validation routines used by run_demo.py and the unit
tests. These are the checks a reviewer should be able to run to convince
themselves the twin rests on correct foundations.
"""

import numpy as np

from .rotor_model import RotorParams, rotor_rhs, rk4_step


def free_decay_frequency(p: RotorParams, dt=1/10000.0, t_end=0.5, x0=1e-4):
    """Free-vibration ring-down on the x axis: release the rotor from an initial
    displacement with the unbalance and gravity switched off, then read the
    decay frequency from a zero-padded FFT. It should match
    f_n = sqrt(kxx/m)/(2*pi) (within the small shift due to damping).

    The window is kept short (the lightly-damped ring-down decays in a few tens
    of milliseconds) and no taper is applied, so the burst is not suppressed;
    the spectrum is zero-padded for fine peak resolution and the near-DC bins
    are excluded when locating the peak.

    Returns (f_fft, f_analytic) in Hz.
    """
    pf = RotorParams(**{**p.__dict__})
    pf.e = 0.0           # no unbalance forcing
    pf.gravity = False   # no static load
    n = int(t_end / dt)
    z = np.array([x0, 0.0, 0.0, 0.0])
    x = np.zeros(n)
    for i in range(n):
        x[i] = z[0]
        z = rk4_step(rotor_rhs, z, i * dt, dt, p=pf)
    x = x - x.mean()
    nfft = 1 << 16
    spec = np.abs(np.fft.rfft(x, n=nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    valid = freqs > 10.0                      # ignore the near-DC region
    f_fft = freqs[valid][np.argmax(spec[valid])]
    f_ana = np.sqrt(p.kxx / p.m) / (2 * np.pi)
    return float(f_fft), float(f_ana)
