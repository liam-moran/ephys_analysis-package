from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "data" / "example_data" / "Ephys practice recordings female P40 26-10-26"


@pytest.fixture
def example_abf():
    """Path factory for a real example ABF; skips the test if data are absent."""
    def _get(name: str) -> Path:
        p = EXAMPLE_DIR / name
        if not p.exists():
            pytest.skip(f"example data not available: {name}")
        return p
    return _get


def simulate_rc_pulse(Ra=20.0, Rm=800.0, Cm=40.0, dV=-10.0, fs=50_000, pulse_s=0.025,
                      pre_s=0.025, post_s=0.025, holding_pA=-20.0, noise_pA=0.0,
                      filter_hz=None, seed=0):
    """Voltage-clamp current of an ideal Ra-(Rm||Cm) cell for one square pulse.

    Ra, Rm in MOhm; Cm in pF; dV in mV; returns t (s), current (pA), command (mV).
    Exact solution per edge: I = dV/Rt + (dV/Ra - dV/Rt) * exp(-t/tau),
    tau = Ra*Cm*Rm/(Ra+Rm).
    """
    rng = np.random.default_rng(seed)
    n_pre, n_pulse, n_post = int(pre_s * fs), int(pulse_s * fs), int(post_s * fs)
    n = n_pre + n_pulse + n_post
    t = np.arange(n) / fs
    command = np.zeros(n)
    command[n_pre:n_pre + n_pulse] = dV
    Rt = Ra + Rm
    tau = Ra * Cm * Rm / Rt * 1e-6  # MOhm*pF = us -> s
    current = np.full(n, holding_pA, dtype=float)
    for edge, step in ((n_pre, dV), (n_pre + n_pulse, -dV)):
        tt = t[edge:] - t[edge]
        current[edge:] += (step / Rt + (step / Ra - step / Rt) * np.exp(-tt / tau)) * 1e3  # mV/MOhm = nA -> pA
    if filter_hz:
        from scipy.signal import bessel, lfilter
        b, a = bessel(4, filter_hz / (fs / 2))
        current = holding_pA + lfilter(b, a, current - holding_pA)
    current = current + rng.normal(0, noise_pA, n) if noise_pA else current
    return t, current, command, tau
