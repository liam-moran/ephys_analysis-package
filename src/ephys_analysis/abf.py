"""Read-only ABF access (pyABF): channel roles from units, unit conversion, command waveforms."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ===================================================================== abf_reader
# Read-only access to ABF files via pyABF.
#
# Design rules:
# * Never assume channel index 0 is the membrane voltage, or DAC 0 the active command.
#   Channel roles are inferred from UNITS (names are unreliable: in the example data,
#   current-clamp voltage channels are named "I_MTest 2").
# * Analyses work in standard units (mV, pA, s). Every conversion factor is recorded.
# * pyABF exposes the command waveform of DAC *d* through ``setSweep(channel=d)``; DACs
#   with an index >= the ADC channel count cannot be read this way (reported as such).


VOLTAGE_TO_MV = {"mv": 1.0, "v": 1000.0, "uv": 1e-3, "µv": 1e-3, "μv": 1e-3}
CURRENT_TO_PA = {"pa": 1.0, "na": 1e3, "ua": 1e6, "µa": 1e6, "μa": 1e6, "a": 1e12}


def open_abf(path: Path | str, load_data: bool = True):
    """Open an ABF read-only. pyABF only reads the file; it never writes to it."""
    import pyabf

    return pyabf.ABF(str(path), loadData=load_data)


def unit_kind(unit: str | None) -> str:
    """Return 'voltage', 'current' or 'unknown' for a unit string."""
    u = (unit or "").strip().lower()
    if u in VOLTAGE_TO_MV:
        return "voltage"
    if u in CURRENT_TO_PA:
        return "current"
    return "unknown"


def standard_scale(unit: str | None) -> tuple[float, str]:
    """Factor converting ``unit`` to the analysis unit (mV or pA), and that unit.

    Unknown units return (1.0, 'unknown') so values are kept unchanged and flagged.
    """
    u = (unit or "").strip().lower()
    if u in VOLTAGE_TO_MV:
        return VOLTAGE_TO_MV[u], "mV"
    if u in CURRENT_TO_PA:
        return CURRENT_TO_PA[u], "pA"
    return 1.0, "unknown"


def channel_table(abf) -> list[dict]:
    """One row per ADC and DAC channel with name, unit and inferred kind."""
    rows = []
    for i, (name, unit) in enumerate(zip(abf.adcNames, abf.adcUnits)):
        rows.append({"direction": "ADC", "index": i, "name": name, "unit": unit,
                     "kind": unit_kind(unit)})
    for i, (name, unit) in enumerate(zip(abf.dacNames, abf.dacUnits)):
        rows.append({"direction": "DAC", "index": i, "name": name, "unit": unit,
                     "kind": unit_kind(unit)})
    return rows


def readable_dacs(abf) -> list[int]:
    """DAC indices whose command waveform pyABF can reconstruct."""
    return [d for d in range(len(abf.dacNames)) if d < abf.channelCount]


@dataclass
class DacSummary:
    """Description of one DAC's command waveform across all sweeps (standard units)."""

    dac: int
    name: str
    unit: str
    kind: str
    epoch_types: list[str]
    per_sweep_min: list[float]
    per_sweep_max: list[float]
    is_flat: bool
    has_ramp: bool
    step_amplitudes: list[float]        # largest deviation from the first sample, per sweep
    pulse_durations_s: list[float]      # durations of non-holding plateaus (sweep 0)
    amplitude_varies_across_sweeps: bool

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _plateaus(cmd: np.ndarray, holding: float, tol: float) -> list[tuple[int, int, float]]:
    """Contiguous runs where the command departs from ``holding`` by more than ``tol``."""
    active = np.abs(cmd - holding) > tol
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(int(a), int(b), float(np.median(cmd[a:b]))) for a, b in zip(starts, stops)]


def summarise_dac(abf, dac: int, flat_tol: dict | None = None) -> DacSummary:
    """Summarise the command of one DAC across sweeps, in mV or pA."""
    flat_tol = flat_tol or {"voltage_mV": 0.5, "current_pA": 1.0}
    unit = abf.dacUnits[dac]
    kind = unit_kind(unit)
    scale, _ = standard_scale(unit)
    tol = flat_tol["voltage_mV"] if kind == "voltage" else flat_tol["current_pA"]

    mins, maxs, amps = [], [], []
    durations: list[float] = []
    has_ramp = False
    epoch_types: list[str] = []
    for s in range(abf.sweepCount):
        abf.setSweep(s, channel=dac)
        if s == 0:
            try:
                epoch_types = [str(t) for t in abf.sweepEpochs.types]
            except Exception:
                epoch_types = []
        cmd = np.asarray(abf.sweepC, dtype=float) * scale
        if cmd.size == 0:
            continue
        mins.append(float(cmd.min()))
        maxs.append(float(cmd.max()))
        holding = float(cmd[0])
        dev = cmd - holding
        amps.append(float(dev[np.argmax(np.abs(dev))]))
        if s == 0:
            durations = [(b - a) / abf.dataRate for a, b, _ in _plateaus(cmd, holding, tol)]
        # A ramp: many distinct command values within a single sweep.
        if np.unique(np.round(cmd, 3)).size > 20:
            has_ramp = True
    if any(t.lower() == "ramp" for t in epoch_types):
        has_ramp = True
    span = (max(maxs) - min(mins)) if mins else 0.0
    amp_arr = np.asarray(amps)
    varies = bool(amp_arr.size > 1 and (amp_arr.max() - amp_arr.min()) > tol)
    return DacSummary(dac=dac, name=abf.dacNames[dac], unit=unit, kind=kind,
                      epoch_types=epoch_types, per_sweep_min=mins, per_sweep_max=maxs,
                      is_flat=span <= tol, has_ramp=has_ramp, step_amplitudes=amps,
                      pulse_durations_s=durations, amplitude_varies_across_sweeps=varies)


def first_adc_of_kind(abf, kind: str) -> int | None:
    for i, unit in enumerate(abf.adcUnits):
        if unit_kind(unit) == kind:
            return i
    return None


@dataclass
class SweepData:
    """One sweep of one recorded ADC plus one DAC command, in standard units."""

    sweep: int
    time_s: np.ndarray
    signal: np.ndarray        # mV or pA
    signal_unit: str
    command: np.ndarray       # mV or pA
    command_unit: str


def read_sweeps(abf, adc: int, dac: int) -> list[SweepData]:
    """Read every sweep of ``adc`` together with the command waveform of ``dac``."""
    if dac not in readable_dacs(abf):
        raise ValueError(f"DAC {dac} command cannot be reconstructed by pyABF for this file")
    s_scale, s_unit = standard_scale(abf.adcUnits[adc])
    c_scale, c_unit = standard_scale(abf.dacUnits[dac])
    out = []
    for s in range(abf.sweepCount):
        abf.setSweep(s, channel=dac)
        command = np.asarray(abf.sweepC, dtype=float) * c_scale
        abf.setSweep(s, channel=adc)
        out.append(SweepData(sweep=s, time_s=np.asarray(abf.sweepX, dtype=float),
                             signal=np.asarray(abf.sweepY, dtype=float) * s_scale,
                             signal_unit=s_unit, command=command, command_unit=c_unit))
    return out
