"""Recording flags: technical QC and acquisition anomalies that must be fixed before analysis.

Everything here only FLAGS. Nothing is changed or removed; fixing belongs to
``02_ephys_anomaly-fix``.

Flag table (long format), one row per recording x flag:
    recording_id | abf_filename | flag | severity | detail | value

``value`` holds the key number behind a flag where there is one (e.g. the median
voltage for a gain error, the bias current parsed from a recording-book comment),
so that 02_ephys_anomaly-fix can act on it without re-parsing text.

Severity:
    needs_fix  acquisition problem that must be corrected before analysis
    review     a person must decide (classification, matching, metadata)
    info       noted, no action required
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .abf import open_abf, standard_scale, unit_kind

FLAG_COLUMNS = ["recording_id", "abf_filename", "flag", "severity", "detail", "value"]


def _flag(rows, rec, flag, severity, detail="", value=np.nan):
    rows.append({"recording_id": rec.get("recording_id", ""), "abf_filename": rec.get("abf_filename", ""),
                 "flag": flag, "severity": severity, "detail": detail, "value": value})


# --------------------------------------------------------------------------- signal checks
def signal_flags(signals: list[np.ndarray], kinds: list[str], names: list[str], cfg: dict) -> list[tuple]:
    """Check recorded signals (standard units: mV / pA), one array per ADC.

    Returns (flag, severity, detail, value) tuples for:

    * ``duplicated_voltage_no_current``: current-clamp file with no current-unit
      channel, where two voltage channels carry the same signal (|r| >= threshold).
      The injected current was therefore not recorded.
    * ``possible_gain_error``: the primary voltage channel lies outside the
      physiological range (e.g. ~20x scaling). An approximate factor is reported.
    """
    a = cfg["anomalies"]
    out = []
    voltage_ch = [i for i, k in enumerate(kinds) if k == "voltage"]
    current_clamp = bool(kinds) and kinds[0] == "voltage"
    if not current_clamp:
        return out

    if "current" not in kinds and len(voltage_ch) >= 2:
        v0 = signals[voltage_ch[0]]
        for j in voltage_ch[1:]:
            vj = signals[j]
            if v0.std() > 0 and vj.std() > 0:
                r = float(np.corrcoef(v0, vj)[0, 1])
                if abs(r) >= a["duplicate_min_correlation"]:
                    slope, offset = np.polyfit(v0, vj, 1)
                    out.append(("duplicated_voltage_no_current", "needs_fix",
                                f"ADC{voltage_ch[0]} '{names[voltage_ch[0]]}' and ADC{j} '{names[j]}' "
                                f"both carry voltage (r={r:.3f}, slope={slope:.2f}, offset={offset:.1f} mV); "
                                f"no current channel recorded", r))

    v = signals[voltage_ch[0]]
    med = float(np.median(v))
    p1, p99 = np.percentile(v, [1, 99])
    lo, hi = a["voltage_median_range_mV"]
    if not (lo <= med <= hi) or (p99 - p1) > a["voltage_max_span_mV"]:
        factor = med / a["typical_rmp_mV"]
        out.append(("possible_gain_error", "needs_fix",
                    f"ADC{voltage_ch[0]} median {med:.1f} mV (1-99 %: {p1:.1f} to {p99:.1f} mV), "
                    f"expected median {lo} to {hi} mV; roughly {factor:.1f}x a typical RMP", med))
    return out


def channel_signal_checks(abf_path, cfg: dict, max_samples: int = 2_000_000) -> list[tuple]:
    """Load the first sweep of every ADC (up to ``max_samples``) and run ``signal_flags``."""
    abf = open_abf(abf_path)
    signals = []
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        scale, _ = standard_scale(abf.adcUnits[ch])
        signals.append(np.asarray(abf.sweepY[:max_samples], dtype=float) * scale)
    return signal_flags(signals, [unit_kind(u) for u in abf.adcUnits], list(abf.adcNames), cfg)


# --------------------------------------------------------------------------- header checks
def _json_list(value) -> list:
    try:
        return list(json.loads(value)) if isinstance(value, str) and value else []
    except (ValueError, TypeError):
        return []


def header_flags(rec: dict, clamp_mode: str, family: str, command_dacs: list[int], cfg: dict) -> list[tuple]:
    """Compare amplifier/Clampex header settings with what the protocol should look like.

    * ``telegraph_off``: Clampex did not receive gains/mode from the amplifier.
    * ``telegraph_mode_mismatch``: amplifier mode disagrees with the mode inferred from units.
    * ``unexpected_command_sensitivity``: command DAC scale differs from the expected
      MultiClamp setting (current clamp pA/V, voltage clamp mV/V).
    * ``baseline_not_zero_current``: a baseline protocol (no current manipulation) whose
      Clampex holding level or command is not 0 pA.
    """
    e = cfg["expected"]
    out = []
    p = cfg["classification"].get("primary_adc_index", 0)
    enabled = _json_list(rec.get("telegraph_enabled"))
    modes = _json_list(rec.get("telegraph_mode"))
    scales = _json_list(rec.get("dac_scale_factor"))
    holding = _json_list(rec.get("dac_holding_level"))
    dac_kinds = str(rec.get("dac_kinds", "")).split(" | ")

    if enabled and p < len(enabled):
        if not enabled[p]:
            out.append(("telegraph_off", "review",
                        "amplifier telegraph disabled: gains and clamp mode were not confirmed by the amplifier", np.nan))
        elif modes and p < len(modes):
            tel_mode = e["telegraph_modes"].get(int(modes[p]), "unknown")
            if clamp_mode in ("voltage_clamp", "current_clamp") and tel_mode != clamp_mode:
                out.append(("telegraph_mode_mismatch", "review",
                            f"amplifier reported {tel_mode} (mode {int(modes[p])}), channel units indicate {clamp_mode}",
                            modes[p]))

    for d in command_dacs:
        if d >= len(scales) or d >= len(dac_kinds):
            continue
        kind = dac_kinds[d]
        expected = {"current": e["cc_command_sensitivity_pA_per_V"],
                    "voltage": e["vc_command_sensitivity_mV_per_V"]}.get(kind)
        if expected is not None and abs(scales[d] - expected) > 1e-6 * max(1.0, expected):
            unit = "pA/V" if kind == "current" else "mV/V"
            out.append(("unexpected_command_sensitivity", "review",
                        f"DAC{d} command sensitivity {scales[d]:g} {unit}, expected {expected:g} {unit} "
                        f"({e['amplifier']})", scales[d]))

    if family in e["baseline_families"]:
        try:
            dacs = json.loads(rec.get("dac_summary_json") or "{}")
        except (ValueError, TypeError):
            dacs = {}
        tol = cfg["classification"]["rules"]["flat_command_tolerance"]["current_pA"]
        for key, s in (dacs.items() if isinstance(dacs, dict) else []):
            d = int(key)
            if s.get("kind") != "current" or "error" in s:
                continue
            hold = holding[d] if d < len(holding) else 0.0
            lo, hi = min(s.get("per_sweep_min", [0])), max(s.get("per_sweep_max", [0]))
            if abs(hold) > tol or abs(lo) > tol or abs(hi) > tol:
                out.append(("baseline_not_zero_current", "needs_fix",
                            f"baseline protocol should inject 0 pA; DAC{d} holding {hold:g} pA, "
                            f"command range {lo:g} to {hi:g} pA", hold if abs(hold) > tol else lo))
    return out


# --------------------------------------------------------------------------- recording-book checks
def parse_current_comment(text, patterns: dict) -> tuple[bool, float]:
    """Does a recording-book comment mention injected/holding/bias current?

    Returns (mentioned, value_pA). ``value_pA`` is the first number followed by pA
    (e.g. "-10pA holding" -> -10.0), or NaN when only a keyword was found.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return False, np.nan
    s = str(text)
    m = re.search(patterns["value_regex"], s, flags=re.IGNORECASE)
    if m:
        return True, float(m.group(1).replace(" ", ""))
    if any(k in s.lower() for k in patterns["keywords"]):
        return True, np.nan
    return False, np.nan


def book_flags(rec: dict, clamp_mode: str, cfg: dict) -> list[tuple]:
    """Compare recording-book entries with the ABF.

    * ``book_mentions_current``: in current clamp, a comment mentions injected / holding /
      bias current that the ABF cannot show (amplifier-set bias is not recorded).
    * ``book_date_mismatch``: 'Date Recorded' differs from the ABF recording date.
    """
    e = cfg["expected"]
    out = []
    if clamp_mode == "current_clamp":
        for col in e["book_comment_columns"]:
            mentioned, value = parse_current_comment(rec.get(col), e["current_comment"])
            if mentioned:
                out.append(("book_mentions_current", "review",
                            f"{col}: '{rec.get(col)}' (current not visible in the ABF)", value))
                break
    from .metadata import book_columns
    book_date, abf_date = rec.get(book_columns(cfg)["date"]), rec.get("recording_start")
    if isinstance(book_date, str) or not pd.isna(book_date):
        try:
            b = pd.to_datetime(book_date).date()
            a = pd.to_datetime(abf_date).date()
            if b != a:
                out.append(("book_date_mismatch", "review", f"book {b}, ABF {a}", np.nan))
        except (ValueError, TypeError):
            pass
    return out


# --------------------------------------------------------------------------- full flag table
def build_flags(recordings: pd.DataFrame, classification: pd.DataFrame, book_issues: pd.DataFrame,
                unmatched_book: pd.DataFrame, cfg: dict, log=print) -> pd.DataFrame:
    """Combine all flags for the meta-org project into one long table."""
    rows: list[dict] = []
    cls = classification.set_index("recording_id")
    for _, r in recordings.iterrows():
        rec = r.to_dict()
        if rec.get("read_status") != "ok":
            _flag(rows, rec, "unreadable_file", "needs_fix", str(rec.get("read_error")))
            continue
        if "unknown" in str(rec.get("adc_kinds", "")).split(" | "):
            _flag(rows, rec, "unknown_channel_units", "review", str(rec.get("adc_units")))
        sr = float(rec.get("sample_rate_Hz") or 0)
        if sr < cfg["anomalies"]["min_sample_rate_Hz"]:
            _flag(rows, rec, "low_sample_rate", "review", f"{sr:g} Hz")
        if rec.get("match_status") != "matched":
            _flag(rows, rec, f"book_{rec.get('match_status')}", "review", str(rec.get("match_note") or ""))
        if rec.get("cell_tag_source") != "recording_book":
            _flag(rows, rec, "missing_cell_tag", "review",
                  f"filename candidate: {rec.get('cell_tag_candidate_from_filename')}")
        c = cls.loc[rec["recording_id"]]
        if c["classification_status"] in ("ambiguous", "mismatch", "unsupported"):
            _flag(rows, rec, f"protocol_{c['classification_status']}", "review", str(c["review_reason"]))
        elif c["review_reason"] and c["classification_status"] == "confirmed":
            _flag(rows, rec, "protocol_note", "info", str(c["review_reason"]))
        dacs = [int(d) for d in str(c["command_dac"]).split(",") if str(d).strip().isdigit()]
        checks = header_flags(rec, c["clamp_mode"], c["abf_family"], dacs, cfg) + book_flags(rec, c["clamp_mode"], cfg)
        try:
            checks += channel_signal_checks(rec["original_path"], cfg)
        except Exception as exc:
            checks.append(("signal_check_failed", "review", repr(exc), np.nan))
        for flag, sev, detail, value in checks:
            _flag(rows, rec, flag, sev, detail, value)

    for _, i in book_issues.iterrows():
        _flag(rows, {}, "recording_book_" + str(i["issue"]), "review",
              f"sheet {i['sheet']}, row {i['row']}, column {i['column']}: {i['value']}")
    if len(unmatched_book):
        names = ", ".join(map(str, unmatched_book.iloc[:, 0].head(5)))
        _flag(rows, {}, "book_rows_without_abf", "info",
              f"{len(unmatched_book)} Recording Log rows have no ABF in abf_root (e.g. {names})")
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


def recording_status(flags: pd.DataFrame, recording_ids: pd.Series) -> pd.Series:
    """Per recording: 'needs_fix' > 'review' > 'ok' (worst flag wins)."""
    rank = {"ok": 0, "review": 1, "needs_fix": 2}
    worst: dict[str, str] = {}
    for rid, sev in zip(flags["recording_id"], flags["severity"]):
        if rid and sev in rank and rank[sev] > rank[worst.get(rid, "ok")]:
            worst[rid] = sev
    return recording_ids.map(lambda rid: worst.get(rid, "ok"))


def flag_summary(flags: pd.DataFrame, recording_ids: pd.Series) -> pd.Series:
    g = flags[flags["recording_id"] != ""].groupby("recording_id")["flag"].apply(lambda s: "; ".join(s))
    return recording_ids.map(g).fillna("")

