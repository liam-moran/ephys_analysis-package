"""Cell-level summary (08): one row per cell, joining the final values of 03-07.

The cell is the unit of analysis. For each protocol one recording per cell is used, never an
average of several:

    1. a per-cell override in config.yaml (``overrides: {cell_tag: {protocol: abf_filename}}``)
    2. otherwise the first recording, in recording order (file name), that passes the
       protocol's QC rule (``usable``); for membrane tests: qc_status != fail; for all
       protocols: no analysis error and none of ``exclude_if_flag`` (e.g. a baseline recorded
       after a drug washout, flagged ``abf_tags_present``) unless no other recording passes.
Every choice is written to ``selection.csv``: the recording used, the alternatives and why.

The drug timecourse is split by condition (``split_by: condition``): one column set per drug,
e.g. ``tc_5HT_*`` and ``tc_SP_*``, each chosen as above (override key 'timecourse:5-HT').
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def usable(row: pd.Series, protocol: str, exclude_flags: list[str]) -> tuple[bool, str]:
    """(usable, reason) for one recording row of a protocol's output table."""
    flags = str(row.get("flags", "") or "")
    if flags.startswith("error") or str(row.get("analysis_status", "ok")).startswith("error"):
        return False, "analysis error"
    if protocol == "membrane_test" and row.get("qc_status") == "fail":
        return False, "membrane test QC fail"
    hit = [f for f in exclude_flags if f and f in flags]
    if hit:
        return False, f"flag {', '.join(hit)}"
    return True, ""


def select_recordings(table: pd.DataFrame, protocol: str, exclude_flags: list[str],
                      overrides: dict) -> pd.DataFrame:
    """One row per cell: the recording used for ``protocol`` and why."""
    rows = []
    if table.empty:
        return pd.DataFrame(columns=["cell_tag", "protocol", "abf_filename", "recording_id", "n_recordings",
                                     "alternatives", "reason"])
    for cell, g in table.groupby("cell_tag", dropna=False):
        g = g.sort_values("abf_filename")
        status = [(r["abf_filename"], *usable(r, protocol, exclude_flags)) for _, r in g.iterrows()]
        forced = (overrides or {}).get(str(cell), {}).get(protocol)
        if forced:
            chosen = forced if forced in set(g["abf_filename"]) else None
            reason = f"override in config.yaml ({forced})" if chosen else f"override {forced} not found"
        else:
            ok = [f for f, u, _ in status if u]
            if ok:
                chosen = ok[0]
                reason = "only recording" if len(g) == 1 else "first usable recording"
            else:
                soft = [f for f, _, why in status if why.startswith("flag")]
                chosen = soft[0] if soft else None
                reason = (f"no unflagged recording; used flagged {chosen}" if chosen
                          else "no usable recording (" + "; ".join(f"{f}: {why}" for f, _, why in status) + ")")
        alts = "; ".join(f"{f}" + (f" ({why})" if why else "") for f, u, why in status if f != chosen)
        rid = g.loc[g["abf_filename"] == chosen, "recording_id"].iloc[0] if chosen else ""
        rows.append({"cell_tag": cell, "protocol": protocol, "abf_filename": chosen or "", "recording_id": rid,
                     "n_recordings": len(g), "alternatives": alts, "reason": reason})
    return pd.DataFrame(rows)


def expand_protocols(cfg: dict, tables: dict) -> list[tuple[str, dict, pd.DataFrame]]:
    """(protocol name, spec, table) for every protocol. A protocol with ``split_by`` (e.g. the
    drug timecourse, split by ``condition``) becomes one protocol per value, e.g.
    'timecourse:5-HT' with prefix 'tc_5HT'. Within one recording only the FIRST row of a value
    (the first application of that condition) is kept, so the selection stays one recording
    per cell and condition; later repeats remain in the 07 tables."""
    out = []
    for protocol, spec in cfg["protocols"].items():
        table = tables.get(protocol, pd.DataFrame())
        key = spec.get("split_by")
        if not key:
            out.append((protocol, spec, table))
            continue
        if table.empty or key not in table.columns:
            continue
        order = [c for c in ("abf_filename", spec.get("order_by", "")) if c and c in table.columns]
        t = table[table[key].fillna("").astype(str) != ""].sort_values(order)
        for value, g in t.groupby(key, sort=True):
            g = g.drop_duplicates(["cell_tag", "abf_filename"], keep="first")
            slug = re.sub(r"[^0-9A-Za-z_]", "", str(value).replace(" + ", "_")) or "unknown"
            out.append((f"{protocol}:{value}", {**spec, "prefix": f"{spec['prefix']}_{slug}"}, g))
    return out


def build_summary(cells: pd.DataFrame, tables: dict, cfg: dict, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(summary: one row per cell; selection: one row per cell x protocol).

    ``tables``: {protocol: output table of that subproject (or empty)}.
    ``cfg``: the 08 config (protocols: {name: {prefix, columns, exclude_if_flag}},
    cell_metadata_columns, overrides)."""
    meta_cols = [c for c in cfg.get("cell_metadata_columns", []) if c in cells.columns]
    summary = cells[["cell_tag"] + meta_cols].copy()
    selections = []
    for protocol, spec, table in expand_protocols(cfg, tables):
        sel = select_recordings(table, protocol, spec.get("exclude_if_flag", []), cfg.get("overrides", {}))
        selections.append(sel)
        prefix = spec["prefix"]
        if table.empty or "cell_tag" not in table.columns:
            summary[f"{prefix}_recording"] = ""
            summary[f"{prefix}_n_recordings"] = 0
            log(f"{protocol}: no output, left empty")
            continue
        cols = [c for c in spec["columns"] if c in table.columns]
        missing = [c for c in spec["columns"] if c not in table.columns]
        if missing and not table.empty:
            log(f"{protocol}: columns not found and skipped: {', '.join(missing)}")
        used = sel[sel["abf_filename"] != ""][["cell_tag", "abf_filename"]]
        vals = used.merge(table, on=["cell_tag", "abf_filename"], how="left")[["cell_tag", "abf_filename"] + cols]
        vals = vals.rename(columns={"abf_filename": f"{prefix}_recording", **{c: f"{prefix}_{c}" for c in cols}})
        summary = summary.merge(vals, on="cell_tag", how="left")
        n = sel.set_index("cell_tag")["n_recordings"]
        summary[f"{prefix}_n_recordings"] = summary["cell_tag"].map(n).fillna(0).astype(int)
        log(f"{protocol}: {len(used)} of {len(summary)} cells have a recording")
    selection = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return summary, selection


def load_tables(cfg: dict, repo_path) -> dict:
    """Read each protocol's output table (empty DataFrame if the file does not exist yet)."""
    out = {}
    for protocol, spec in cfg["protocols"].items():
        p = repo_path(cfg, spec["table"])
        out[protocol] = pd.read_csv(p, dtype={"cell_tag": "string"}) if Path(p).exists() else pd.DataFrame()
    return out
