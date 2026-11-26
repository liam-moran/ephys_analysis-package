"""Cell overview (09): every cell's key properties side by side. Descriptive only.

    heatmap  cells (rows) x features (columns, grouped by area), each feature z-scored
             across cells (after log10 for skewed features), clipped at +-``clip_sd``;
             missing values hatched grey; intrinsic labels in a side strip
    strips   one panel per feature in real units: one dot per cell, colour = firing class,
             marker = depolarisation block, filled / open = rebound; line = median

No PCA / LDA: with the expected ~10 cells there are fewer cells than features, and there
are no experimental groups (see 09_ephys_cell-overview/README.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import save_figure


def feature_table(summary: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values: cells x features in real units, NaN kept; missing report per feature).
    Features not present in the summary are reported as 'column not found'."""
    present = [f for f in features if f in summary.columns]
    values = summary.set_index("cell_tag")[present].apply(pd.to_numeric, errors="coerce")
    rows = []
    for f in features:
        if f not in summary.columns:
            rows.append({"feature": f, "n_cells": 0, "n_missing": len(summary), "note": "column not found"})
        else:
            n_miss = int(values[f].isna().sum())
            rows.append({"feature": f, "n_cells": len(values) - n_miss, "n_missing": n_miss, "note": ""})
    return values, pd.DataFrame(rows)


def zscore_table(values: pd.DataFrame, log_features: list[str] | None = None, log_offset: float = 0.01) -> pd.DataFrame:
    """Z-score each feature across cells (ddof=1), after log10(x + offset) for ``log_features``.
    Features with < 2 values or zero variance become NaN (shown as n/a)."""
    z = pd.DataFrame(index=values.index, columns=values.columns, dtype=float)
    for f in values.columns:
        x = values[f].astype(float)
        if f in (log_features or []):
            x = np.log10(x.clip(lower=0) + log_offset)
        n = int(x.notna().sum())
        sd = float(x.std(ddof=1)) if n >= 2 else np.nan
        if n < 2 or not np.isfinite(sd) or sd == 0:
            continue
        z[f] = (x - x.mean()) / sd
    return z


def cell_labels(summary: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Intrinsic labels per cell (index cell_tag): firing_class, block (bool), rebound (bool)."""
    s = summary.set_index("cell_tag")
    out = pd.DataFrame(index=s.index)
    fc = rules.get("firing_class")
    out["firing_class"] = s[fc].fillna("unknown").astype(str) if fc in s else "unknown"
    blk = rules.get("block_if_not_missing")
    out["block"] = s[blk].notna() if blk in s else False
    rb = rules.get("rebound_if_positive")
    out["rebound"] = pd.to_numeric(s[rb], errors="coerce").fillna(0) > 0 if rb in s else False
    return out


def _order(labels: pd.DataFrame) -> list:
    order = {"silent": 0, "tonic": 1}
    return sorted(labels.index, key=lambda c: (order.get(labels.loc[c, "firing_class"], 2), str(c)))


CLASS_COLOURS = {"silent": "tab:blue", "tonic": "tab:red", "unknown": "0.5"}


def plot_heatmap(z: pd.DataFrame, labels: pd.DataFrame, names: dict, areas: dict, out_stem: Path | None = None,
                 clip_sd: float = 2.5, formats=("png", "svg")):
    """Heatmap of z-scores: rows = cells (silent first), columns = features grouped by area."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = _order(labels)
    cols = list(z.columns)
    data = z.loc[rows, cols].astype(float).to_numpy()
    fig_w = max(8, 0.55 * len(cols) + 4)
    fig_h = max(3.5, 0.35 * len(rows) + 3)
    fig, (ax_l, ax) = plt.subplots(1, 2, figsize=(fig_w, fig_h), gridspec_kw={"width_ratios": [1.2, len(cols)]},
                                   constrained_layout=True)
    im = ax.imshow(np.clip(data, -clip_sd, clip_sd), cmap="RdBu_r", vmin=-clip_sd, vmax=clip_sd, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isfinite(data[i, j]):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.9", hatch="///", edgecolor="0.6", lw=0))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([names.get(c, c) for c in cols], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8)
    # area separators, and each area name centred over its columns
    groups: list[tuple[str, int, int]] = []
    for j, c in enumerate(cols):
        a = areas.get(c, "")
        if groups and groups[-1][0] == a:
            groups[-1] = (a, groups[-1][1], j)
        else:
            groups.append((a, j, j))
    for k, (a, j0, j1) in enumerate(groups):
        if k:
            ax.axvline(j0 - 0.5, color="k", lw=1)
        ax.text((j0 + j1) / 2, -0.6, a, fontsize=7, fontweight="bold", va="bottom", ha="center")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"z-score across cells (clipped at ±{clip_sd:g})")
    # label strip
    lab = labels.loc[rows]
    for i, c in enumerate(rows):
        ax_l.add_patch(Rectangle((0, i - 0.5), 1, 1, color=CLASS_COLOURS.get(lab.loc[c, "firing_class"], "0.5")))
        ax_l.text(1.5, i, "B" if lab.loc[c, "block"] else "", ha="center", va="center", fontsize=8)
        ax_l.text(2.5, i, "R" if lab.loc[c, "rebound"] else "", ha="center", va="center", fontsize=8)
    ax_l.set_xlim(0, 3)
    ax_l.set_ylim(len(rows) - 0.5, -0.5)
    ax_l.set_xticks([0.5, 1.5, 2.5])
    ax_l.set_xticklabels(["class", "block", "rebound"], rotation=60, ha="right", fontsize=8)
    ax_l.set_yticks([])
    for sp in ax_l.spines.values():
        sp.set_visible(False)
    fig.suptitle(f"Cell overview: {len(rows)} cells x {len(cols)} features (hatched = missing; "
                 "class: blue silent, red tonic; B = depolarisation block, R = rebound)", fontsize=9)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats)


def plot_strips(values: pd.DataFrame, labels: pd.DataFrame, names: dict, out_stem: Path | None = None,
                ncols: int = 5, formats=("png", "svg"), seed: int = 0):
    """One panel per feature in real units: dots = cells, colour = firing class, square =
    block, open = no rebound; horizontal line = median."""
    import matplotlib
    if out_stem is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(values.columns)
    nrows = int(np.ceil(len(cols) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.3 * nrows), constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(seed)
    for k, c in enumerate(cols):
        ax = axes[k // ncols][k % ncols]
        x = values[c].astype(float)
        ok = x.notna()
        jit = rng.uniform(-0.15, 0.15, size=len(x))
        for (cell, val), dx in zip(x[ok].items(), jit[ok.to_numpy()]):
            lab = labels.loc[cell]
            col = CLASS_COLOURS.get(lab["firing_class"], "0.5")
            ax.plot(dx, val, "s" if lab["block"] else "o", ms=6, mec=col,
                    mfc=col if lab["rebound"] else "none", mew=1.2)
        if ok.any():
            ax.axhline(float(x[ok].median()), color="k", lw=0.8)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(f"{names.get(c, c)}\n(n = {int(ok.sum())})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    for k in range(len(cols), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle("Cell properties: one dot per cell (blue silent, red tonic; square = block; filled = rebound; "
                 "line = median)", fontsize=9)
    if out_stem is None:
        return fig
    return save_figure(fig, out_stem, formats)
