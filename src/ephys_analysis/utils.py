"""Shared helpers: project config, run logging/provenance, safe output writing."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import __version__


# --------------------------------------------------------------------------- config
def find_repo_root(start: Path | str | None = None) -> Path:
    """The project root: the first folder upwards (from ``start`` or the current
    folder) that contains ``pyproject.toml``. Lets notebooks run from their own folder."""
    here = Path(start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    raise FileNotFoundError(f"No pyproject.toml found above {here}")


def load_config(project_dir: Path | str) -> dict:
    """Read ``<project_dir>/config.yaml``. Adds ``_project_dir`` and ``_repo_root``.

    Relative paths in a project config are relative to the repository root
    (the parent of the project folder).
    """
    project_dir = Path(project_dir).resolve()
    with open(project_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_project_dir"] = project_dir
    cfg["_repo_root"] = project_dir.parent
    return cfg


def repo_path(cfg: dict, value: str | Path) -> Path:
    """Resolve a path from the config against the repository root."""
    p = Path(value)
    return p if p.is_absolute() else cfg["_repo_root"] / p


def clear_output(out: Path, keep: tuple[str, ...] = ()) -> None:
    """Delete everything in a subproject ``output/`` folder (derived files only).

    Files open in another program (e.g. Excel) cannot be deleted on Windows; they are
    listed in one clear error instead of failing half-way.
    """
    import shutil

    if not out.exists():
        return
    locked = []
    for item in sorted(out.iterdir()):
        if item.name in keep:
            continue
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
        except PermissionError:
            locked.append(str(item))
    if locked:
        raise PermissionError("Close these files (open in Excel or another program?) and run again:\n  "
                              + "\n  ".join(locked))


# --------------------------------------------------------------------------- provenance
def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip() if out.returncode == 0 else "no_commit"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return commit + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


class RunLog:
    """Prints messages; with ``save=True`` also appends them to ``<output_dir>/log.txt``
    and writes ``run_info.json`` on ``finish``."""

    def __init__(self, cfg: dict, output_dir: Path, save: bool = True):
        self.cfg = cfg
        self.output_dir = output_dir
        self.save = save
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.info = {"run_id": self.run_id, "project": cfg["_project_dir"].name,
                     "package_version": __version__, "git_commit": git_commit(cfg["_repo_root"]),
                     "python": sys.version.split()[0], "platform": platform.platform(),
                     "started": datetime.now().isoformat(timespec="seconds")}
        self("=" * 60)
        self(f"run {self.run_id} | {self.info['project']} | ephys_analysis {__version__} | git {self.info['git_commit']}")

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
        print(line)
        if not self.save:
            return
        with open(self.output_dir / "log.txt", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def finish(self, outputs: list[Path]) -> None:
        """Write ``run_info.json`` (latest run: versions, config, outputs)."""
        if not self.save:
            return
        self.info["finished"] = datetime.now().isoformat(timespec="seconds")
        self.info["outputs"] = [str(p) for p in outputs]
        self.info["config"] = {k: v for k, v in self.cfg.items() if not k.startswith("_")}
        (self.output_dir / "run_info.json").write_text(json.dumps(self.info, indent=2, default=str), encoding="utf-8")


def stamp(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Add ``analysis_run_id`` and ``package_version`` columns."""
    df = df.copy()
    df["analysis_run_id"] = run_id
    df["package_version"] = __version__
    return df


# --------------------------------------------------------------------------- outputs
class OutputExistsError(FileExistsError):
    pass


def check_writable(paths: list[Path], overwrite: bool) -> None:
    """Raise before anything is written if an output exists and overwrite is False."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise OutputExistsError("These outputs already exist (re-run with --overwrite to replace them):\n  "
                                + "\n  ".join(existing))


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- plotting
def envelope(t, v, n_bins: int = 4000):
    """Min/max envelope of a long trace for plotting: ``n_bins`` equal bins, each drawn as
    its minimum and maximum (time-ordered), so every spike stays visible while vector
    figures (SVG) stay small. Returns (t_env, v_env); short traces are returned unchanged."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    if t.size <= 2 * n_bins:
        return t, v
    edges = np.linspace(0, t.size, n_bins + 1).astype(int)
    i_lo = np.array([a + np.argmin(v[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    i_hi = np.array([a + np.argmax(v[a:b]) for a, b in zip(edges[:-1], edges[1:])])
    first_lo = i_lo <= i_hi
    idx = np.where(first_lo[:, None], np.c_[i_lo, i_hi], np.c_[i_hi, i_lo]).ravel()
    return t[idx], v[idx]
