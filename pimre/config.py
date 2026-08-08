"""Configuration management: OUTCAR/KPOINTS parsing, unified config I/O."""

import os
import re

import numpy as np
import yaml

# ── OUTCAR parser ──────────────────────────────────────────────────────

def parse_outcar(path):
    """Extract lattice parameters from VASP OUTCAR.

    Returns (a, b, c, alpha_deg, beta_deg, gamma_deg) or None.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    m = re.search(r"A1\s*=\s*\(([^)]+)\)\s*\n\s*A2\s*=\s*\(([^)]+)\)\s*\n\s*A3\s*=\s*\(([^)]+)\)", text)
    if not m:
        return None
    a1 = np.array([float(x) for x in m.group(1).split(",")])
    a2 = np.array([float(x) for x in m.group(2).split(",")])
    a3 = np.array([float(x) for x in m.group(3).split(",")])
    a, b, c = np.linalg.norm(a1), np.linalg.norm(a2), np.linalg.norm(a3)
    alpha = np.rad2deg(np.arccos(np.dot(a2, a3) / (b * c)))
    beta  = np.rad2deg(np.arccos(np.dot(a1, a3) / (a * c)))
    gamma = np.rad2deg(np.arccos(np.dot(a1, a2) / (a * b)))
    return (round(float(a), 4), round(float(b), 4), round(float(c), 4),
            round(float(alpha), 4), round(float(beta), 4), round(float(gamma), 4))


# ── KPOINTS parser ─────────────────────────────────────────────────────

def parse_kpoints(path):
    """Extract k-grid size from VASP KPOINTS (line-mode).

    Returns (nkx, nky) or None.  The first degenerate segment
    (Gamma→Gamma) is excluded from the count.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.readlines()
    if len(lines) < 4:
        return None
    npts = int(lines[1].strip())
    segments = []
    i = 4
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "":
            i += 1
            continue
        try:
            p0 = [float(x) for x in stripped.split()]
            if len(p0) < 3:
                i += 1
                continue
            i += 1
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            if i >= len(lines):
                break
            p1 = [float(x) for x in lines[i].strip().split()]
            if len(p1) >= 3:
                segments.append((p0, p1))
            i += 1
        except ValueError:
            i += 1
    # Exclude first segment if it is degenerate (start == end)
    n_segments = len(segments)
    if n_segments > 0 and segments[0][0] == segments[0][1]:
        n_segments -= 1
    return (n_segments, npts)


# ── Config I/O ─────────────────────────────────────────────────────────

DEFAULTS = {
    "sample": {"name": "unnamed"},
    "dft": {
        "path": "",
        "k_grid": [20, 20],
        "output_grid": [101, 101],
        "fermi_file": "BAND_GAP",
        "csv_file": "extracted_data.csv",
    },
    "lattice": {"a": 5.8077, "b": 5.8077, "c": 9.1297,
                "alpha": 90.0, "beta": 90.0, "gamma": 120.0},
    "arpes": {
        "path": "",
        "dataset": "",
        "energy":    {"start": 1.06, "delta": -0.01, "npts": 111, "flip": False},
        "kx_angle":  {"start": -20.3729, "delta": 0.0448678, "npts": 700, "flip": False},
        "ky_angle":  {"start": -7.3251, "delta": 1.0, "npts": 36, "flip": False},
        "work_function": 16.03,
    },
    "calibration": {
        "kx_shift": 0.0, "ky_shift": 0.0,
        "kx_grid_shift": 0.0, "ky_grid_shift": 0.0,
        "hsps": {
            "manual": False,
            "rotation_angle": 0.0,
            "scale": 1.0,
        },
    },
    "preprocessing": {
        "output_grid": 200,
        "kd_radius": 0.05,
        "n_rotations": 6,
        "stride": 10,
        "output_path": "",
        "sort_axes": False,
        "sign_correct": False,
        "auto_grid": False,
    },
    "mrf": {
        "bands": [
            {"index": 0, "k_scale": 1.24, "offset": 0.75, "eta": 6.5e-06},
            {"index": 1, "k_scale": 1.24, "offset": 0.80, "eta": 6.5e-09},
            {"index": 2, "k_scale": 1.00, "offset": 0.76, "eta": 0.05},
            {"index": 3, "k_scale": 1.24, "offset": 0.63, "eta": 6.5e-08},
            {"index": 4, "k_scale": 1.24, "offset": 0.62, "eta": 0.0045},
        ],
        "bsfi": {
            "offset_range": 1.0,
            "offset_step": 0.1,
            "fine_tune_range": 0.05,
            "weights": {"correlation": 0.6, "intensity": 0.3, "snr": 0.1},
        },
        "offset_mode": "hierarchical",
        "smooth_sigma": [0.5, 0.5, 0.1],
    },
}


def deep_merge(base, override):
    """Recursively merge override into base dict."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path):
    """Load config from YAML, filling missing keys with defaults."""
    cfg = yaml.safe_load(yaml.dump(DEFAULTS))  # deep copy
    if os.path.exists(path):
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        deep_merge(cfg, user)
    return cfg


def _to_python(obj):
    """Recursively convert numpy types to Python native types."""
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def save_config(cfg, path):
    """Save config dict to YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(_to_python(cfg), f, default_flow_style=False,
                       sort_keys=False, allow_unicode=True)


def crystallographic_data(cfg):
    """Return [a, b, c, alpha, beta, gamma] list from config."""
    lat = cfg["lattice"]
    return [lat["a"], lat["b"], lat["c"],
            lat["alpha"], lat["beta"], lat["gamma"]]
