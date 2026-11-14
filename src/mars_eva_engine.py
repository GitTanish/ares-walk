#!/usr/bin/env python3
"""
ARES-WALK :: 0.38 g suited EVA routing engine
=============================================

Builds a calibrated boxwork digital twin for the MSL/Curiosity traverse across
sols 4535-4611, extracts TPI ridges, models THEMIS-style thermal-inertia sand
penalties and CRISM D2300 mineralogy, ray-traces RF line-of-sight, and plans an
asymmetric, energy-minimising EVA route that preserves a strict 25 % PLSS
walkback reserve.

The terrain fields are a *scenario twin*: real Curiosity waypoints constrain the
elevation trend and every route endpoint; the fine-scale boxwork relief,
thermal inertia, D2300 and RF viewshed are physically plausible model fields.
Drop real HiRISE/THEMIS/CRISM GeoTIFFs into data/raw/ and pds_ingest.py will use
the raster crop instead.

Run:
    python src/mars_eva_engine.py
to regenerate assets/mount_sharp_eva_plan.png from data/mission_bundle.json
(or from the built-in verified waypoint fallback if the bundle is absent).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
BUNDLE_PATH = DATA_DIR / "mission_bundle.json"
PLOT_PATH = ASSETS_DIR / "mount_sharp_eva_plan.png"

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

GRID_W = 72
GRID_H = 72
CELL_M = 15.0
MARGIN_M = 320.0
SEED = 20250513

G_MARS = 3.721
G_EARTH = 9.80665
GRAVITY_RATIO = G_MARS / G_EARTH

PHYSICS: dict[str, Any] = {
    "gravity_m_s2": G_MARS,
    "earth_gravity_m_s2": G_EARTH,
    "gravity_ratio": round(GRAVITY_RATIO, 5),
    "astronaut_body_kg": 78.0,
    "suit_plss_kg": 52.0,
    "walk_speed_flat_m_s": 1.15,
    "sprint_multiplier": 1.24,
    "suit_metabolic_overhead": 1.10,
    "plss_energy_budget_wh": 400.0,
    "walkback_reserve_fraction": 0.25,
    "dwell_seconds_per_station": 180.0,
    "dwell_metabolic_w": 150.0,
    "o2_capacity_kg": 0.32,
    "o2_metabolic_mj_per_kg": 13.5,
    "water_capacity_kg": 1.40,
    "water_rate_kg_per_min": 0.0032,
    "spe_dose_rate_msv_h": 85.0,
    "gcr_dose_rate_msv_h": 0.0096,
    "acute_dose_limit_msv": 50.0,
    "simulation_time_scale": 12.0,
    "hazard_slope_deg": 20.5,
    "hazard_coregistration_m": 2.0,
    "comm_energy_penalty": 0.05,
}

SCENARIO: dict[str, Any] = {
    "name": "Altadena Boxwork EVA",
    "short_name": "ALT-BXW-01",
    "sol_window": [4535, 4611],
    "gravity_label": "0.38 g",
    "base_name": "PRESSURIZED ROVER / SHELTER",
    "terrain_mode": "calibrated_digital_twin",
    "notes": [
        "Real rover waypoints and Mastcam frames; terrain physics fields are a modeled twin.",
        "PLSS energy is a metabolic-energy-equivalent consumable budget, not a battery state-of-charge.",
        "Acute RAD dose is an operational scenario, not a mission medical forecast.",
    ],
}

# Real final-drive MSL waypoints, sols 4535-4611, from the NASA/JPL MMGIS
# MSL_waypoints layer. Easting/northing are JPL map metres; lat/lon are Mars
# planetographic degrees. This is also the offline fallback for pds_ingest.py.
FALLBACK_WAYPOINTS: list[dict[str, Any]] = [
    {"sol": 4537, "site": 116, "drive": 156, "easting": 8143525.451, "northing": -284387.362, "elev_geoid": -3655.635, "lat": -4.797787, "lon": 137.386200, "dist_m": 18.6},
    {"sol": 4539, "site": 116, "drive": 408, "easting": 8143508.892, "northing": -284389.092, "elev_geoid": -3654.970, "lat": -4.797816, "lon": 137.385921, "dist_m": 22.4},
    {"sol": 4541, "site": 116, "drive": 414, "easting": 8143509.198, "northing": -284389.083, "elev_geoid": -3654.970, "lat": -4.797816, "lon": 137.385926, "dist_m": 0.3},
    {"sol": 4544, "site": 116, "drive": 576, "easting": 8143493.652, "northing": -284397.630, "elev_geoid": -3654.481, "lat": -4.797960, "lon": 137.385664, "dist_m": 19.9},
    {"sol": 4546, "site": 116, "drive": 870, "easting": 8143454.896, "northing": -284415.721, "elev_geoid": -3653.485, "lat": -4.798265, "lon": 137.385010, "dist_m": 45.4},
    {"sol": 4548, "site": 116, "drive": 1260, "easting": 8143397.716, "northing": -284447.000, "elev_geoid": -3650.774, "lat": -4.798793, "lon": 137.384045, "dist_m": 69.2},
    {"sol": 4552, "site": 116, "drive": 1596, "easting": 8143369.718, "northing": -284475.277, "elev_geoid": -3648.368, "lat": -4.799270, "lon": 137.383573, "dist_m": 42.5},
    {"sol": 4554, "site": 116, "drive": 1806, "easting": 8143361.680, "northing": -284501.604, "elev_geoid": -3645.732, "lat": -4.799714, "lon": 137.383437, "dist_m": 29.3},
    {"sol": 4558, "site": 116, "drive": 1890, "easting": 8143365.353, "northing": -284507.384, "elev_geoid": -3646.638, "lat": -4.799812, "lon": 137.383499, "dist_m": 7.3},
    {"sol": 4560, "site": 116, "drive": 1944, "easting": 8143367.862, "northing": -284509.493, "elev_geoid": -3647.181, "lat": -4.799847, "lon": 137.383541, "dist_m": 3.8},
    {"sol": 4571, "site": 116, "drive": 2214, "easting": 8143334.999, "northing": -284542.666, "elev_geoid": -3643.934, "lat": -4.800407, "lon": 137.382987, "dist_m": 48.0},
    {"sol": 4573, "site": 116, "drive": 2562, "easting": 8143307.139, "northing": -284587.534, "elev_geoid": -3644.188, "lat": -4.801164, "lon": 137.382517, "dist_m": 53.5},
    {"sol": 4575, "site": 117, "drive": 0, "easting": 8143283.206, "northing": -284631.667, "elev_geoid": -3644.423, "lat": -4.801908, "lon": 137.382113, "dist_m": 55.1},
    {"sol": 4578, "site": 117, "drive": 168, "easting": 8143259.032, "northing": -284632.984, "elev_geoid": -3645.460, "lat": -4.801931, "lon": 137.381705, "dist_m": 24.5},
    {"sol": 4580, "site": 117, "drive": 420, "easting": 8143232.113, "northing": -284656.443, "elev_geoid": -3643.912, "lat": -4.802326, "lon": 137.381251, "dist_m": 37.1},
    {"sol": 4582, "site": 117, "drive": 492, "easting": 8143232.195, "northing": -284657.622, "elev_geoid": -3643.818, "lat": -4.802346, "lon": 137.381253, "dist_m": 4.0},
    {"sol": 4585, "site": 117, "drive": 702, "easting": 8143220.363, "northing": -284670.749, "elev_geoid": -3644.110, "lat": -4.802568, "lon": 137.381053, "dist_m": 21.6},
    {"sol": 4587, "site": 117, "drive": 840, "easting": 8143204.780, "northing": -284680.734, "elev_geoid": -3643.500, "lat": -4.802736, "lon": 137.380790, "dist_m": 18.5},
    {"sol": 4588, "site": 117, "drive": 1008, "easting": 8143193.222, "northing": -284662.565, "elev_geoid": -3645.303, "lat": -4.802430, "lon": 137.380595, "dist_m": 23.4},
    {"sol": 4591, "site": 117, "drive": 1122, "easting": 8143184.610, "northing": -284667.312, "elev_geoid": -3644.620, "lat": -4.802510, "lon": 137.380450, "dist_m": 10.6},
    {"sol": 4594, "site": 117, "drive": 1434, "easting": 8143183.545, "northing": -284694.179, "elev_geoid": -3642.423, "lat": -4.802963, "lon": 137.380432, "dist_m": 34.6},
    {"sol": 4596, "site": 117, "drive": 1890, "easting": 8143162.349, "northing": -284732.280, "elev_geoid": -3640.497, "lat": -4.803606, "lon": 137.380074, "dist_m": 60.6},
    {"sol": 4598, "site": 117, "drive": 2166, "easting": 8143143.461, "northing": -284756.596, "elev_geoid": -3635.866, "lat": -4.804016, "lon": 137.379756, "dist_m": 34.7},
    {"sol": 4600, "site": 117, "drive": 2370, "easting": 8143139.321, "northing": -284766.850, "elev_geoid": -3632.878, "lat": -4.804189, "lon": 137.379686, "dist_m": 21.2},
    {"sol": 4602, "site": 117, "drive": 2490, "easting": 8143135.643, "northing": -284777.388, "elev_geoid": -3630.754, "lat": -4.804367, "lon": 137.379624, "dist_m": 13.1},
    {"sol": 4605, "site": 117, "drive": 2538, "easting": 8143131.699, "northing": -284781.339, "elev_geoid": -3631.479, "lat": -4.804433, "lon": 137.379557, "dist_m": 5.3},
    {"sol": 4609, "site": 117, "drive": 2730, "easting": 8143117.662, "northing": -284799.519, "elev_geoid": -3624.329, "lat": -4.804740, "lon": 137.379320, "dist_m": 24.1},
]

DEFAULT_STATION_SPECS: list[dict[str, Any]] = [
    {"id": "ALT-01", "sol": 4539, "name": "Boxwork Gate", "science_target": "TPI ridge crest / fracture-margin reconnaissance"},
    {"id": "ALT-02", "sol": 4546, "name": "Sand Apron", "science_target": "Low-TI sand sheet / trafficability and slip risk"},
    {"id": "ALT-03", "sol": 4554, "name": "Ridge Shoulder", "science_target": "Boxwork flank / slope and RF horizon geometry"},
    {"id": "ALT-04", "sol": 4560, "name": "Nodule Bench", "science_target": "Diagenetic nodule float / D2300 context"},
    {"id": "ALT-05", "sol": 4571, "name": "Layered Spur", "science_target": "Laminated mudstone bench / stratigraphic section"},
    {"id": "ALT-06", "sol": 4580, "name": "Hematite Shelf", "science_target": "Iron-oxide bearing pavement / thermal contrast"},
    {"id": "ALT-07", "sol": 4588, "name": "Clay Lens", "science_target": "D2300 phyllosilicate candidate / sample cache"},
    {"id": "ALT-08", "sol": 4596, "name": "Sulfate Rise", "science_target": "Sulfate-bearing outcrop / ridge traverse"},
    {"id": "ALT-09", "sol": 4602, "name": "Boxwork Crest", "science_target": "High-TPI crest / comm relay sightline"},
    {"id": "ALT-10", "sol": 4609, "name": "Upper Bench", "science_target": "Upper boxwork bench / ascent turnaround"},
]


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------

def harmonic_field(
    shape: tuple[int, int],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    count: int = 7,
    wavelength_m: tuple[float, float] = (70.0, 360.0),
) -> np.ndarray:
    """Deterministic sum of plane waves, normalised to roughly [-1, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype=float)
    for _ in range(count):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        wl = rng.uniform(*wavelength_m)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.35, 1.0)
        out += amp * np.sin(
            2.0 * math.pi * (x * math.cos(theta) + y * math.sin(theta)) / wl + phase
        )
    scale = np.max(np.abs(out))
    return out / scale if scale else out


def idw_grid(
    xs: Sequence[float],
    ys: Sequence[float],
    values: Sequence[float],
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    power: float = 2.0,
    smoothing_m2: float = 140.0,
    neighbours: int = 9,
) -> np.ndarray:
    """Inverse-distance interpolation of waypoint residuals across the grid."""
    points = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    query = np.column_stack([gx.ravel(), gy.ravel()])
    k = min(neighbours, len(points))
    tree = cKDTree(points)
    dist, idx = tree.query(query, k=k)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    weights = 1.0 / (np.power(dist, power) + smoothing_m2)
    vals = np.asarray(values, float)[idx]
    result = (weights * vals).sum(axis=1) / weights.sum(axis=1)
    return result.reshape(gx.shape)


def distance_to_polyline(
    px: Sequence[float],
    py: Sequence[float],
    gx: np.ndarray,
    gy: np.ndarray,
) -> np.ndarray:
    """Shortest 2-D distance (m) from every grid cell to a polyline."""
    best = np.full(gx.shape, np.inf)
    pts_x = np.asarray(px, float)
    pts_y = np.asarray(py, float)
    for k in range(len(pts_x) - 1):
        ax, ay = pts_x[k], pts_y[k]
        bx, by = pts_x[k + 1], pts_y[k + 1]
        vx, vy = bx - ax, by - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-9:
            d = (gx - ax) ** 2 + (gy - ay) ** 2
        else:
            t = np.clip(((gx - ax) * vx + (gy - ay) * vy) / vv, 0.0, 1.0)
            d = (gx - (ax + t * vx)) ** 2 + (gy - (ay + t * vy)) ** 2
        best = np.minimum(best, d)
    return np.sqrt(np.maximum(best, 0.0))


def project_track(track: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], float, float, float, float]:
    """Add local x/y map metres to waypoint dicts (JPL easting/northing)."""
    if not track:
        raise ValueError("track is empty")
    has_enu = all("easting" in p and "northing" in p for p in track)
    if has_enu:
        e0 = min(float(p["easting"]) for p in track)
        n0 = min(float(p["northing"]) for p in track)
        projected = []
        for p in track:
            q = dict(p)
            q["x"] = float(p["easting"]) - e0
            q["y"] = float(p["northing"]) - n0
            projected.append(q)
    else:
        e0 = n0 = 0.0
        projected = [dict(p) for p in track]
        for q in projected:
            q["x"] = float(q["x"])
            q["y"] = float(q["y"])
    xs = [float(p["x"]) for p in projected]
    ys = [float(p["y"]) for p in projected]
    return projected, e0, n0, min(xs), min(ys)


def cell_center(x0: float, y0: float, i: int, j: int) -> tuple[float, float]:
    return x0 + (i + 0.5) * CELL_M, y0 + (j + 0.5) * CELL_M


def nearest_cell(x: float, y: float, x0: float, y0: float) -> tuple[int, int]:
    i = int(np.clip(round((x - x0) / CELL_M - 0.5), 0, GRID_W - 1))
    j = int(np.clip(round((y - y0) / CELL_M - 0.5), 0, GRID_H - 1))
    return i, j


def _frange(start: float, end: float, step: float) -> Iterable[float]:
    value = start
    while value < end:
        yield value
        value += step


# ---------------------------------------------------------------------------
# Terrain twin
# ---------------------------------------------------------------------------

def _line_of_sight(
    elevation: np.ndarray,
    base_i: int,
    base_j: int,
    target_i: int,
    target_j: int,
    *,
    base_antenna_m: float = 4.5,
    suit_antenna_m: float = 1.8,
    clearance_m: float = 0.55,
) -> bool:
    """Ray-trace a terrain line-of-sight between two antenna tips."""
    x1 = (base_i + 0.5) * CELL_M
    y1 = (base_j + 0.5) * CELL_M
    x2 = (target_i + 0.5) * CELL_M
    y2 = (target_j + 0.5) * CELL_M
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1e-6:
        return True
    z1 = float(elevation[base_j, base_i]) + base_antenna_m
    z2 = float(elevation[target_j, target_i]) + suit_antenna_m
    samples = max(12, int(dist / (CELL_M / 5.0)))
    ts = np.linspace(0.0, 1.0, samples + 1)[1:-1]
    xx = x1 + (x2 - x1) * ts
    yy = y1 + (y2 - y1) * ts
    ii = np.clip((xx / CELL_M).astype(int), 0, GRID_W - 1)
    jj = np.clip((yy / CELL_M).astype(int), 0, GRID_H - 1)
    ray_z = z1 + (z2 - z1) * ts
    terrain_z = elevation[jj, ii]
    return bool(np.all(terrain_z + clearance_m <= ray_z))


# Three portable ridge repeaters close the worst comm-shadow pockets around
# the boxwork. They are scenario infrastructure, not real landed hardware.
RELAY_SPECS: list[dict[str, Any]] = [
    {"id": "R1", "name": "North Ridge Repeater", "i": 30, "j": 51, "mast_m": 12.0},
    {"id": "R2", "name": "Southwest Repeater", "i": 10, "j": 14, "mast_m": 12.0},
    {"id": "R3", "name": "Massif Repeater", "i": 68, "j": 23, "mast_m": 12.0},
]


def _build_rf_viewshed(
    elevation: np.ndarray,
    base_i: int,
    base_j: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Union line-of-sight for the base mast and three ridge repeaters."""
    visible = np.zeros_like(elevation, dtype=bool)
    for j in range(GRID_H):
        for i in range(GRID_W):
            visible[j, i] = _line_of_sight(elevation, base_i, base_j, i, j)
    visible[base_j, base_i] = True
    relays: list[dict[str, Any]] = []
    for relay in RELAY_SPECS:
        ri, rj = int(relay["i"]), int(relay["j"])
        relay_visible = np.zeros_like(elevation, dtype=bool)
        for j in range(GRID_H):
            for i in range(GRID_W):
                relay_visible[j, i] = _line_of_sight(
                    elevation, ri, rj, i, j, base_antenna_m=float(relay["mast_m"])
                )
        relay_visible[rj, ri] = True
        visible |= relay_visible
        relays.append({
            **relay,
            "x": round((ri + 0.5) * CELL_M, 2),
            "y": round((rj + 0.5) * CELL_M, 2),
            "solo_visible_fraction": round(float(relay_visible.mean()), 3),
        })
    return visible, relays


def build_terrain(
    track: Sequence[dict[str, Any]],
    base_xy: tuple[float, float],
    station_xy: Sequence[tuple[float, float]],
    *,
    seed: int = SEED,
    raster_layers: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Create the 72 x 72 boxwork twin and quantised JSON fields.

    raster_layers may provide ``elevation``, ``thermal_inertia`` and ``d2300``
    arrays already resampled to GRID_H x GRID_W. Real rasters replace the
    corresponding synthetic field; the remaining fields stay calibrated.
    """
    raster_layers = raster_layers or {}
    projected, e0, n0, min_x, min_y = project_track(track)
    x0 = min_x - MARGIN_M
    y0 = min_y - MARGIN_M
    domain = GRID_W * CELL_M
    if max(p["x"] for p in projected) > x0 + domain or max(p["y"] for p in projected) > y0 + domain:
        raise ValueError("waypoint track does not fit inside the configured grid")

    xs = x0 + (np.arange(GRID_W) + 0.5) * CELL_M
    ys = y0 + (np.arange(GRID_H) + 0.5) * CELL_M
    gx, gy = np.meshgrid(xs, ys)

    tr_x = np.array([float(p["x"]) for p in projected])
    tr_y = np.array([float(p["y"]) for p in projected])
    tr_z = np.array([float(p["elev_geoid"]) for p in projected])

    # Low-order elevation trend fitted to the real Curiosity waypoints.
    design = np.column_stack([tr_x, tr_y, np.ones_like(tr_x)])
    coef, *_ = np.linalg.lstsq(design, tr_z, rcond=None)
    trend = coef[0] * gx + coef[1] * gy + coef[2]
    residual = tr_z - (coef[0] * tr_x + coef[1] * tr_y + coef[2])
    residual_grid = idw_grid(tr_x, tr_y, residual, gx, gy)
    residual_grid = np.clip(residual_grid, -11.0, 11.0)

    d_track = distance_to_polyline(tr_x, tr_y, gx, gy)
    corridor = 1.0 - 0.86 * np.exp(-np.square(d_track / 28.0))

    noise_a = harmonic_field((GRID_H, GRID_W), gx, gy, seed + 11, wavelength_m=(105.0, 330.0))
    noise_b = harmonic_field((GRID_H, GRID_W), gx, gy, seed + 12, wavelength_m=(90.0, 290.0))
    noise_c = harmonic_field((GRID_H, GRID_W), gx, gy, seed + 13, wavelength_m=(55.0, 210.0))

    # Two intersecting ridge sets create the cellular boxwork texture.
    theta_a = math.radians(24.0)
    theta_b = math.radians(112.0)
    erosion = harmonic_field(
        (GRID_H, GRID_W), gx, gy, seed + 15, wavelength_m=(180.0, 520.0)
    )
    u_a = (
        gx * math.cos(theta_a)
        + gy * math.sin(theta_a)
        + 34.0 * noise_a
        + 8.0 * noise_c
    )
    u_b = gx * math.cos(theta_b) + gy * math.sin(theta_b) + 30.0 * noise_b
    crest_a = np.exp(-(1.0 - np.cos(2.0 * math.pi * u_a / 96.0)) / 0.23)
    crest_b = np.exp(-(1.0 - np.cos(2.0 * math.pi * u_b / 84.0)) / 0.23)
    boxwork = (5.8 * crest_a + 4.6 * crest_b + 2.4 * crest_a * crest_b) * corridor
    boxwork *= 0.52 + 0.48 * np.clip((560.0 - gx) / 520.0, 0.0, 1.0)
    boxwork *= 0.68 + 0.32 * np.clip(0.5 + 0.5 * erosion, 0.0, 1.0)

    # Distant Mount Sharp flank (east), a relay knoll, and fine aeolian relief.
    massif = 88.0 * np.exp(
        -(((gx - 900.0) ** 2 + (gy - 120.0) ** 2) / (2.0 * 185.0**2))
    )
    knoll = 14.0 * np.exp(
        -(((gx - 150.0) ** 2 + (gy - 455.0) ** 2) / (2.0 * 82.0**2))
    )
    fine_relief = 0.62 * noise_c + 0.26 * harmonic_field(
        (GRID_H, GRID_W), gx, gy, seed + 14, wavelength_m=(35.0, 110.0)
    )

    # Scattered metre-scale boulders: authentic trafficability hazards and a
    # useful test of the +/-2 m co-registration buffer.
    rng = np.random.default_rng(seed)
    boulder_bump = np.zeros_like(gx)
    boulder_cores: list[tuple[float, float, float]] = []
    attempts = 0
    while len(boulder_cores) < 16 and attempts < 400:
        attempts += 1
        bx = rng.uniform(x0 + 55.0, x0 + domain - 55.0)
        by = rng.uniform(y0 + 55.0, y0 + domain - 55.0)
        local_d = float(distance_to_polyline(tr_x, tr_y, np.array([[bx]]), np.array([[by]]))[0, 0])
        if local_d < 32.0:
            continue
        if min(math.hypot(bx - sx, by - sy) for sx, sy in station_xy) < 40.0:
            continue
        radius = rng.uniform(3.2, 8.5)
        height = rng.uniform(1.4, 3.8)
        boulder_cores.append((bx, by, radius))
        boulder_bump += height * np.exp(
            -(((gx - bx) ** 2 + (gy - by) ** 2) / (2.0 * (radius * 0.72) ** 2))
        )

    if "elevation" in raster_layers:
        raw = np.asarray(raster_layers["elevation"], float)
        if raw.shape != (GRID_H, GRID_W):
            raise ValueError("elevation raster must be resampled to the 72 x 72 grid")
        elevation = raw
        terrain_mode = "local_geotiff_crop"
    else:
        elevation = trend + residual_grid + boxwork + massif + knoll + fine_relief + boulder_bump
        terrain_mode = "calibrated_digital_twin"

    base_i, base_j = nearest_cell(base_xy[0], base_xy[1], x0, y0)
    z_base = float(elevation[base_j, base_i])

    # Slope from the metric grid.
    gy_grad, gx_grad = np.gradient(elevation, CELL_M, CELL_M)
    slope_deg = np.degrees(np.arctan(np.hypot(gx_grad, gy_grad)))
    slope_deg = np.clip(slope_deg, 0.0, 89.0)

    # Topographic Position Index: elevation minus an annulus mean (60 m radius).
    footprint = np.ones((9, 9), dtype=float)
    footprint[3:6, 3:6] = 0.0
    footprint /= footprint.sum()
    local_mean = ndimage.convolve(elevation, footprint, mode="nearest")
    tpi = elevation - local_mean
    ridge_threshold = max(0.55, float(np.quantile(tpi, 0.78)))
    ridge = (tpi >= ridge_threshold) & (slope_deg < 31.0)

    # Local roughness (1-cell standard deviation) feeds both hazard and penalty.
    local_mean3 = ndimage.uniform_filter(elevation, size=3, mode="nearest")
    local_var = ndimage.uniform_filter(elevation * elevation, size=3, mode="nearest") - local_mean3**2
    roughness = np.sqrt(np.maximum(local_var, 0.0))

    if "thermal_inertia" in raster_layers:
        thermal_inertia = np.asarray(raster_layers["thermal_inertia"], float)
        if thermal_inertia.shape != (GRID_H, GRID_W):
            raise ValueError("thermal inertia raster must be resampled to 72 x 72")
    else:
        sand_signal = (
            0.68 * np.clip(-tpi / 5.2, 0.0, 1.0)
            + 0.18 * (1.0 - np.clip(roughness / 2.2, 0.0, 1.0))
            + 0.14 * (0.5 + 0.5 * noise_b)
        )
        sand_signal = ndimage.gaussian_filter(sand_signal, 1.15, mode="nearest")
        lo, hi = np.quantile(sand_signal, [0.04, 0.96])
        sand_probability = np.clip((sand_signal - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        thermal_inertia = (
            438.0
            - 258.0 * sand_probability
            + 30.0 * ridge.astype(float)
            + 17.0 * noise_c
        )
        thermal_inertia = np.clip(thermal_inertia, 145.0, 490.0)

    sand_probability = np.clip((385.0 - thermal_inertia) / 235.0, 0.0, 1.0)

    if "d2300" in raster_layers:
        d2300 = np.asarray(raster_layers["d2300"], float)
        if d2300.shape != (GRID_H, GRID_W):
            raise ValueError("D2300 raster must be resampled to 72 x 72")
    else:
        clay_a = np.exp(-(((gx - 330.0) ** 2 + (gy - 205.0) ** 2) / (2.0 * 132.0**2)))
        clay_b = np.exp(-(((gx - 55.0) ** 2 + (gy - 75.0) ** 2) / (2.0 * 105.0**2)))
        noise_d = harmonic_field(
            (GRID_H, GRID_W), gx, gy, seed + 21, wavelength_m=(60.0, 250.0)
        )
        d2300 = 0.006 + 0.019 * ridge.astype(float) + 0.018 * clay_a + 0.010 * clay_b + 0.0045 * noise_d
        d2300 = np.clip(d2300, 0.0, 0.055)

    # Hazard cores: steep slopes, rough benches, and boulder footprints.
    boulder_core = np.zeros_like(elevation, dtype=bool)
    for bx, by, radius in boulder_cores:
        boulder_core |= (gx - bx) ** 2 + (gy - by) ** 2 <= radius * radius
    hazard_core = (
        (slope_deg > float(PHYSICS["hazard_slope_deg"]))
        | (boulder_core & (slope_deg > 7.0))
        | ((roughness > 4.6) & (slope_deg > 13.0))
    )

    # +/-2 m co-registration buffer: upsample 8x (1.875 m subcells), dilate one
    # subcell, then collapse back to the analysis grid.
    sub = max(2, int(round(CELL_M / float(PHYSICS["hazard_coregistration_m"]))))
    core_up = np.repeat(np.repeat(hazard_core, sub, axis=0), sub, axis=1)
    buffered_up = ndimage.binary_dilation(core_up, structure=np.ones((3, 3), bool), iterations=1)
    hazard = buffered_up.reshape(GRID_H, sub, GRID_W, sub).any(axis=(1, 3))

    # Hardware pads at base and stations remain trafficable.
    def clear_pad(x: float, y: float, radius_cells: int = 1) -> None:
        ci, cj = nearest_cell(x, y, x0, y0)
        i0, i1 = max(0, ci - radius_cells), min(GRID_W, ci + radius_cells + 1)
        j0, j1 = max(0, cj - radius_cells), min(GRID_H, cj + radius_cells + 1)
        hazard[j0:j1, i0:i1] = False

    clear_pad(*base_xy, radius_cells=1)
    for sx, sy in station_xy:
        clear_pad(sx, sy, radius_cells=1)

    visible, relays = _build_rf_viewshed(elevation, base_i, base_j)
    rf_shadow = ~visible
    z_rel = elevation - z_base
    for relay in relays:
        relay["x"] = round(x0 + float(relay["x"]), 2)
        relay["y"] = round(y0 + float(relay["y"]), 2)
        relay["z_rel_m"] = round(float(z_rel[int(relay["j"]), int(relay["i"])]), 2)
        relay["network_visible_fraction"] = round(float(visible.mean()), 3)
    base_index = base_j * GRID_W + base_i

    return {
        "width": GRID_W,
        "height": GRID_H,
        "cell_m": CELL_M,
        "x0": x0,
        "y0": y0,
        "origin_easting": e0 + x0,
        "origin_northing": n0 + y0,
        "base_index": base_index,
        "base_i": base_i,
        "base_j": base_j,
        "terrain_mode": terrain_mode,
        "elevation_dm": np.rint(z_rel * 10.0).astype(int).ravel().tolist(),
        "slope_deg_x10": np.rint(slope_deg * 10.0).astype(int).ravel().tolist(),
        "tpi_dm": np.rint(tpi * 10.0).astype(int).ravel().tolist(),
        "thermal_inertia": np.rint(thermal_inertia).astype(int).ravel().tolist(),
        "d2300_x1000": np.rint(d2300 * 1000.0).astype(int).ravel().tolist(),
        "sand_x100": np.rint(sand_probability * 100.0).astype(int).ravel().tolist(),
        "hazard": hazard.astype(int).ravel().tolist(),
        "rf_shadow": rf_shadow.astype(int).ravel().tolist(),
        "ridge": ridge.astype(int).ravel().tolist(),
        "relays": relays,
        "stats": {
            "elevation_min_m": round(float(z_rel.min()), 2),
            "elevation_max_m": round(float(z_rel.max()), 2),
            "mean_slope_deg": round(float(slope_deg.mean()), 2),
            "max_slope_deg": round(float(slope_deg.max()), 2),
            "ridge_fraction": round(float(ridge.mean()), 3),
            "hazard_fraction": round(float(hazard.mean()), 3),
            "rf_shadow_fraction": round(float(rf_shadow.mean()), 3),
            "sand_fraction": round(float((sand_probability > 0.55).mean()), 3),
            "mean_thermal_inertia": round(float(thermal_inertia.mean()), 1),
            "mean_d2300": round(float(d2300.mean()), 4),
            "boulder_count": len(boulder_cores),
            "boulders": [
                {"x": round(float(bx), 1), "y": round(float(by), 1), "radius_m": round(float(r), 1)}
                for bx, by, r in boulder_cores
            ],
        },
        "_arrays": {
            "elevation": elevation,
            "z_rel": z_rel,
            "slope": slope_deg,
            "tpi": tpi,
            "thermal_inertia": thermal_inertia,
            "d2300": d2300,
            "sand": sand_probability,
            "hazard": hazard,
            "rf_shadow": rf_shadow,
            "ridge": ridge,
            "roughness": roughness,
        },
    }


# ---------------------------------------------------------------------------
# Asymmetric routing engine
# ---------------------------------------------------------------------------

NEIGHBOURS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)


class EVAEngine:
    """Energy-minimising, direction-dependent graph search on the twin."""

    def __init__(self, terrain: dict[str, Any]):
        self.terrain = terrain
        self.w = int(terrain["width"])
        self.h = int(terrain["height"])
        self.cell = float(terrain["cell_m"])
        self.x0 = float(terrain["x0"])
        self.y0 = float(terrain["y0"])
        arrays = terrain["_arrays"]
        self.z = np.asarray(arrays["elevation"], float)
        self.z_rel = np.asarray(arrays["z_rel"], float)
        self.slope = np.asarray(arrays["slope"], float)
        self.tpi = np.asarray(arrays["tpi"], float)
        self.thermal_inertia = np.asarray(arrays["thermal_inertia"], float)
        self.d2300 = np.asarray(arrays["d2300"], float)
        self.sand = np.asarray(arrays["sand"], float)
        self.hazard = np.asarray(arrays["hazard"], bool)
        self.rf_shadow = np.asarray(arrays["rf_shadow"], bool)
        self.ridge = np.asarray(arrays["ridge"], bool)
        self.roughness = np.asarray(arrays["roughness"], float)
        self.passable = ~self.hazard
        self.base_index = int(terrain["base_index"])
        self.walkback_energy_wh: np.ndarray | None = None
        self.walkback_time_s: np.ndarray | None = None
        self._prepare_walkback()

    # -- geometry ---------------------------------------------------------
    def idx(self, i: int, j: int) -> int:
        return j * self.w + i

    def ij(self, index: int) -> tuple[int, int]:
        return index % self.w, index // self.w

    def xy(self, index: int) -> tuple[float, float]:
        i, j = self.ij(index)
        return self.x0 + (i + 0.5) * self.cell, self.y0 + (j + 0.5) * self.cell

    def nearest_node(self, x: float, y: float) -> int:
        i = int(np.clip(round((x - self.x0) / self.cell - 0.5), 0, self.w - 1))
        j = int(np.clip(round((y - self.y0) / self.cell - 0.5), 0, self.h - 1))
        return self.idx(i, j)

    def passable_node(self, x: float, y: float, radius: int = 6) -> int:
        """Nearest trafficable analysis node to a map coordinate."""
        return self._passable_nearest(x, y, radius)

    def _passable_nearest(self, x: float, y: float, radius: int = 4) -> int:
        seed = self.nearest_node(x, y)
        if self.passable.ravel()[seed]:
            return seed
        si, sj = self.ij(seed)
        for r in range(1, radius + 1):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    i, j = si + di, sj + dj
                    if 0 <= i < self.w and 0 <= j < self.h and self.passable[j, i]:
                        return self.idx(i, j)
        return seed

    # -- edge physics -----------------------------------------------------
    def edge(self, source: int, target: int, *, sprint: bool = False) -> dict[str, float]:
        """Directional edge metrics. All energy values are watt-hours."""
        si, sj = self.ij(source)
        ti, tj = self.ij(target)
        dx = (ti - si) * self.cell
        dy = (tj - sj) * self.cell
        dist = math.hypot(dx, dy)
        dz = float(self.z[tj, ti] - self.z[sj, si])
        grade = dz / max(dist, 1e-9)

        sand = 0.5 * (self.sand[sj, si] + self.sand[tj, ti])
        rough = 0.5 * (self.roughness[sj, si] + self.roughness[tj, ti])
        rough_norm = min(1.0, rough / 2.0)

        speed = float(PHYSICS["walk_speed_flat_m_s"])
        if grade >= 0.0:
            speed /= 1.0 + 4.2 * grade
        else:
            speed *= 1.0 + 0.22 * min(0.60, -grade)
        speed *= 1.0 - 0.38 * sand
        speed *= 1.0 - 0.10 * rough_norm
        speed = float(np.clip(speed, 0.30, 1.45))
        if sprint:
            speed *= float(PHYSICS["sprint_multiplier"])

        time_s = dist / eps if (eps := max(speed, 1e-6)) else 0.0

        body = float(PHYSICS["astronaut_body_kg"])
        load = float(PHYSICS["suit_plss_kg"])
        g_eff = 100.0 * grade * GRAVITY_RATIO
        if g_eff >= 0.0:
            grade_term = 0.35 * speed * g_eff
        else:
            grade_term = 0.35 * speed * g_eff * 0.55
        terrain_factor = 1.0 + 0.55 * sand + 0.12 * rough_norm
        metabolic_w = (
            1.5 * body
            + 2.0 * (body + load) * (load / body) ** 2
            + terrain_factor * (body + load) * (1.5 * speed * speed + grade_term)
        )
        metabolic_w *= float(PHYSICS["suit_metabolic_overhead"])
        metabolic_w = float(np.clip(metabolic_w, 140.0, 1250.0))
        energy_wh = metabolic_w * time_s / 3600.0

        shadow = bool(self.rf_shadow[tj, ti])
        comm_penalty = 1.0 + (float(PHYSICS["comm_energy_penalty"]) if shadow else 0.0)
        return {
            "dist_m": dist,
            "dz_m": dz,
            "grade": grade,
            "speed_m_s": speed,
            "time_s": time_s,
            "metabolic_w": metabolic_w,
            "energy_wh": energy_wh,
            "comm_energy_wh": energy_wh * comm_penalty,
            "sand": sand,
            "shadow": 1.0 if shadow else 0.0,
        }

    # -- graph search -----------------------------------------------------
    def _dijkstra(
        self,
        source: int,
        goal: int | None,
        mode: str,
        *,
        reverse: bool = False,
        allow_hazard: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.w * self.h
        dist = np.full(n, np.inf, dtype=float)
        parent = np.full(n, -1, dtype=int)
        if not 0 <= source < n:
            return dist, parent
        if not allow_hazard and not self.passable.ravel()[source]:
            source = self._passable_nearest(*self.xy(source))
        dist[source] = 0.0
        heap: list[tuple[float, int]] = [(0.0, source)]

        def cost(metrics: dict[str, float]) -> float:
            if mode == "time":
                return metrics["time_s"]
            if mode == "energy_physical":
                return metrics["energy_wh"]
            return metrics["comm_energy_wh"]

        while heap:
            d, u = heappop(heap)
            if d > dist[u] + 1e-12:
                continue
            if goal is not None and u == goal:
                break
            ui, uj = self.ij(u)
            for di, dj in NEIGHBOURS:
                vi, vj = ui + di, uj + dj
                if not (0 <= vi < self.w and 0 <= vj < self.h):
                    continue
                v = self.idx(vi, vj)
                if not allow_hazard and not self.passable[vj, vi]:
                    continue
                metrics = self.edge(v, u) if reverse else self.edge(u, v)
                extra = cost(metrics)
                if allow_hazard and self.hazard[vj, vi]:
                    extra *= 22.0
                nd = d + extra
                if nd + 1e-12 < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heappush(heap, (nd, v))
        return dist, parent

    def _reconstruct(self, parent: np.ndarray, source: int, goal: int) -> list[int] | None:
        if source == goal:
            return [source]
        if not np.isfinite(parent[goal]) and parent[goal] < 0:
            return None
        path = [goal]
        guard = 0
        while path[-1] != source:
            p = int(parent[path[-1]])
            if p < 0:
                return None
            path.append(p)
            guard += 1
            if guard > self.w * self.h:
                return None
        path.reverse()
        return path

    def short_path(
        self,
        source: int,
        goal: int,
        mode: str = "energy",
        *,
        sprint: bool = False,
        allow_hazard: bool = False,
    ) -> dict[str, Any] | None:
        dist, parent = self._dijkstra(source, goal, mode, allow_hazard=allow_hazard)
        idxs = self._reconstruct(parent, source, goal)
        if idxs is None:
            return None
        steps: list[dict[str, float]] = []
        for a, b in zip(idxs, idxs[1:]):
            metrics = self.edge(a, b, sprint=sprint)
            metrics["from"] = a
            metrics["to"] = b
            steps.append(metrics)
        return {
            "path_idx": idxs,
            "steps": steps,
            "cost": float(dist[goal]),
            "distance_m": float(sum(s["dist_m"] for s in steps)),
            "time_s": float(sum(s["time_s"] for s in steps)),
            "energy_wh": float(sum(s["energy_wh"] for s in steps)),
            "comm_energy_wh": float(sum(s["comm_energy_wh"] for s in steps)),
        }

    def _prepare_walkback(self) -> None:
        e_dist, _ = self._dijkstra(self.base_index, None, "energy_physical", reverse=True)
        t_dist, _ = self._dijkstra(self.base_index, None, "time", reverse=True)
        self.walkback_energy_wh = e_dist
        self.walkback_time_s = t_dist

    # -- plan -------------------------------------------------------------
    def plan(
        self,
        stations: Sequence[dict[str, Any]],
        station_indices: Sequence[int],
    ) -> dict[str, Any]:
        assert self.walkback_energy_wh is not None and self.walkback_time_s is not None
        cap = float(PHYSICS["plss_energy_budget_wh"])
        reserve_fraction = float(PHYSICS["walkback_reserve_fraction"])
        reserve_floor = cap * reserve_fraction
        dwell_s = float(PHYSICS["dwell_seconds_per_station"])
        dwell_w = float(PHYSICS["dwell_metabolic_w"])
        dwell_wh = dwell_w * dwell_s / 3600.0

        cur = self.base_index
        path_nodes: list[int] = [cur]
        timeline: list[list[float | int]] = []
        bx, by = self.xy(cur)
        timeline.append([round(bx, 2), round(by, 2), round(float(self.z_rel.ravel()[cur]), 2), 0.0, 0.0, 0.0, 0, -1])

        cum_t = 0.0
        cum_e = 0.0
        cum_d = 0.0
        walk_time = 0.0
        walk_energy = 0.0
        dwell_total_s = 0.0
        dwell_total_wh = 0.0
        station_metrics: list[dict[str, Any]] = []
        min_margin = math.inf
        min_margin_idx = cur
        min_margin_station = -1
        breach = False
        deferred = False

        for k, target in enumerate(station_indices):
            if deferred:
                r = self.xy(target)
                station_metrics.append({
                    "index": k,
                    "node": int(target),
                    "x": round(r[0], 2),
                    "y": round(r[1], 2),
                    "status": "DEFERRED",
                    "feasible": False,
                    "arrival_time_s": None,
                    "arrival_energy_wh": None,
                    "depart_time_s": None,
                    "depart_energy_wh": None,
                    "walkback_energy_wh": round(float(self.walkback_energy_wh[target]), 2),
                    "walkback_time_s": round(float(self.walkback_time_s[target]), 1),
                    "reserve_floor_wh": round(reserve_floor, 1),
                    "margin_wh": None,
                    "distance_m": None,
                    "path_index": None,
                    "timeline_arrival_index": None,
                    "timeline_depart_index": None,
                })
                continue

            result = self.short_path(cur, int(target), "energy")
            if result is None:
                r = self.xy(target)
                station_metrics.append({
                    "index": k,
                    "node": int(target),
                    "x": round(r[0], 2),
                    "y": round(r[1], 2),
                    "status": "UNREACHABLE",
                    "feasible": False,
                    "arrival_time_s": None,
                    "arrival_energy_wh": None,
                    "depart_time_s": None,
                    "depart_energy_wh": None,
                    "walkback_energy_wh": round(float(self.walkback_energy_wh[target]), 2),
                    "walkback_time_s": round(float(self.walkback_time_s[target]), 1),
                    "reserve_floor_wh": round(reserve_floor, 1),
                    "margin_wh": None,
                    "distance_m": None,
                    "path_index": None,
                    "timeline_arrival_index": None,
                    "timeline_depart_index": None,
                })
                deferred = True
                breach = True
                continue

            for step in result["steps"]:
                node = int(step["to"])
                path_nodes.append(node)
                cum_t += float(step["time_s"])
                cum_e += float(step["energy_wh"])
                cum_d += float(step["dist_m"])
                walk_time += float(step["time_s"])
                walk_energy += float(step["energy_wh"])
                x, y = self.xy(node)
                timeline.append([
                    round(x, 2), round(y, 2), round(float(self.z_rel.ravel()[node]), 2),
                    round(cum_t, 2), round(cum_e, 3), round(cum_d, 2), 0, -1,
                ])
                # Strict reserve test at every traversed node, not just stations.
                wb = float(self.walkback_energy_wh[node])
                if math.isfinite(wb):
                    margin = cap - cum_e - wb - reserve_floor
                    if margin < min_margin:
                        min_margin = margin
                        min_margin_idx = node
                        min_margin_station = k
                    if margin < 0.0:
                        breach = True

            cur = int(target)
            arrival_t = cum_t
            arrival_e = cum_e
            dwell_e = dwell_wh
            cum_t += dwell_s
            cum_e += dwell_e
            dwell_total_s += dwell_s
            dwell_total_wh += dwell_e
            x, y = self.xy(cur)
            departure_index = len(timeline)
            timeline.append([
                round(x, 2), round(y, 2), round(float(self.z_rel.ravel()[cur]), 2),
                round(cum_t, 2), round(cum_e, 3), round(cum_d, 2), 1, k,
            ])

            wb_e = float(self.walkback_energy_wh[cur])
            wb_t = float(self.walkback_time_s[cur])
            remaining = cap - cum_e
            margin = remaining - wb_e - reserve_floor if math.isfinite(wb_e) else -math.inf
            feasible = bool(math.isfinite(wb_e) and margin >= 0.0 and cum_e <= cap)
            if not feasible:
                breach = True
            if math.isfinite(margin) and margin < min_margin:
                min_margin = margin
                min_margin_idx = cur
                min_margin_station = k
            status = "GO" if feasible else "RESERVE BREACH"
            station_metrics.append({
                "index": k,
                "node": int(cur),
                "x": round(x, 2),
                "y": round(y, 2),
                "status": status,
                "feasible": feasible,
                "arrival_time_s": round(arrival_t, 1),
                "arrival_energy_wh": round(arrival_e, 2),
                "depart_time_s": round(cum_t, 1),
                "depart_energy_wh": round(cum_e, 2),
                "walkback_energy_wh": round(wb_e, 2),
                "walkback_time_s": round(wb_t, 1),
                "reserve_floor_wh": round(reserve_floor, 1),
                "margin_wh": round(margin, 2) if math.isfinite(margin) else None,
                "distance_m": round(cum_d, 1),
                "path_index": len(path_nodes) - 1,
                "timeline_arrival_index": departure_index - 1,
                "timeline_depart_index": departure_index,
            })

        # Contingency walkback from the furthest occupied station.
        final_station_metric = next(
            (m for m in reversed(station_metrics) if m.get("node") is not None and m.get("status") in {"GO", "RESERVE BREACH"}),
            None,
        )
        walkback_path: list[list[float]] = []
        walkback_energy = 0.0
        walkback_time = 0.0
        walkback_dist = 0.0
        if final_station_metric is not None:
            wres = self.short_path(int(final_station_metric["node"]), self.base_index, "energy_physical")
            if wres is not None:
                walkback_energy = wres["energy_wh"]
                walkback_time = wres["time_s"]
                walkback_dist = wres["distance_m"]
                for node in wres["path_idx"]:
                    x, y = self.xy(node)
                    walkback_path.append([
                        round(x, 2), round(y, 2), round(float(self.z_rel.ravel()[node]), 2),
                    ])

        path_xy = []
        for node in path_nodes:
            x, y = self.xy(node)
            path_xy.append([
                round(x, 2), round(y, 2), round(float(self.z_rel.ravel()[node]), 2),
            ])

        total_time = walk_time + dwell_total_s
        total_energy = walk_energy + dwell_total_wh
        o2_used = total_energy * 3600.0 / (float(PHYSICS["o2_metabolic_mj_per_kg"]) * 1.0e6)
        water_used = (total_time / 60.0) * float(PHYSICS["water_rate_kg_per_min"])
        final_margin = station_metrics[-1]["margin_wh"] if station_metrics else None
        if math.isinf(min_margin):
            min_margin_value = None
        else:
            min_margin_value = round(min_margin, 2)
        feasible_plan = bool((not breach) and min_margin_value is not None and min_margin_value >= 0)

        return {
            "path": path_xy,
            "timeline": timeline,
            "station_metrics": station_metrics,
            "walkback": {
                "from_node": int(final_station_metric["node"]) if final_station_metric else self.base_index,
                "path": walkback_path,
                "distance_m": round(walkback_dist, 1),
                "time_s": round(walkback_time, 1),
                "energy_wh": round(walkback_energy, 2),
            },
            "reserve": {
                "capacity_wh": round(cap, 1),
                "fraction": reserve_fraction,
                "floor_wh": round(reserve_floor, 1),
                "worst_margin_wh": min_margin_value,
                "worst_station_index": int(min_margin_station),
                "feasible": feasible_plan,
            },
            "stats": {
                "distance_m": round(cum_d, 1),
                "walk_time_s": round(walk_time, 1),
                "dwell_time_s": round(dwell_total_s, 1),
                "total_time_s": round(total_time, 1),
                "walk_energy_wh": round(walk_energy, 2),
                "dwell_energy_wh": round(dwell_total_wh, 2),
                "total_energy_wh": round(total_energy, 2),
                "remaining_wh": round(cap - total_energy, 2),
                "remaining_pct": round((cap - total_energy) / cap * 100.0, 1),
                "o2_used_kg": round(o2_used, 4),
                "o2_remaining_pct": round(max(0.0, 1.0 - o2_used / float(PHYSICS["o2_capacity_kg"])) * 100.0, 1),
                "water_used_kg": round(water_used, 4),
                "water_remaining_pct": round(max(0.0, 1.0 - water_used / float(PHYSICS["water_capacity_kg"])) * 100.0, 1),
                "final_walkback_margin_wh": final_margin,
                "final_walkback_energy_wh": round(walkback_energy, 2),
                "station_count": len(station_metrics),
                "feasible_station_count": sum(1 for m in station_metrics if m["feasible"]),
                "asymmetry_pct": round(
                    (walk_energy - walkback_energy) / walk_energy * 100.0, 1
                ) if walk_energy > 1e-9 and walkback_energy > 0 else 0.0,
                "note": "walkback reserve is carried on top of the 25 % PLSS floor",
            },
        }


# ---------------------------------------------------------------------------
# Mission bundle assembly
# ---------------------------------------------------------------------------

def _scene_image(spec: dict[str, Any]) -> dict[str, Any]:
    image = spec.get("image") or {}
    url = image.get("url") or spec.get("image_url")
    fallbacks = image.get("fallback_urls") or spec.get("fallback_image_urls") or []
    return {
        "url": url,
        "sol": image.get("sol", spec.get("image_sol", spec.get("sol"))),
        "title": image.get("title", spec.get("image_title")),
        "imageid": image.get("imageid"),
        "date_taken": image.get("date_taken"),
        "fallback_urls": list(fallbacks)[:4],
        "page_url": image.get("page_url") or (
            f"https://mars.nasa.gov/msl/multimedia/raw-images/?instrument=MAST_LEFT&sol={spec.get('sol')}"
        ),
    }


def build_scene(
    track: Sequence[dict[str, Any]],
    station_specs: Sequence[dict[str, Any]],
    *,
    data_sources: dict[str, Any] | None = None,
    raster_layers: dict[str, np.ndarray] | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    """Build terrain, plan the EVA and return a JSON-ready bundle plus arrays."""
    projected, e0, n0, min_x, min_y = project_track(track)
    by_sol: dict[int, dict[str, Any]] = {}
    for p in projected:
        by_sol.setdefault(int(p["sol"]), p)

    base_waypoint = by_sol.get(4537, projected[0])
    base_xy = (float(base_waypoint["x"]), float(base_waypoint["y"]))

    stations: list[dict[str, Any]] = []
    station_xy: list[tuple[float, float]] = []
    for spec in station_specs:
        sol = int(spec["sol"])
        wp = by_sol.get(sol)
        if wp is None:
            # nearest sol fallback keeps the demo robust if the API shifts.
            wp = min(projected, key=lambda p: abs(int(p["sol"]) - sol))
        sx, sy = float(wp["x"]), float(wp["y"])
        station_xy.append((sx, sy))
        image = _scene_image(spec)
        stations.append({
            **spec,
            "sol": sol,
            "x": round(sx, 2),
            "y": round(sy, 2),
            "lat": float(wp["lat"]),
            "lon": float(wp["lon"]),
            "elev_geoid": float(wp["elev_geoid"]),
            "site": int(wp.get("site", 0)),
            "drive": int(wp.get("drive", 0)),
            "real_dist_m": float(wp.get("dist_m", 0.0)),
            "image": image,
            "image_url": image["url"],
            "image_title": image["title"],
            "image_sol": image["sol"],
            "image_date_taken": image["date_taken"],
            "fallback_image_urls": image["fallback_urls"],
        })

    terrain = build_terrain(
        projected,
        base_xy,
        station_xy,
        seed=seed,
        raster_layers=raster_layers,
    )
    engine = EVAEngine(terrain)
    station_nodes = [engine.passable_node(x, y) for x, y in station_xy]
    route = engine.plan(stations, station_nodes)

    arrays = terrain["_arrays"]
    z_rel = arrays["z_rel"]
    rf_shadow = arrays["rf_shadow"]
    hazard = arrays["hazard"]
    ridge = arrays["ridge"]

    base_i = int(terrain["base_i"])
    base_j = int(terrain["base_j"])
    base = {
        "id": "BASE",
        "name": SCENARIO["base_name"],
        "sol": int(base_waypoint["sol"]),
        "x": round(base_xy[0], 2),
        "y": round(base_xy[1], 2),
        "lat": float(base_waypoint["lat"]),
        "lon": float(base_waypoint["lon"]),
        "elev_geoid": float(base_waypoint["elev_geoid"]),
        "elevation_rel_m": 0.0,
        "node": int(terrain["base_index"]),
        "comm_mast_m": 4.5,
        "los": True,
        "base_los": True,
        "rf_shadow": False,
        "hazard": False,
        "shelter": "Pressurized rover / base camp",
    }

    for idx, station in enumerate(stations):
        metric = route["station_metrics"][idx]
        i, j = engine.ij(engine.passable_node(station["x"], station["y"]))
        station["x"] = station["x"]
        station["y"] = station["y"]
        station["node"] = int(metric["node"])
        station["route"] = metric
        station["status"] = metric["status"]
        station["los"] = bool(not rf_shadow[j, i])
        station["base_los"] = bool(
            _line_of_sight(arrays["elevation"], base_i, base_j, i, j)
        )
        station["comm_network"] = "BASE + R1/R2/R3" if station["los"] else "SHADOW"
        station["terrain"] = {
            "elev_rel_m": round(float(z_rel[j, i]), 2),
            "slope_deg": round(float(arrays["slope"][j, i]), 1),
            "tpi_m": round(float(arrays["tpi"][j, i]), 2),
            "thermal_inertia": round(float(arrays["thermal_inertia"][j, i]), 0),
            "d2300": round(float(arrays["d2300"][j, i]), 4),
            "sand_x100": int(round(float(arrays["sand"][j, i]) * 100.0)),
            "ridge": bool(ridge[j, i]),
            "hazard": bool(hazard[j, i]),
            "hazard_buffer_m": float(PHYSICS["hazard_coregistration_m"]),
        }
        station["science_layers"] = [
            "HiRISE TPI ridge" if ridge[j, i] else "TPI flank",
            "THEMIS sand" if bool(arrays["sand"][j, i] > 0.5) else "THEMIS competent",
            "CRISM D2300" if float(arrays["d2300"][j, i]) >= 0.025 else "CRISM background",
        ]

    track_out = []
    for p in projected:
        ti = engine.nearest_node(float(p["x"]), float(p["y"]))
        ti_i, ti_j = engine.ij(ti)
        track_out.append({
            "sol": int(p["sol"]),
            "site": int(p.get("site", 0)),
            "drive": int(p.get("drive", 0)),
            "x": round(float(p["x"]), 2),
            "y": round(float(p["y"]), 2),
            "lat": float(p["lat"]),
            "lon": float(p["lon"]),
            "elev_geoid": float(p["elev_geoid"]),
            "twin_elev_rel_m": round(float(z_rel[ti_j, ti_i]), 2),
            "real_dist_m": float(p.get("dist_m", 0.0)),
        })

    center_lat = float(np.mean([p["lat"] for p in projected]))
    center_lon = float(np.mean([p["lon"] for p in projected]))
    terrain_json = {k: v for k, v in terrain.items() if k != "_arrays"}

    data_sources = dict(data_sources or {})
    data_sources.setdefault("waypoints", {
        "url": "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json",
        "status": "built_in_verified_fallback",
        "count": len(track_out),
    })
    data_sources.setdefault("mastcam", {
        "url": "https://mars.nasa.gov/api/v1/raw_image_items",
        "status": "built_in_verified_fallback",
        "instrument": "MAST_LEFT",
    })
    data_sources["terrain"] = {
        "mode": terrain["terrain_mode"],
        "note": (
            "Real Curiosity waypoints constrain the elevation trend. Boxwork relief, "
            "THEMIS-style TI, CRISM-style D2300, TPI and RF viewshed are modeled unless "
            "local rasters are supplied in data/raw/."
        ),
        "hazard_coregistration_m": PHYSICS["hazard_coregistration_m"],
        "comm_network": "Base mast + three 12 m ridge repeaters; shadow = no ray-traced LOS.",
        "relays": terrain_json.get("relays", []),
    }

    bundle = {
        "schema": "ares-walk/mission-bundle@1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "scenario": {
            **SCENARIO,
            "principal_route": "BASE -> ALT-01 ... ALT-10",
            "route_feasible": bool(route["reserve"]["feasible"]),
            "notes": list(SCENARIO["notes"]),
        },
        "site": {
            "name": "Mount Sharp / Gale Crater, Mars",
            "feature": "Altadena boxwork unit (operational label)",
            "center_lat": round(center_lat, 6),
            "center_lon": round(center_lon, 6),
            "projection": "local ENU metres from JPL MMGIS easting/northing",
            "origin_easting": e0,
            "origin_northing": n0,
            "sol_window": [4535, 4611],
        },
        "physics": {
            **PHYSICS,
            "pandolf_adapted": (
                "M = 1.5W + 2.0(W+L)(L/W)^2 + eta(W+L)(1.5V^2 + 0.35 V G*0.3794)"
            ),
            "reserve_rule": (
                "At every traversed node: capacity - consumed - walkback_energy "
                ">= 0.25 * capacity"
            ),
        },
        "base": base,
        "stations": stations,
        "track": track_out,
        "terrain": terrain_json,
        "route": route,
        "mastcam": {
            "instrument": "MAST_LEFT",
            "sol_window": [4535, 4611],
            "image_count": sum(1 for s in stations if s.get("image_url")),
            "images": [
                {
                    "sol": s["sol"],
                    "station": s["id"],
                    "url": s.get("image_url"),
                    "title": s.get("image_title"),
                    "fallbacks": s.get("fallback_image_urls", []),
                }
                for s in stations
            ],
        },
        "data_sources": data_sources,
        "attribution": [
            "Curiosity waypoints: NASA/JPL-Caltech, MSL MMGIS waypoint layer.",
            "Mastcam imagery: NASA/JPL-Caltech/MSSS, Mars Science Laboratory raw images.",
            "Terrain twin: ARES-WALK scenario model, calibrated to real waypoint elevations.",
            "HiRISE/THEMIS/CRISM references: NASA/JPL/University of Arizona and ASU.",
        ],
        "ai_disclosure": (
            "ARES-WALK was scaffolded and implemented with an AI coding agent under "
            "human direction. Physics, data provenance and limitations are documented in README.md."
        ),
    }
    return {
        "bundle": bundle,
        "terrain": terrain,
        "engine": engine,
        "route": route,
    }


# ---------------------------------------------------------------------------
# Static verification plot
# ---------------------------------------------------------------------------

def decode_terrain(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    """Decode the compact quantised fields back into float grids."""
    t = bundle["terrain"]
    w, h = int(t["width"]), int(t["height"])

    def field(key: str, scale: float = 1.0, dtype: Any = float) -> np.ndarray:
        return np.asarray(t[key], dtype=dtype).reshape(h, w) * scale

    return {
        "elevation": field("elevation_dm", 0.1),
        "slope": field("slope_deg_x10", 0.1),
        "tpi": field("tpi_dm", 0.1),
        "thermal_inertia": field("thermal_inertia"),
        "d2300": field("d2300_x1000", 0.001),
        "sand": field("sand_x100", 0.01),
        "hazard": field("hazard", dtype=bool),
        "rf_shadow": field("rf_shadow", dtype=bool),
        "ridge": field("ridge", dtype=bool),
    }


def render_plan(bundle: dict[str, Any], out_path: Path = PLOT_PATH) -> Path:
    """Render the committed two-panel verification figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource, LinearSegmentedColormap, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t = bundle["terrain"]
    w, h = int(t["width"]), int(t["height"])
    cell = float(t["cell_m"])
    x0, y0 = float(t["x0"]), float(t["y0"])
    extent = [x0, x0 + w * cell, y0, y0 + h * cell]
    xs = x0 + (np.arange(w) + 0.5) * cell
    ys = y0 + (np.arange(h) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    arrays = decode_terrain(bundle)
    z = arrays["elevation"]

    mars = LinearSegmentedColormap.from_list(
        "ares_mars",
        ["#1b0d0b", "#452017", "#7a3c27", "#b5663e", "#d99a67", "#f0c79c"],
    )
    light = LightSource(azdeg=315.0, altdeg=43.0)
    shade = light.shade(
        z,
        cmap=mars,
        vert_exag=1.25,
        dx=cell,
        dy=cell,
        blend_mode="soft",
        fraction=1.15,
    )

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": "#dfe7ee",
        "axes.labelcolor": "#c2ccd6",
        "xtick.color": "#94a3b0",
        "ytick.color": "#94a3b0",
        "axes.edgecolor": "#31404f",
    })

    fig = plt.figure(figsize=(15.2, 7.9), facecolor="#070b10")
    gs = fig.add_gridspec(
        1, 2, width_ratios=(1.08, 1.0), left=0.045, right=0.975,
        top=0.84, bottom=0.095, wspace=0.10,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor("#070b10")
    ax.imshow(shade, origin="lower", extent=extent, interpolation="bilinear", zorder=1)

    # TPI ridge extraction: translucent gold cells + contour crests.
    ridge_rgba = np.zeros((h, w, 4), dtype=float)
    ridge_rgba[..., 0] = 0.96
    ridge_rgba[..., 1] = 0.72
    ridge_rgba[..., 2] = 0.32
    ridge_rgba[..., 3] = np.where(arrays["ridge"], 0.16, 0.0)
    ax.imshow(ridge_rgba, origin="lower", extent=extent, zorder=2, interpolation="nearest")
    ax.contour(
        gx, gy, arrays["tpi"], levels=[0.7, 1.5, 2.5],
        colors=["#f2b35f"], linewidths=[0.35, 0.45, 0.55], alpha=0.55, zorder=3,
    )

    # Thermal inertia (sand) wash.
    ti_rgba = plt.get_cmap("Blues_r")(np.clip((arrays["thermal_inertia"] - 145) / 345.0, 0, 1))
    ti_rgba[..., 3] = np.where(arrays["sand"] > 0.45, 0.13, 0.0)
    ax.imshow(ti_rgba, origin="lower", extent=extent, zorder=2, interpolation="bilinear")

    # D2300 mineralogy contours.
    ax.contour(
        gx, gy, arrays["d2300"], levels=[0.028, 0.038],
        colors=["#d8a1ff", "#b36bff"], linewidths=[0.5, 0.6], alpha=0.8, zorder=3,
    )

    # RF shadow and hazard overlays.
    shadow_rgba = np.zeros((h, w, 4), dtype=float)
    shadow_rgba[..., 0] = 0.24
    shadow_rgba[..., 1] = 0.67
    shadow_rgba[..., 2] = 0.86
    shadow_rgba[..., 3] = np.where(arrays["rf_shadow"], 0.22, 0.0)
    ax.imshow(shadow_rgba, origin="lower", extent=extent, zorder=3, interpolation="nearest")
    ax.contourf(
        gx, gy, arrays["rf_shadow"].astype(float), levels=[0.5, 1.5],
        colors="none", hatches=["////"], edgecolor="#4fc3e8", linewidth=0.0, alpha=0.24, zorder=3,
    )
    hazard_rgba = np.zeros((h, w, 4), dtype=float)
    hazard_rgba[..., 0] = 1.0
    hazard_rgba[..., 1] = 0.23
    hazard_rgba[..., 2] = 0.16
    hazard_rgba[..., 3] = np.where(arrays["hazard"], 0.25, 0.0)
    ax.imshow(hazard_rgba, origin="lower", extent=extent, zorder=4, interpolation="nearest")
    ax.contour(
        gx, gy, arrays["hazard"].astype(float), levels=[0.5],
        colors=["#ff5d4a"], linewidths=0.45, alpha=0.65, zorder=4,
    )

    # Real Curiosity track.
    if bundle.get("track"):
        tx = [p["x"] for p in bundle["track"]]
        ty = [p["y"] for p in bundle["track"]]
        ax.plot(tx, ty, color="#f3f1e8", lw=1.15, ls=(0, (1.0, 2.2)), alpha=0.82, zorder=6,
                label="Curiosity track (sols 4537-4609)")

    # Walkback then outbound route.
    wb = bundle["route"].get("walkback", {}).get("path", [])
    if wb:
        ax.plot([p[0] for p in wb], [p[1] for p in wb], color="#ffb45c", lw=1.45,
                ls=(0, (4.0, 2.1)), alpha=0.95, zorder=7, label="Contingency walkback")
    route_xy = bundle["route"].get("path", [])
    if route_xy:
        rx = [p[0] for p in route_xy]
        ry = [p[1] for p in route_xy]
        ax.plot(rx, ry, color="#42e4f2", lw=4.2, alpha=0.18, zorder=8, solid_capstyle="round")
        ax.plot(rx, ry, color="#74f0f8", lw=1.55, alpha=1.0, zorder=9,
                label="0.38 g energy-minimum route")

    # Relays.
    for relay in t.get("relays", []):
        ax.scatter(relay["x"], relay["y"], marker="^", s=54, facecolor="#69d4ff",
                   edgecolor="#071018", linewidth=0.8, zorder=10)
        ax.text(relay["x"] + 13, relay["y"] + 11, relay["id"], color="#8fe2ff", fontsize=6.5,
                fontweight="bold", zorder=11, path_effects=[pe.withStroke(linewidth=2, foreground="#071018")])

    # Base and science stations.
    base = bundle["base"]
    ax.scatter(base["x"], base["y"], marker="*", s=250, facecolor="#8ff7ff",
               edgecolor="#031116", linewidth=0.9, zorder=12)
    ax.annotate(
        "BASE / ROVER", (base["x"], base["y"]), xytext=(0, -23), textcoords="offset points",
        color="#a9f8ff", fontsize=7.2, fontweight="bold", ha="center", va="top", zorder=12,
        path_effects=[pe.withStroke(linewidth=2.2, foreground="#071018")],
    )
    station_x = [s["x"] for s in bundle["stations"]]
    station_y = [s["y"] for s in bundle["stations"]]
    ax.scatter(station_x, station_y, s=34, facecolor="#ff8d4d", edgecolor="#160b08",
               linewidth=0.7, zorder=12)
    for k, s in enumerate(bundle["stations"]):
        dx = 7 if k % 2 == 0 else -7
        ha = "left" if k % 2 == 0 else "right"
        ax.annotate(
            f'{k+1:02d}',
            (s["x"], s["y"]), xytext=(dx, 5), textcoords="offset points",
            fontsize=5.5, color="#ffd2b1", ha=ha, va="bottom", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=1.8, foreground="#071018")],
            zorder=13,
        )

    # North arrow + scale bar.
    ax.annotate(
        "N", xy=(0.925, 0.90), xytext=(0.925, 0.80), xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#e8eef4", lw=1.1),
        ha="center", va="center", fontsize=8.5, fontweight="bold", color="#e8eef4",
    )
    sb_len = 200.0
    sb_x = x0 + 0.055 * (w * cell)
    sb_y = y0 + 0.055 * (h * cell)
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="#e9eef4", lw=1.8, solid_capstyle="butt", zorder=12)
    ax.plot([sb_x, sb_x], [sb_y - 6, sb_y + 6], color="#e9eef4", lw=1.0, zorder=12)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - 6, sb_y + 6], color="#e9eef4", lw=1.0, zorder=12)
    ax.text(sb_x + sb_len / 2, sb_y + 14, "200 m", ha="center", va="bottom",
            color="#dce6ee", fontsize=6.8, zorder=12)

    legend_handles = [
        Line2D([0], [0], color="#f3f1e8", lw=1.2, ls=(0, (1, 2)), label="Curiosity track"),
        Line2D([0], [0], color="#74f0f8", lw=1.6, label="Energy-minimum EVA route"),
        Line2D([0], [0], color="#ffb45c", lw=1.5, ls=(0, (4, 2)), label="Walkback reserve leg"),
        Patch(facecolor="#f2b35f", alpha=0.35, edgecolor="none", label="HiRISE TPI ridge"),
        Patch(facecolor="#4fc3e8", alpha=0.35, hatch="////", edgecolor="#4fc3e8", lw=0.0, label="RF shadow (no LOS)"),
        Patch(facecolor="#ff5d4a", alpha=0.35, edgecolor="#ff5d4a", label="Hazard + 2 m buffer"),
        Line2D([0], [0], color="#d8a1ff", lw=0.9, label="CRISM D2300 > 0.028"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#69d4ff",
               markeredgecolor="#071018", markersize=6, label="Ridge repeater"),
    ]
    leg = ax.legend(
        handles=legend_handles, loc="lower left", fontsize=5.9, ncol=2,
        facecolor="#071018", edgecolor="#2b3945", framealpha=0.86,
        handlelength=2.4, columnspacing=1.0, borderpad=0.55, labelcolor="#cfdae4",
    )
    leg.set_zorder(20)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("local easting (m)", fontsize=7.4)
    ax.set_ylabel("local northing (m)", fontsize=7.4)
    ax.tick_params(labelsize=6.6)
    ax.set_title(
        "A  ·  TERRAIN  ·  TPI  ·  THERMAL INERTIA  ·  RF VIEWSHED",
        loc="left", fontsize=8.7, color="#eaf2f8", fontweight="bold", pad=7,
    )
    for spine in ax.spines.values():
        spine.set_color("#31404f")

    # ------------------------------------------------------------------ right
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#0a1017")
    timeline = bundle["route"]["timeline"]
    td = np.array([row[5] for row in timeline], dtype=float) / 1000.0
    tt = np.array([row[3] for row in timeline], dtype=float)
    te = np.array([row[4] for row in timeline], dtype=float)
    tz = np.array([row[2] for row in timeline], dtype=float)
    cap = float(bundle["physics"]["plss_energy_budget_wh"])
    floor_pct = float(bundle["physics"]["walkback_reserve_fraction"]) * 100.0
    remaining_pct = np.clip((cap - te) / cap * 100.0, 0.0, 100.0)

    ax2.fill_between(td, tz, 0.0, color="#d77847", alpha=0.13, zorder=1)
    ax2.plot(td, tz, color="#ffb374", lw=1.45, zorder=3, label="Twin elevation")
    ax2.set_xlabel("cumulative route distance (km)", fontsize=7.6)
    ax2.set_ylabel("elevation above base (m)", fontsize=7.6, color="#ffc397")
    ax2.tick_params(axis="y", colors="#ffc397", labelsize=6.8)
    ax2.tick_params(axis="x", labelsize=6.8)
    ax2.set_xlim(0.0, max(td.max(), 0.05))
    y_pad = max(5.0, (tz.max() - tz.min()) * 0.35 if tz.max() > tz.min() else 5.0)
    ax2.set_ylim(min(-3.0, tz.min() - y_pad), tz.max() + y_pad)
    ax2.grid(color="#22303c", lw=0.45, ls=":", alpha=0.75, zorder=0)

    ax3 = ax2.twinx()
    ax3.plot(td, remaining_pct, color="#68e1ee", lw=1.7, zorder=4, label="PLSS remaining")
    ax3.axhline(floor_pct, color="#ff5d4a", lw=1.05, ls=(0, (4, 2)), zorder=2,
                label="25 % reserve floor")
    ax3.fill_between(td, 0.0, floor_pct, color="#ff5d4a", alpha=0.07, zorder=0)
    ax3.set_ylim(0.0, 103.0)
    ax3.set_ylabel("PLSS remaining (% of budget)", fontsize=7.6, color="#78e5f1")
    ax3.tick_params(axis="y", colors="#78e5f1", labelsize=6.8)

    # Station call-outs.
    for s in bundle["stations"]:
        m = s.get("route", {})
        if m.get("distance_m") is None:
            continue
        sx = float(m["distance_m"]) / 1000.0
        ax2.axvline(sx, color="#5a6b79", lw=0.55, ls=":", alpha=0.75, zorder=2)
        ax2.scatter([sx], [s.get("terrain", {}).get("elev_rel_m", 0.0)], s=26,
                    facecolor="#ff8d4d", edgecolor="#160b08", lw=0.5, zorder=6)
        if m.get("margin_wh") is not None:
            color = "#6fe08a" if m["margin_wh"] >= 0 else "#ff6254"
            ax3.scatter([sx], [float(m["margin_wh"]) / cap * 100.0 + floor_pct], s=13,
                        facecolor=color, edgecolor="#071018", lw=0.4, zorder=7)

    stats = bundle["route"]["stats"]
    reserve = bundle["route"]["reserve"]
    profile_text = (
        f'DIST {stats["distance_m"]/1000:.2f} km   ·   WALK {stats["walk_time_s"]/60:.1f} min   ·   '
        f'DWELL {stats["dwell_time_s"]/60:.1f} min\n'
        f'METABOLIC {stats["total_energy_wh"]:.1f} Wh   ·   REMAIN {stats["remaining_wh"]:.1f} Wh '
        f'({stats["remaining_pct"]:.0f}%)\n'
        f'FINAL WALK-BACK {reserve["floor_wh"]:.0f} Wh floor + '
        f'{stats["final_walkback_energy_wh"]:.1f} Wh path\n'
        f'STRICT MARGIN {reserve["worst_margin_wh"]:+.1f} Wh   ·   '
        f'ASYMMETRY {stats["asymmetry_pct"]:+.1f}%'
    )
    ax2.text(
        0.022, 0.035, profile_text, transform=ax2.transAxes, fontsize=6.2,
        color="#c9d5df", va="bottom", ha="left", linespacing=1.55,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#0a1017", edgecolor="#31404f", alpha=0.92),
        zorder=10,
    )

    lines = [
        Line2D([0], [0], color="#ffb374", lw=1.4, label="elevation"),
        Line2D([0], [0], color="#68e1ee", lw=1.5, label="PLSS remaining"),
        Line2D([0], [0], color="#ff5d4a", lw=1.0, ls=(0, (4, 2)), label="25 % floor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6fe08a",
               markeredgecolor="#071018", markersize=4.5, label="station margin OK"),
    ]
    leg2 = ax2.legend(
        handles=lines, loc="upper right", fontsize=5.9, ncol=2,
        facecolor="#0a1017", edgecolor="#2b3945", framealpha=0.88,
        handlelength=2.2, borderpad=0.5, labelcolor="#cfdae4",
    )
    leg2.set_zorder(20)
    ax2.set_title(
        "B  ·  0.38 g METABOLIC PROFILE  ·  25 % PLSS WALKBACK RESERVE",
        loc="left", fontsize=8.7, color="#eaf2f8", fontweight="bold", pad=7,
    )
    for spine in ax2.spines.values():
        spine.set_color("#31404f")
    for spine in ax3.spines.values():
        spine.set_color("#31404f")

    # Header / footer.
    fig.text(
        0.045, 0.945, "ARES-WALK", color="#f4f7f9", fontsize=18, fontweight="bold",
        ha="left", va="center",
    )
    fig.text(
        0.196, 0.945, "ALTADENA BOXWORK EVA  /  MSL SOLS 4535-4611",
        color="#ffad72", fontsize=8.8, fontweight="bold", ha="left", va="center",
    )
    fig.text(
        0.045, 0.905,
        "0.38 g suited traversal  ·  HiRISE-style TPI  ·  THEMIS-style thermal inertia  ·  "
        "CRISM-style D2300  ·  ray-traced RF  ·  asymmetric Dijkstra",
        color="#8e9eac", fontsize=7.3, ha="left", va="center",
    )
    fig.text(
        0.045, 0.035,
        "Real Curiosity waypoints + Mastcam; terrain physics fields are a calibrated scenario twin. "
        "Waypoints: NASA/JPL MMGIS. Images: NASA/JPL-Caltech/MSSS. Not a flight plan.",
        color="#738391", fontsize=6.2, ha="left", va="center",
    )
    fig.text(
        0.955, 0.035, "VERIFICATION SHEET  /  ARES-WALK v1.0",
        color="#738391", fontsize=6.2, ha="right", va="center",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return out_path


def main() -> None:
    if BUNDLE_PATH.exists():
        with BUNDLE_PATH.open("r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        source = str(BUNDLE_PATH)
    else:
        scene = build_scene(
            FALLBACK_WAYPOINTS,
            DEFAULT_STATION_SPECS,
            data_sources={
                "mode": "engine_standalone",
                "note": "mission_bundle.json missing; verified fallback waypoints used",
            },
        )
        bundle = scene["bundle"]
        source = "built-in verified waypoint fallback"
    out = render_plan(bundle, PLOT_PATH)
    stats = bundle["route"]["stats"]
    print(f"ARES-WALK plan source: {source}")
    print(f"route: {stats['distance_m']:.1f} m, {stats['total_time_s']/60:.1f} min, "
          f"{stats['total_energy_wh']:.1f} Wh, {stats['feasible_station_count']}/{stats['station_count']} GO")
    print(f"strict reserve margin: {bundle['route']['reserve']['worst_margin_wh']:+.2f} Wh")
    print(f"verification plot: {out}")


if __name__ == "__main__":
    main()



