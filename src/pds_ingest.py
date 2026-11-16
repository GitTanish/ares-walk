#!/usr/bin/env python3
"""
ARES-WALK :: NASA/JPL PDS ingest and bundle compiler
====================================================

Fetches the real Curiosity waypoint layer and Mastcam (MAST_LEFT) frames for
MSL sols 4535-4611, optionally ingests local GeoTIFF/TIFF/IMG crops from
data/raw/, builds the calibrated boxwork twin, runs the EVA planner, and writes
data/mission_bundle.json. A compact fallback twin is synchronised into
index.html so the page still works when opened via file://.

Usage:
    python src/pds_ingest.py
    python src/pds_ingest.py --offline          # verified fallback data only
    python src/pds_ingest.py --no-plot          # skip the PNG regeneration

Optional local rasters (install rasterio first):
    data/raw/*dtm*.tif  or  *dem*.tif  or  *elev*.tif   -> elevation (metres)
    data/raw/*themis*.tif or *inertia*.tif             -> thermal inertia
    data/raw/*crism*.tif  or *d2300*.tif               -> D2300 absorption
Rasters are resampled to the 72 x 72 EVA analysis grid. For best results supply
pre-cropped rasters covering the mission box; this keeps the ingest dependency
light and avoids silently mis-registering full planetary scenes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ASSETS_DIR = ROOT / "assets"
BUNDLE_PATH = DATA_DIR / "mission_bundle.json"
INDEX_PATH = ROOT / "index.html"
PLOT_PATH = ASSETS_DIR / "mount_sharp_eva_plan.png"
MANIFEST_PATH = ASSETS_DIR / "panos" / "manifest.json"

# Mars 2000 sphere equirectangular CRS constants
MARS_RADIUS = 3396190.0  # meters
GALE_LAT_REF = -4.80     # degrees South
GALE_LON_REF = 137.38    # degrees East
ORIGIN_EASTING = 8142797.662   # lower-left corner of 72x72 analysis grid
ORIGIN_NORTHING = -285119.519  # lower-left corner of 72x72 analysis grid
CELL_M = 15.0

# Local imports work both for `python src/pds_ingest.py` and from a notebook.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mars_eva_engine import (  # noqa: E402
    ASSETS_DIR as ENGINE_ASSETS_DIR,
    DEFAULT_STATION_SPECS,
    FALLBACK_WAYPOINTS,
    GRID_H,
    GRID_W,
    MARGIN_M,
    build_scene,
    project_track,
    render_plan,
)

WAYPOINTS_URL = "https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json"
IMAGES_API = "https://mars.nasa.gov/api/v1/raw_image_items"
MMGIS_LAYER = "MSL_waypoints"
SOL_MIN, SOL_MAX = 4535, 4611

USER_AGENT = "ARES-WALK/1.0 (EVA planning demo; NASA open data; contact: local)"
HTTP_TIMEOUT = 45

# Curated MAST_LEFT frames from the exact station sols. These were selected
# from the NASA raw-image API and verified during development. They are the
# offline fallback and the preferred images when the API is reachable.
STATION_IMAGE_CURATION: dict[int, list[dict[str, Any]]] = {
    4539: [
        {"imageid": "4539ML1075910171800987C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04539/mcam/4539ML1075910171800987C00_DXXX.jpg", "title": "Sol 4539: Mast Camera (Mastcam)", "date_taken": "2025-05-13T21:56:16.000Z"},
        {"imageid": "4539ML1075950001800996C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04539/mcam/4539ML1075950001800996C00_DXXX.jpg", "title": "Sol 4539: Mast Camera (Mastcam)", "date_taken": "2025-05-14T00:32:36.000Z"},
        {"imageid": "4539ML1075910071800977C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04539/mcam/4539ML1075910071800977C00_DXXX.jpg", "title": "Sol 4539: Mast Camera (Mastcam)", "date_taken": "2025-05-13T21:53:30.000Z"},
    ],
    4546: [
        {"imageid": "4546ML1076320021801419C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04546/mcam/4546ML1076320021801419C00_DXXX.jpg", "title": "Sol 4546: Mast Camera (Mastcam)", "date_taken": "2025-05-21T01:16:00.000Z"},
        {"imageid": "4546ML1076330081801442C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04546/mcam/4546ML1076330081801442C00_DXXX.jpg", "title": "Sol 4546: Mast Camera (Mastcam)", "date_taken": "2025-05-21T01:21:03.000Z"},
        {"imageid": "4546ML1076360011801453C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04546/mcam/4546ML1076360011801453C00_DXXX.jpg", "title": "Sol 4546: Mast Camera (Mastcam)", "date_taken": "2025-05-21T01:24:49.000Z"},
    ],
    4554: [
        {"imageid": "4554ML1076830231801736C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04554/mcam/4554ML1076830231801736C00_DXXX.jpg", "title": "Sol 4554: Mast Camera (Mastcam)", "date_taken": "2025-05-29T07:31:30.000Z"},
        {"imageid": "4554ML1076830111801724C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04554/mcam/4554ML1076830111801724C00_DXXX.jpg", "title": "Sol 4554: Mast Camera (Mastcam)", "date_taken": "2025-05-29T07:28:32.000Z"},
        {"imageid": "4554ML1076850061801749C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04554/mcam/4554ML1076850061801749C00_DXXX.jpg", "title": "Sol 4554: Mast Camera (Mastcam)", "date_taken": "2025-05-29T10:27:00.000Z"},
    ],
    4560: [
        {"imageid": "4560ML1077200001801854C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04560/mcam/4560ML1077200001801854C00_DXXX.jpg", "title": "Sol 4560: Mast Camera (Mastcam)", "date_taken": "2025-06-04T13:14:47.000Z"},
        {"imageid": "4560ML1077200331801887C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04560/mcam/4560ML1077200331801887C00_DXXX.jpg", "title": "Sol 4560: Mast Camera (Mastcam)", "date_taken": "2025-06-04T13:21:44.000Z"},
        {"imageid": "4560ML1077200071801861C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04560/mcam/4560ML1077200071801861C00_DXXX.jpg", "title": "Sol 4560: Mast Camera (Mastcam)", "date_taken": "2025-06-04T13:16:17.000Z"},
    ],
    4571: [
        {"imageid": "4571ML1077870131802066C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04571/mcam/4571ML1077870131802066C00_DXXX.jpg", "title": "Sol 4571: Mast Camera (Mastcam)", "date_taken": "2025-06-15T21:34:15.000Z"},
        {"imageid": "4571ML1077870091802062C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04571/mcam/4571ML1077870091802062C00_DXXX.jpg", "title": "Sol 4571: Mast Camera (Mastcam)", "date_taken": "2025-06-15T21:33:20.000Z"},
        {"imageid": "4571ML1077860061802049C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04571/mcam/4571ML1077860061802049C00_DXXX.jpg", "title": "Sol 4571: Mast Camera (Mastcam)", "date_taken": "2025-06-15T21:30:24.000Z"},
    ],
    4580: [
        {"imageid": "4580ML1078400001802486C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04580/mcam/4580ML1078400001802486C00_DXXX.jpg", "title": "Sol 4580: Mast Camera (Mastcam)", "date_taken": "2025-06-25T04:01:37.000Z"},
        {"imageid": "4580ML1078340061802466C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04580/mcam/4580ML1078340061802466C00_DXXX.jpg", "title": "Sol 4580: Mast Camera (Mastcam)", "date_taken": "2025-06-25T00:00:55.000Z"},
        {"imageid": "4580ML1078410101802506C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04580/mcam/4580ML1078410101802506C00_DXXX.jpg", "title": "Sol 4580: Mast Camera (Mastcam)", "date_taken": "2025-06-25T04:06:02.000Z"},
    ],
    4588: [
        {"imageid": "4588ML1078820051802814C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04588/mcam/4588ML1078820051802814C00_DXXX.jpg", "title": "Sol 4588: Mast Camera (Mastcam)", "date_taken": "2025-07-03T09:23:27.000Z"},
        {"imageid": "4588ML1078760191802801C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04588/mcam/4588ML1078760191802801C00_DXXX.jpg", "title": "Sol 4588: Mast Camera (Mastcam)", "date_taken": "2025-07-03T05:54:32.000Z"},
        {"imageid": "4588ML1078750211802775C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04588/mcam/4588ML1078750211802775C00_DXXX.jpg", "title": "Sol 4588: Mast Camera (Mastcam)", "date_taken": "2025-07-03T05:46:20.000Z"},
    ],
    4596: [
        {"imageid": "4596ML1079290081803072C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04596/mcam/4596ML1079290081803072C00_DXXX.jpg", "title": "Sol 4596: Mast Camera (Mastcam)", "date_taken": "2025-07-11T11:53:36.000Z"},
        {"imageid": "4596ML1079290021803066C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04596/mcam/4596ML1079290021803066C00_DXXX.jpg", "title": "Sol 4596: Mast Camera (Mastcam)", "date_taken": "2025-07-11T11:52:28.000Z"},
        {"imageid": "4596ML1079290051803069C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04596/mcam/4596ML1079290051803069C00_DXXX.jpg", "title": "Sol 4596: Mast Camera (Mastcam)", "date_taken": "2025-07-11T11:53:04.000Z"},
    ],
    4602: [
        {"imageid": "4602ML1079690001803352C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04602/mcam/4602ML1079690001803352C00_DXXX.jpg", "title": "Sol 4602: Mast Camera (Mastcam)", "date_taken": "2025-07-17T18:53:36.000Z"},
        {"imageid": "4602ML1079690101803362C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04602/mcam/4602ML1079690101803362C00_DXXX.jpg", "title": "Sol 4602: Mast Camera (Mastcam)", "date_taken": "2025-07-17T18:55:54.000Z"},
        {"imageid": "4602ML1079640061803341C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04602/mcam/4602ML1079640061803341C00_DXXX.jpg", "title": "Sol 4602: Mast Camera (Mastcam)", "date_taken": "2025-07-17T14:43:00.000Z"},
    ],
    4609: [
        {"imageid": "4609ML1079970101803512C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04609/mcam/4609ML1079970101803512C00_DXXX.jpg", "title": "Sol 4609: Mast Camera (Mastcam)", "date_taken": "2025-07-24T21:37:06.000Z"},
        {"imageid": "4609ML1079940231803490C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04609/mcam/4609ML1079940231803490C00_DXXX.jpg", "title": "Sol 4609: Mast Camera (Mastcam)", "date_taken": "2025-07-24T18:43:05.000Z"},
        {"imageid": "4609ML1079940131803480C00_DXXX", "url": "https://mars.nasa.gov/msl-raw-images/msss/04609/mcam/4609ML1079940131803480C00_DXXX.jpg", "title": "Sol 4609: Mast Camera (Mastcam)", "date_taken": "2025-07-24T18:40:57.000Z"},
    ],
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(url: str, method: str = "GET", *, timeout: int = HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*,*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def fetch_json(url: str, *, retries: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = _request(url)
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def image_url_ok(url: str) -> bool:
    """Verify a JPL image exists. HEAD, with a Range GET fallback."""
    try:
        _request(url, method="HEAD", timeout=20)
        return True
    except Exception:
        pass
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Range": "bytes=0-512", "Accept": "image/*"},
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            return int(response.status) in (200, 206)
    except Exception:
        return False


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Real MSL data: waypoints + Mastcam
# ---------------------------------------------------------------------------

def fetch_waypoints(offline: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status: dict[str, Any] = {
        "url": WAYPOINTS_URL,
        "layer": MMGIS_LAYER,
        "sol_window": [SOL_MIN, SOL_MAX],
        "status": "verified_offline_fallback",
        "fetched_utc": None,
        "count": 0,
    }
    if not offline:
        try:
            geojson = fetch_json(WAYPOINTS_URL, retries=1)
            features = geojson.get("features", [])
            points = []
            for feat in features:
                props = feat.get("properties", {}) or {}
                if props.get("drive_type") != "final":
                    continue
                sol = int(props.get("sol", -1))
                if not (SOL_MIN <= sol <= SOL_MAX):
                    continue
                if not all(k in props for k in ("easting", "northing", "elev_geoid", "lat", "lon")):
                    continue
                points.append({
                    "sol": sol,
                    "site": int(props.get("site", 0)),
                    "drive": int(props.get("drive", 0)),
                    "easting": float(props["easting"]),
                    "northing": float(props["northing"]),
                    "elev_geoid": float(props["elev_geoid"]),
                    "lat": float(props["lat"]),
                    "lon": float(props["lon"]),
                    "dist_m": float(props.get("dist_m", 0.0)),
                })
            points.sort(key=lambda p: p["sol"])
            if points:
                status.update({
                    "status": "live",
                    "fetched_utc": _now_utc(),
                    "count": len(points),
                    "source_features": len(features),
                })
                return points, status
            status["error"] = "live layer returned no final-drive points in window"
        except Exception as exc:
            status["error"] = str(exc)[:240]
    fallback = [dict(p) for p in FALLBACK_WAYPOINTS if SOL_MIN <= int(p["sol"]) <= SOL_MAX]
    status["count"] = len(fallback)
    return fallback, status


def fetch_mastcam_items(sol: int) -> tuple[list[dict[str, Any]], int | None]:
    query = urllib.parse.urlencode({
        "order": "sol asc",
        "per_page": 100,
        "page": 0,
        "condition_1": "msl:mission",
        "condition_2": f"{sol}:sol",
        "condition_3": "MAST_LEFT:instrument",
    })
    payload = fetch_json(f"{IMAGES_API}?{query}", retries=1)
    items = payload.get("items", []) or []
    total = payload.get("total")
    return items, int(total) if total is not None else None


def _curated(sol: int) -> list[dict[str, Any]]:
    return [dict(x) for x in STATION_IMAGE_CURATION.get(sol, [])]


def choose_station_image(
    sol: int,
    *,
    offline: bool,
    verify: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick a real MAST_LEFT frame, with curated verified URLs as fallback."""
    report: dict[str, Any] = {
        "sol": sol,
        "api_status": "offline_curated" if offline else "unavailable",
        "api_items": 0,
        "verified_url": None,
    }
    curated = _curated(sol)
    items: list[dict[str, Any]] = []
    if not offline:
        try:
            items, total = fetch_mastcam_items(sol)
            report["api_status"] = "live"
            report["api_items"] = len(items)
            report["api_total"] = total
        except Exception as exc:
            report["error"] = str(exc)[:240]

    api_by_id = {str(item.get("imageid")): item for item in items}
    merged: list[dict[str, Any]] = []

    # 1. Curated primary is preferred (it is a deliberate scenic frame).
    for c in curated:
        item = api_by_id.get(c["imageid"], {})
        merged.append({**c, **{k: item[k] for k in ("title", "url", "date_taken") if item.get(k)}})

    # 2. Add other API C00 frames as extra fallbacks.
    for item in items:
        imageid = str(item.get("imageid", ""))
        url = item.get("url")
        if not url or "C00_DXXX.jpg" not in url:
            continue
        if any(m.get("imageid") == imageid for m in merged):
            continue
        merged.append({
            "imageid": imageid,
            "url": url,
            "title": item.get("title"),
            "date_taken": item.get("date_taken"),
            "imageid_raw": imageid,
        })
        if len(merged) >= 8:
            break

    if not merged:
        # Last-resort: construct a raw URL from the curated ID pattern.
        merged = curated

    # 3. Verify candidates in order when online. Offline keeps curated URLs.
    chosen = merged[0]
    if not offline and verify:
        for candidate in merged[:6]:
            if image_url_ok(str(candidate["url"])):
                chosen = candidate
                report["verified_url"] = candidate["url"]
                break
        else:
            report["verify_warning"] = "no candidate passed URL verification; kept curated primary"
    elif offline:
        report["verified_url"] = chosen.get("url")

    fallbacks = []
    for candidate in merged[1:6]:
        url = candidate.get("url")
        if url and url != chosen.get("url") and url not in fallbacks:
            fallbacks.append(url)

    image = {
        "url": chosen.get("url"),
        "sol": sol,
        "title": chosen.get("title") or f"Sol {sol}: Mast Camera (Mastcam)",
        "imageid": chosen.get("imageid"),
        "date_taken": chosen.get("date_taken"),
        "fallback_urls": fallbacks[:3],
        "page_url": f"https://mars.nasa.gov/msl/multimedia/raw-images/?instrument=MAST_LEFT&sol={sol}",
        "verified": bool(not offline and report.get("verified_url")),
        "offline_curated": bool(offline),
    }
    report["chosen"] = image["imageid"]
    report["fallbacks"] = len(fallbacks)
    return image, report


def attach_images(
    station_specs: list[dict[str, Any]],
    *,
    offline: bool,
    verify: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs = copy.deepcopy(station_specs)
    reports: list[dict[str, Any]] = []
    for spec in specs:
        sol = int(spec["sol"])
        print(f"  Mastcam sol {sol} ...", end=" ", flush=True)
        image, report = choose_station_image(sol, offline=offline, verify=verify)
        spec["image"] = image
        reports.append(report)
        flag = "OK" if image.get("url") else "MISS"
        print(f"{flag}  {image.get('imageid')}  fallbacks={len(image.get('fallback_urls', []))}")
    return specs, reports


# ---------------------------------------------------------------------------
# Optional local raster ingestion
# ---------------------------------------------------------------------------

RASTER_SUFFIXES = {".tif", ".tiff", ".img", ".jp2", ".vrt"}


def _resize_to_grid(array: np.ndarray) -> np.ndarray:
    """Resample an arbitrary raster band to the 72 x 72 analysis grid."""
    from scipy import ndimage

    arr = np.asarray(array, dtype=float)
    bad = ~np.isfinite(arr)
    if bad.all():
        raise ValueError("raster band contains no finite values")
    if bad.any():
        arr = arr.copy()
        arr[bad] = np.nanmedian(arr)
    if arr.shape == (GRID_H, GRID_W):
        return arr
    factors = (GRID_H / arr.shape[0], GRID_W / arr.shape[1])
    out = ndimage.zoom(arr, factors, order=1, mode="nearest")
    if out.shape[0] < GRID_H or out.shape[1] < GRID_W:
        pad_y = max(0, GRID_H - out.shape[0])
        pad_x = max(0, GRID_W - out.shape[1])
        out = np.pad(out, ((0, pad_y), (0, pad_x)), mode="edge")
    out = out[:GRID_H, :GRID_W]
    return out


def _normalise_layer(array: np.ndarray, layer: str) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if layer == "thermal_inertia":
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi <= 1.5:
            return 150.0 + 330.0 * np.clip(arr, 0.0, 1.0)
        if hi <= 120.0:
            return 150.0 + 330.0 * np.clip(arr / max(hi, 1e-9), 0.0, 1.0)
        return np.clip(arr, 145.0, 490.0)
    if layer == "d2300":
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if hi <= 0.08:
            return np.clip(arr, 0.0, 0.06)
        if hi <= 1.0:
            return np.clip(arr, 0.0, 1.0) * 0.055
        lo, hi = np.nanpercentile(arr, [2.0, 98.0])
        return np.clip((arr - lo) / max(hi - lo, 1e-9) * 0.05, 0.0, 0.06)
    return arr


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# Mars 2000 sphere, used as the geographic hub for all raster reprojection.
MARS_GEO_PROJ4 = "+proj=longlat +a=3396190 +b=3396190 +no_defs"


def _enu_to_lonlat(easting: np.ndarray, northing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mars 2000 sphere equirectangular: x = R*lon, y = R*lat (radians)."""
    lon = np.degrees(np.asarray(easting, dtype=float) / MARS_RADIUS)
    lat = np.degrees(np.asarray(northing, dtype=float) / MARS_RADIUS)
    return lon, lat


def _sample_band_on_enu_grid(
    band: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    origin_easting: float,
    origin_northing: float,
    width: int,
    height: int,
    cell_m: float,
) -> tuple[np.ndarray, str]:
    """Bilinearly sample a georeferenced band at ENU cell centres.

    The target grid is the same 72 x 72 Mars 2000 sphere ENU grid used by the
    EVA engine.  Points are converted ENU -> lon/lat -> source CRS, then
    sampled in source pixel space.  This is a real georeferenced resample, not
    a stretch of the raster's pixel array.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform
    from scipy import ndimage

    cols = np.arange(width, dtype=float)
    rows = np.arange(height, dtype=float)
    east = origin_easting + (cols + 0.5) * cell_m
    north = origin_northing + (rows + 0.5) * cell_m
    east_grid, north_grid = np.meshgrid(east, north)
    lon, lat = _enu_to_lonlat(east_grid, north_grid)

    if crs is None:
        raise ValueError("raster has no CRS; cannot georeference onto the ENU grid")

    # Always reproject through rasterio so planetographic latitude, prime
    # meridian conventions, and projected Mars CRSs are all handled correctly.
    hub = CRS.from_proj4(MARS_GEO_PROJ4)
    source_x, source_y = warp_transform(
        hub, crs, lon.ravel().tolist(), lat.ravel().tolist()
    )
    source_x = np.asarray(source_x, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    sampling = (
        "enu_bilinear_geographic" if crs.is_geographic else "enu_bilinear_projected"
    )

    inv = ~transform
    pixel_x, pixel_y = inv * (source_x, source_y)
    pixel_x = np.asarray(pixel_x, dtype=float)
    pixel_y = np.asarray(pixel_y, dtype=float)

    band = np.asarray(band, dtype=float)
    src_h, src_w = band.shape
    in_bounds = (
        (pixel_x >= -0.5) & (pixel_x <= src_w - 0.5)
        & (pixel_y >= -0.5) & (pixel_y <= src_h - 0.5)
    )
    sampled = ndimage.map_coordinates(
        band,
        np.vstack([np.clip(pixel_y, 0, src_h - 1), np.clip(pixel_x, 0, src_w - 1)]),
        order=1,
        mode="nearest",
    )
    sampled = sampled.reshape(height, width)
    sampled[~in_bounds.reshape(height, width)] = np.nan
    if not np.isfinite(sampled).any():
        raise ValueError(
            "no raster pixels intersect the mission ENU box; check the CRS/product footprint"
        )
    return sampled, sampling


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    bad = ~np.isfinite(out)
    if bad.all():
        raise ValueError("raster band contains no finite values")
    if bad.any():
        out[bad] = np.nanmedian(out)
    return out


def attach_panoramas(stations: list[dict[str, Any]], manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Wire stitched landmark panoramas into station specs."""
    default_panos = {
        "ALT-05": "assets/panos/alt05_layered_spur_sol4571.jpg",
        "ALT-10": "assets/panos/alt10_upper_bench_sol4609.jpg",
    }
    landmarks = {}
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            landmarks = manifest_data.get("landmarks", {})
        except Exception as exc:
            print(f"  [warn] Could not load panorama manifest: {exc}")

    for st in stations:
        st_id = st.get("id")
        if st_id in landmarks:
            lm = landmarks[st_id]
            st["pano_url"] = lm.get("pano_url")
            st["pano"] = {
                "file_name": lm.get("file_name"),
                "dimensions": lm.get("dimensions"),
                "frame_count": lm.get("frame_count"),
                "az_range": lm.get("az_range"),
                "el_range": lm.get("el_range"),
                "sha256": lm.get("sha256"),
            }
        elif st_id in default_panos and (ROOT / default_panos[st_id]).exists():
            st["pano_url"] = default_panos[st_id]
            st["pano"] = {
                "file_name": Path(default_panos[st_id]).name,
                "sha256": _file_sha256(ROOT / default_panos[st_id]),
            }

    return stations


def load_local_rasters(
    raw_dir: Path = RAW_DIR,
    *,
    origin_easting: float = ORIGIN_EASTING,
    origin_northing: float = ORIGIN_NORTHING,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read and georeference optional rasters onto the 72 x 72 ENU grid.

    Rasters are converted ENU -> Mars 2000 lon/lat -> source CRS and sampled at
    cell centres.  A raster with no CRS is resized as a last resort and marked
    ``resized_no_crs`` so provenance stays honest.  Missing rasterio is never
    fatal: the calibrated twin is used instead.
    """
    report: dict[str, Any] = {
        "status": "not_used",
        "raw_dir": str(raw_dir),
        "files": [],
        "used_layers": [],
        "layers": {
            "elevation": {"status": "fallback", "source": "calibrated_digital_twin"},
            "thermal_inertia": {"status": "fallback", "source": "calibrated_digital_twin"},
            "d2300": {"status": "fallback", "source": "calibrated_digital_twin"},
        },
        "grid_crs": "Mars 2000 sphere equirectangular (R=3396190 m, Gale 4.80S, 137.38E)",
        "grid": {
            "width": GRID_W,
            "height": GRID_H,
            "cell_m": CELL_M,
            "origin_easting": round(float(origin_easting), 3),
            "origin_northing": round(float(origin_northing), 3),
            "sample": "cell centres, easting/northing increasing with column/row",
        },
        "note": "Pre-cropped local rasters are optional; the calibrated twin is the fallback.",
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in RASTER_SUFFIXES and not p.name.startswith(".")
    )
    if not files:
        return {}, report
    try:
        import rasterio
    except Exception:
        report["status"] = "rasterio_missing"
        report["message"] = "Rasters found but rasterio is not installed; using calibrated twin."
        return {}, report

    layers: dict[str, np.ndarray] = {}
    used = []
    for path in files:
        name = path.stem.lower()
        layer = None
        if re.search(r"(^|[_\-. ])(dtm|dem|elev|elevation)([_\-.]|$)", name) or (
            "hirise" in name and "dtm" in name
        ):
            layer = "elevation"
        elif any(token in name for token in ("themis", "inertia", "thermal")):
            layer = "thermal_inertia"
        elif any(token in name for token in ("crism", "d2300", "mineral")):
            layer = "d2300"
        if layer is None or layer in layers:
            continue

        file_sha256 = _file_sha256(path)
        pds_product_id = None
        pds_label_path = None
        for cand in [path.with_suffix(".xml"), path.with_suffix(".lbl"), path.with_suffix(".LBL")]:
            if cand.exists():
                try:
                    txt = cand.read_text(encoding="utf-8", errors="ignore")
                    pds_label_path = _display_path(cand)
                    m = re.search(r"urn:nasa:pds:[^\s<>\"]+", txt)
                    if m:
                        pds_product_id = m.group(0)
                        break
                    m = re.search(r"PRODUCT_ID\s*=\s*\"?([^\s\"]+)\"?", txt)
                    if m:
                        pds_product_id = m.group(1)
                        break
                except Exception:
                    pass
        if not pds_product_id:
            pds_product_id = path.stem.upper()

        try:
            with rasterio.open(path) as src:
                band = np.asarray(src.read(1, masked=True).filled(np.nan), dtype=float)
                raster_crs = str(src.crs) if src.crs else None
                if src.crs:
                    arr, sampling = _sample_band_on_enu_grid(
                        band,
                        src.transform,
                        src.crs,
                        origin_easting=origin_easting,
                        origin_northing=origin_northing,
                        width=GRID_W,
                        height=GRID_H,
                        cell_m=CELL_M,
                    )
                    arr = _fill_nan(arr)
                else:
                    # Keep a documented fallback rather than pretending a
                    # stretched array is georeferenced.
                    arr = _resize_to_grid(np.flipud(band))
                    sampling = "resized_no_crs"
                arr = _normalise_layer(arr, layer)
                bounds = src.bounds
                source_meta = {
                    "source_shape": [int(src.height), int(src.width)],
                    "source_bounds": [
                        round(float(bounds.left), 6), round(float(bounds.bottom), 6),
                        round(float(bounds.right), 6), round(float(bounds.top), 6),
                    ],
                    "source_transform": [round(float(v), 9) for v in tuple(src.transform)[:6]],
                }
            layers[layer] = arr
            layer_record = {
                "path": _display_path(path),
                "layer": layer,
                "shape": [GRID_H, GRID_W],
                "crs": raster_crs or "unknown",
                "sampling": sampling,
                "sha256": file_sha256,
                "pds_product_id": pds_product_id,
                "pds_label_path": pds_label_path,
                **source_meta,
            }
            used.append(layer_record)
            report["layers"][layer] = {
                "status": "loaded",
                "source": "pds_raster",
                "sha256": file_sha256,
                "pds_product_id": pds_product_id,
                "pds_label_path": pds_label_path,
                "path": _display_path(path),
                "crs": raster_crs or "unknown",
                "sampling": sampling,
                "grid_crs": report["grid_crs"],
                "grid_origin_easting": round(float(origin_easting), 3),
                "grid_origin_northing": round(float(origin_northing), 3),
                **source_meta,
            }
            print(
                f"  raster: {path.name} -> {layer} "
                f"({sampling}, PDS: {pds_product_id}, SHA: {file_sha256[:12]})"
            )
        except Exception as exc:
            report.setdefault("errors", []).append(f"{path.name}: {str(exc)[:220]}")

    if layers:
        report["status"] = "loaded"
        report["used_layers"] = sorted(layers.keys())
        report["files"] = used
        report["note"] = (
            "Rasters reprojected and sampled at ENU cell centres on the Gale Crater grid "
            "(Mars 2000 sphere, 4.80°S, 137.38°E). Unsupplied layers remain calibrated twin."
        )
    else:
        report["status"] = "no_usable_layers"
    return layers, report


# ---------------------------------------------------------------------------
# Embedded fallback twin (for file:// and fetch failures)
# ---------------------------------------------------------------------------

EMBED_START = "/* ARES_EMBEDDED_TWIN_START */"
EMBED_END = "/* ARES_EMBEDDED_TWIN_END */"


def _downsample_field(values: list[int], h: int, w: int, factor: int, mode: str) -> list[int]:
    arr = np.asarray(values, dtype=float).reshape(h, w)
    new_h, new_w = h // factor, w // factor
    arr = arr[: new_h * factor, : new_w * factor].reshape(new_h, factor, new_w, factor)
    if mode == "max":
        out = arr.max(axis=(1, 3))
    else:
        out = arr.mean(axis=(1, 3))
    return np.rint(out).astype(int).ravel().tolist()


def make_embedded_fallback(bundle: dict[str, Any], factor: int = 2) -> dict[str, Any]:
    """Compact the analysis grid while preserving stations, route and imagery."""
    fb = copy.deepcopy(bundle)
    t = bundle["terrain"]
    h, w = int(t["width"]), int(t["height"])
    factor = max(1, int(factor))
    if factor == 1:
        fb["embedded_fallback"] = True
        return fb
    new_h, new_w = h // factor, w // factor
    new_cell = float(t["cell_m"]) * factor
    fields = {
        "elevation_dm": "mean",
        "slope_deg_x10": "mean",
        "tpi_dm": "mean",
        "thermal_inertia": "mean",
        "d2300_x1000": "mean",
        "sand_x100": "mean",
        "hazard": "max",
        "rf_shadow": "max",
        "ridge": "max",
    }
    out_terrain = {k: v for k, v in t.items() if k not in fields}
    out_terrain.update({
        "width": new_w,
        "height": new_h,
        "cell_m": new_cell,
        "embedded_scale": factor,
    })
    for key, mode in fields.items():
        out_terrain[key] = _downsample_field(t[key], h, w, factor, mode)
    fb["terrain"] = out_terrain
    fb["embedded_fallback"] = True
    fb["embedded_note"] = (
        "Compact twin embedded in index.html. Route and station coordinates are full-resolution; "
        "terrain fields are block-averaged for file:// fallback."
    )
    # Recompute indices that referenced the fine grid.
    x0, y0 = float(t["x0"]), float(t["y0"])
    for obj in [fb.get("base", {})] + list(fb.get("stations", [])):
        x, y = float(obj.get("x", 0.0)), float(obj.get("y", 0.0))
        i = int(np.clip(round((x - x0) / new_cell - 0.5), 0, new_w - 1))
        j = int(np.clip(round((y - y0) / new_cell - 0.5), 0, new_h - 1))
        obj["embedded_node"] = j * new_w + i
    for relay in out_terrain.get("relays", []):
        x, y = float(relay.get("x", 0.0)), float(relay.get("y", 0.0))
        relay["embedded_i"] = int(np.clip(round((x - x0) / new_cell - 0.5), 0, new_w - 1))
        relay["embedded_j"] = int(np.clip(round((y - y0) / new_cell - 0.5), 0, new_h - 1))
    return fb


def update_index_fallback(bundle: dict[str, Any], index_path: Path = INDEX_PATH) -> bool:
    """Replace the compact fallback object between marker comments, if present."""
    if not index_path.exists():
        print("  index.html not found; embedded fallback not updated")
        return False
    text = index_path.read_text(encoding="utf-8")
    start = text.find(EMBED_START)
    end = text.find(EMBED_END)
    if start < 0 or end < 0 or end <= start:
        print("  embedded-twin markers not found in index.html; skipped")
        return False
    fallback = make_embedded_fallback(bundle, factor=2)
    payload = json.dumps(fallback, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")
    replacement = (
        f"{EMBED_START}\n"
        f"const EMBEDDED_BUNDLE = {payload};\n"
        f"{EMBED_END}"
    )
    new_text = text[:start] + replacement + text[end + len(EMBED_END):]
    index_path.write_text(new_text, encoding="utf-8")
    print(f"  embedded fallback updated: {index_path.name} ({len(payload)/1024:.1f} KB)")
    return True


# ---------------------------------------------------------------------------
# Bundle compilation
# ---------------------------------------------------------------------------

def _mastcam_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    live = sum(1 for r in reports if r.get("api_status") == "live")
    verified = sum(1 for r in reports if r.get("verified_url"))
    return {
        "url": IMAGES_API,
        "instrument": "MAST_LEFT",
        "sol_window": [SOL_MIN, SOL_MAX],
        "stations_requested": len(reports),
        "api_live": live,
        "urls_verified": verified,
        "offline_curated": sum(1 for r in reports if r.get("api_status") == "offline_curated"),
        "reports": reports,
        "credit": "NASA/JPL-Caltech/MSSS, MSL raw images",
    }


def validate_bundle(bundle: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    terrain = bundle.get("terrain", {})
    count = int(terrain.get("width", 0)) * int(terrain.get("height", 0))
    for key in (
        "elevation_dm", "slope_deg_x10", "tpi_dm", "thermal_inertia",
        "d2300_x1000", "sand_x100", "hazard", "rf_shadow", "ridge",
    ):
        if len(terrain.get(key, [])) != count:
            errors.append(f"terrain.{key} length mismatch")
    if not bundle.get("route", {}).get("path"):
        errors.append("route path is empty")
    if len(bundle.get("stations", [])) == 0:
        errors.append("no science stations")
    if bundle.get("route", {}).get("reserve", {}).get("worst_margin_wh") is None:
        errors.append("reserve margin missing")
    return errors


def compile_mission_bundle(
    *,
    offline: bool = False,
    verify_images: bool = True,
    raw_dir: Path = RAW_DIR,
    sync_index: bool = True,
    make_plot: bool = True,
) -> dict[str, Any]:
    print("ARES-WALK / NASA/JPL ingest")
    print(f"  sol window: {SOL_MIN}-{SOL_MAX}")
    print("  1/4  fetching MSL waypoints ...")
    track, waypoint_status = fetch_waypoints(offline=offline)
    print(
        f"       {waypoint_status.get('status')}  "
        f"{len(track)} final-drive points"
    )

    print("  2/4  fetching MAST_LEFT station frames ...")
    station_specs, image_reports = attach_images(
        DEFAULT_STATION_SPECS,
        offline=offline,
        verify=verify_images and not offline,
    )
    station_specs = attach_panoramas(station_specs)

    print("  3/4  optional local rasters ...")
    # The terrain grid is anchored to the same origin build_scene will use:
    # min track easting/northing minus the grid margin.  Computing it here
    # keeps raster sampling and the resulting bundle perfectly registered.
    _, track_e0, track_n0, _, _ = project_track(track)
    grid_origin_easting = track_e0 - MARGIN_M
    grid_origin_northing = track_n0 - MARGIN_M
    raster_layers, raster_report = load_local_rasters(
        raw_dir,
        origin_easting=grid_origin_easting,
        origin_northing=grid_origin_northing,
    )
    print(f"       raster status: {raster_report.get('status')}")

    print("  4/4  building twin + asymmetric Dijkstra ...")
    mastcam_summary = _mastcam_summary(image_reports)
    data_sources = {
        "waypoints": waypoint_status,
        "mastcam": mastcam_summary,
        "raster": raster_report,
        "terrain": {
            "mode": "local_geotiff_crop" if "elevation" in raster_layers else "calibrated_digital_twin",
            "note": (
                "Real Curiosity waypoints constrain the elevation trend. Boxwork relief, "
                "THEMIS-style TI, CRISM-style D2300, TPI and RF viewshed are modeled unless "
                "local rasters are supplied."
            ),
            "geotiff": raster_report,
        },
    }
    scene = build_scene(
        track,
        station_specs,
        data_sources=data_sources,
        raster_layers=raster_layers if raster_layers else None,
    )
    bundle = scene["bundle"]
    bundle["site"]["mission"] = "Mars Science Laboratory / Curiosity"
    bundle["data_sources"]["raster"] = raster_report
    bundle["data_sources"]["mastcam"] = mastcam_summary
    bundle["build"] = {
        "script": "src/pds_ingest.py",
        "utc": _now_utc(),
        "offline": bool(offline),
        "image_verification": bool(verify_images and not offline),
        "raster_layers": sorted(raster_layers.keys()),
        "planner": "src/mars_eva_engine.py",
    }
    errors = validate_bundle(bundle)
    if errors:
        raise RuntimeError("bundle validation failed: " + "; ".join(errors))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    BUNDLE_PATH.write_text(payload, encoding="utf-8")
    stats = bundle["route"]["stats"]
    print(f"       wrote {BUNDLE_PATH.relative_to(ROOT)}  ({len(payload)/1024:.1f} KB)")

    if make_plot:
        render_plan(bundle, PLOT_PATH)
        print(f"       wrote {PLOT_PATH.relative_to(ROOT)}")
    if sync_index:
        update_index_fallback(bundle)

    print("  summary")
    print(
        f"    route      {stats['distance_m']:.1f} m / "
        f"{stats['total_time_s']/60:.1f} min / {stats['total_energy_wh']:.1f} Wh"
    )
    print(
        f"    stations   {stats['feasible_station_count']}/{stats['station_count']} GO  "
        f"({bundle['stations'][0]['sol']}-{bundle['stations'][-1]['sol']})"
    )
    print(
        f"    reserve    {bundle['route']['reserve']['worst_margin_wh']:+.1f} Wh above "
        f"25% floor + walkback"
    )
    print(
        f"    comm       {bundle['terrain']['stats']['rf_shadow_fraction']*100:.1f}% shadow  |  "
        f"{len(bundle['terrain'].get('relays', []))} ridge repeaters"
    )
    print(
        f"    terrain    {bundle['terrain']['stats']['hazard_fraction']*100:.1f}% hazard  |  "
        f"{bundle['terrain']['stats']['ridge_fraction']*100:.1f}% TPI ridge"
    )
    if not bundle["route"]["reserve"]["feasible"]:
        print("    WARNING: strict 25% walkback reserve is not satisfied")
    print("  next: python -m http.server 8000  ->  http://localhost:8000")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the ARES-WALK mission bundle from NASA/JPL data."
    )
    parser.add_argument("--offline", action="store_true", help="use verified fallback data only")
    parser.add_argument("--skip-verify", action="store_true", help="skip HEAD checks of image URLs")
    parser.add_argument("--no-plot", action="store_true", help="do not regenerate the verification PNG")
    parser.add_argument("--no-index-sync", action="store_true", help="do not update the embedded twin")
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help="optional directory containing pre-cropped GeoTIFF/TIFF/IMG layers",
    )
    args = parser.parse_args(argv)
    try:
        compile_mission_bundle(
            offline=bool(args.offline),
            verify_images=not bool(args.skip_verify),
            raw_dir=args.raw_dir,
            sync_index=not bool(args.no_index_sync),
            make_plot=not bool(args.no_plot),
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
