#!/usr/bin/env python3
"""
ARES-WALK :: Mastcam stitched-mosaic builder
============================================

Fetches real MSL MAST_LEFT frame sequences from the NASA raw-image API, enriches
each frame with its PDS4 label (azimuth / elevation resolver angles, drive, site,
detector window and product metadata), rejects downward rover-deck frames and
thumbnail products, selects the largest horizontal sequence for the requested
station sols, crops the bottom JPL metadata strip, and projects the frames into a
wide equirectangular JPEG mosaic.

Only the two rock-rich landmark stations called out in the mission plan are
stitched by default:

    ALT-05  Layered Spur  (sol 4571, drive 2214)
    ALT-10  Upper Bench   (sol 4609, drive 2730)

Outputs:
    assets/panos/alt05_layered_spur_sol4571.jpg
    assets/panos/alt10_upper_bench_sol4609.jpg
    assets/panos/manifest.json          (frame IDs, angles, PDS IDs, SHA-256)

Usage:
    python src/build_mastcam_panos.py
    python src/build_mastcam_panos.py --offline          # use data/cache only
    python src/build_mastcam_panos.py --refresh          # ignore cached downloads
    python src/build_mastcam_panos.py --station ALT-05   # one station only

The PDS labels are authoritative for pointing.  The raw-image API exposes the
frame URLs but not usable mast_az / mast_el values, so every candidate frame is
enriched from its PDS4 label before filtering or stitching.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
ASSETS_PANO_DIR = ROOT / "assets" / "panos"
MANIFEST_PATH = ASSETS_PANO_DIR / "manifest.json"
CACHE_DIR = ROOT / "data" / "cache" / "mastcam"

# Local imports work for `python src/build_mastcam_panos.py`.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mars_eva_engine import DEFAULT_STATION_SPECS  # noqa: E402

IMAGES_API = "https://mars.nasa.gov/api/v1/raw_image_items"
RAW_IMAGE_BASE = "https://mars.nasa.gov/msl-raw-images/msss/{sol:05d}/mcam/{imageid}.jpg"
PDS_LABEL_BASE = (
    "https://pds-imaging.jpl.nasa.gov/data/msl/msl_mmm/data_MSLMST/"
    "volume_0040_raw/SURFACE/{sol}/{imageid_lower}.xml"
)
USER_AGENT = "ARES-WALK/1.0 (EVA planning demo; NASA open data; contact: local)"
HTTP_TIMEOUT = 60

# The MSL Mastcam left full-frame detector is 1648 x 1200 with a 34 x 26 degree
# field of view.  EDR sub-frames are windows into that detector, so their angular
# footprint scales with the recorded sample/line window.
FULL_FRAME_SIZE = (1648, 1200)
FULL_FRAME_FOV_DEG = (34.0, 26.0)

# JPL raw-image products carry a black metadata strip along the bottom edge.
# It is removed before projection so it cannot become a seam in the mosaic.
CROP_BOTTOM_PX = 24

# A portrait product is only treated as a "narrow portrait thumbnail" when it is
# also tiny.  The 672 x 896 Mastcam EDR sub-frames are portrait windows, not
# thumbnails, and are the actual multi-frame mosaic products; the 80 x 112 I01
# products are thumbnails and are excluded.
THUMBNAIL_MAX_WIDTH = 480

# Downward rover-deck / hardware frames: resolver elevation is measured from the
# mast's horizontal zero, so actual camera elevation is resolver - 90 degrees.
DOWNWARD_MAST_EL_DEG = -22.0
EL_CLUSTER_TOLERANCE_DEG = 3.0

# The stations called out in the mission plan.  Sol / drive / name come from the
# same curated station table used by the EVA engine and pds_ingest.py.
TARGET_STATIONS = ("ALT-05", "ALT-10")
STATION_FILENAMES = {
    "ALT-05": "alt05_layered_spur_sol4571.jpg",
    "ALT-10": "alt10_upper_bench_sol4609.jpg",
}
# Curiosity rover drive / site numbers come from the mission bundle waypoints.
STATION_DRIVE_SITE = {
    "ALT-05": {"drive": 2214, "site": 116},
    "ALT-10": {"drive": 2730, "site": 117},
}


# ---------------------------------------------------------------------------
# HTTP / cache helpers
# ---------------------------------------------------------------------------

def _request(url: str, *, timeout: int = HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,image/*,text/xml,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def _cached_fetch(url: str, cache_path: Path, *, refresh: bool = False) -> tuple[bytes, str]:
    """Return (payload, source) where source is 'cache' or 'network'."""
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes(), "cache"
    payload = _request(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return payload, "network"


def _fetch_json(url: str, *, retries: int = 2) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return json.loads(_request(url).decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# NASA raw-image API
# ---------------------------------------------------------------------------

def fetch_mastcam_items(sol: int, *, cache_dir: Path, refresh: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch every MAST_LEFT item for a sol, following API pagination."""
    items: list[dict[str, Any]] = []
    page = 0
    total: int | None = None
    sources: list[str] = []
    while page < 12:
        query = urllib.parse.urlencode({
            "order": "sol asc",
            "per_page": 100,
            "page": page,
            "condition_1": "msl:mission",
            "condition_2": f"{sol}:sol",
            "condition_3": "MAST_LEFT:instrument",
        })
        url = f"{IMAGES_API}?{query}"
        cache_path = cache_dir / "api" / f"sol_{sol}_page_{page}.json"
        try:
            payload_raw, source = _cached_fetch(url, cache_path, refresh=refresh)
            sources.append(source)
        except Exception as exc:
            if page == 0:
                raise RuntimeError(f"MASTcam API fetch failed for sol {sol}: {exc}") from exc
            break
        payload = json.loads(payload_raw.decode("utf-8"))
        batch = payload.get("items", []) or []
        items.extend(batch)
        total = payload.get("total")
        if not batch or (total is not None and len(items) >= int(total)):
            break
        page += 1
    # De-duplicate by imageid, preserving API order.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in items:
        imageid = str(item.get("imageid", ""))
        if imageid and imageid not in seen:
            seen.add(imageid)
            unique.append(item)
    return unique, {
        "sol": sol,
        "api_items": len(unique),
        "api_total": total,
        "pages": page + 1,
        "cache_sources": sorted(set(sources)),
    }


# ---------------------------------------------------------------------------
# PDS4 label parsing
# ---------------------------------------------------------------------------

def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


def _indexed_value(root: ET.Element, index_id: str, value_tag: str) -> str | None:
    """Read <..._Index><index_id>ID</index_id><..._value...>value</...>."""
    for parent in root.iter():
        if _local_name(parent.tag) != "Device_Angle_Index" and _local_name(parent.tag) != "Coordinate_Space_Index":
            continue
        ident = _child_text(parent, "index_id")
        if ident != index_id:
            continue
        for child in parent:
            if _local_name(child.tag) == value_tag:
                text = (child.text or "").strip()
                if text:
                    return text
    return None


def parse_pds_label(xml_text: str, *, imageid: str) -> dict[str, Any]:
    """Extract pointing, drive/site, detector window and PDS product metadata."""
    root = ET.fromstring(xml_text)

    az_resolver = _indexed_value(root, "AZIMUTH FINAL-RESOLVER", "index_value_angle")
    el_resolver = _indexed_value(root, "ELEVATION FINAL-RESOLVER", "index_value_angle")
    if az_resolver is None or el_resolver is None:
        raise ValueError("PDS label is missing final-resolver azimuth/elevation")
    resolver_az_deg = round(math.degrees(float(az_resolver)), 6)
    resolver_el_deg = round(math.degrees(float(el_resolver)), 6)
    # Resolver elevation is measured from the mast horizontal; camera elevation
    # is resolver - 90 deg.  A frame pointed at the horizon therefore has
    # mast_el ~= -6 deg for these near-level Mastcam sequences.
    mast_el_deg = round(resolver_el_deg - 90.0, 6)

    drive_raw = _indexed_value(root, "DRIVE", "index_value_number")
    site_raw = _indexed_value(root, "SITE", "index_value_number")

    detector = None
    for element in root.iter():
        if _local_name(element.tag) == "Detector":
            detector = element
            break
    width = height = None
    first_line = first_sample = None
    if detector is not None:
        samples = _child_text(detector, "samples")
        lines = _child_text(detector, "lines")
        if samples and lines:
            width, height = int(samples), int(lines)
        first_line = _child_text(detector, "first_line")
        first_sample = _child_text(detector, "first_sample")

    product_type = None
    camera_product_id = None
    spacecraft_clock_start = None
    start_sol = None
    release_number = None
    for element in root.iter():
        name = _local_name(element.tag)
        if name == "product_type_name" and product_type is None:
            product_type = (element.text or "").strip() or None
        elif name == "camera_product_id" and camera_product_id is None:
            camera_product_id = (element.text or "").strip() or None
        elif name == "spacecraft_clock_start" and spacecraft_clock_start is None:
            spacecraft_clock_start = (element.text or "").strip() or None
        elif name == "start_sol_number" and start_sol is None:
            start_sol = (element.text or "").strip() or None
        elif name == "release_number" and release_number is None:
            release_number = (element.text or "").strip() or None

    return {
        "pds_product_id": imageid,
        "pds_product_type": product_type,
        "pds_camera_product_id": camera_product_id,
        "pds_spacecraft_clock_start": spacecraft_clock_start,
        "pds_start_sol": start_sol,
        "pds_release_number": release_number,
        "resolver_az_deg": resolver_az_deg,
        "resolver_el_deg": resolver_el_deg,
        "mast_az_deg": round(resolver_az_deg % 360.0, 6),
        "mast_el_deg": mast_el_deg,
        "drive": int(drive_raw) if drive_raw is not None else None,
        "site": int(site_raw) if site_raw is not None else None,
        "detector_width": width,
        "detector_height": height,
        "detector_first_line": int(first_line) if first_line is not None else None,
        "detector_first_sample": int(first_sample) if first_sample is not None else None,
    }


# ---------------------------------------------------------------------------
# Frame discovery, enrichment and filtering
# ---------------------------------------------------------------------------

def _sequence_id(imageid: str) -> str | None:
    """4571ML1077860001802043C00_DXXX -> 4571ML107786."""
    match = re.match(r"^(.{12})(\d{3})", imageid)
    return match.group(1) if match else None


def _is_thumbnail_imageid(imageid: str) -> bool:
    return "I01_DXXX" in imageid.upper()


def _pds_urn(imageid: str) -> str:
    """Canonical PDS4 product LIDVID-style identifier for an MSL EDR frame."""
    return f"urn:nasa:pds:msl_mmm:data_mslmst:{imageid.lower()}"


def enrich_frame(
    item: dict[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
    offline: bool,
) -> dict[str, Any]:
    """Download/cache the PDS label and return a pointing-enriched frame record."""
    imageid = str(item.get("imageid", ""))
    sol = int(item.get("sol", 0)) if item.get("sol") else int(imageid[:4])
    label_url = PDS_LABEL_BASE.format(sol=sol, imageid_lower=imageid.lower())
    label_cache = cache_dir / "labels" / f"{sol}" / f"{imageid.lower()}.xml"
    record: dict[str, Any] = {
        "imageid": imageid,
        "url": str(item.get("url") or RAW_IMAGE_BASE.format(sol=sol, imageid=imageid)),
        "pds_label_url": label_url,
        "sol": sol,
        "sequence_id": _sequence_id(imageid),
        "title": item.get("title"),
        "date_taken": item.get("date_taken"),
        "pds_label_sha256": None,
        "source_jpeg_sha256": None,
        "source_width": None,
        "source_height": None,
        "crop_bottom_px": CROP_BOTTOM_PX,
        "crop_bounds": None,
        "filter_reasons": [],
        "errors": [],
    }
    try:
        if offline and not label_cache.exists():
            raise RuntimeError("offline mode and label is not cached")
        label_raw, source = _cached_fetch(label_url, label_cache, refresh=refresh)
        record["pds_label_source"] = source
        record["pds_label_sha256"] = _sha256(label_raw)
        label = parse_pds_label(label_raw.decode("utf-8", errors="replace"), imageid=imageid)
        record.update(label)
    except Exception as exc:
        record["errors"].append(f"label: {str(exc)[:220]}")
        return record

    width = record.get("detector_width")
    height = record.get("detector_height")
    if not width or not height:
        record["errors"].append("label has no detector sample/line dimensions")
        return record
    record["source_width"] = int(width)
    record["source_height"] = int(height)
    record["crop_bounds"] = [0, 0, int(width), max(0, int(height) - CROP_BOTTOM_PX)]

    # Filter 1: downward rover-deck / hardware pointings.
    if record.get("mast_el_deg") is not None and record["mast_el_deg"] < DOWNWARD_MAST_EL_DEG:
        record["filter_reasons"].append(
            f"downward mast_el {record['mast_el_deg']:.2f} deg < {DOWNWARD_MAST_EL_DEG:.1f} deg"
        )
    # Filter 2: narrow portrait thumbnails.  A 672 x 896 EDR window is portrait
    # but is a real science frame, not a thumbnail; the 80 x 112 I01 products are
    # the thumbnails this filter is meant to remove.
    if width < height and width < THUMBNAIL_MAX_WIDTH:
        record["filter_reasons"].append(
            f"narrow portrait thumbnail {width}x{height} (width < {THUMBNAIL_MAX_WIDTH})"
        )
    if _is_thumbnail_imageid(imageid):
        record["filter_reasons"].append("I01 thumbnail product")
    if record.get("drive") is None:
        record["filter_reasons"].append("missing PDS drive index")
    return record


def _cluster_by_elevation(frames: list[dict[str, Any]], tolerance: float) -> list[list[dict[str, Any]]]:
    """Single-linkage clusters: consecutive sorted elevations within tolerance."""
    ordered = sorted(frames, key=lambda f: float(f["mast_el_deg"]))
    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in ordered:
        if not current or abs(float(frame["mast_el_deg"]) - float(current[-1]["mast_el_deg"])) <= tolerance:
            current.append(frame)
        else:
            clusters.append(current)
            current = [frame]
    if current:
        clusters.append(current)
    return clusters


def select_sequence(
    frames: list[dict[str, Any]],
    *,
    station_drive: int | None,
    el_tolerance: float = EL_CLUSTER_TOLERANCE_DEG,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the largest drive-matched horizontal sequence for a station."""
    usable = [f for f in frames if not f.get("filter_reasons") and not f.get("errors")]
    dropped = [f for f in frames if f.get("filter_reasons") or f.get("errors")]

    by_drive: dict[int, list[dict[str, Any]]] = {}
    for frame in usable:
        drive = frame.get("drive")
        if drive is not None:
            by_drive.setdefault(int(drive), []).append(frame)

    candidates: list[dict[str, Any]] = []
    for drive, group in by_drive.items():
        for cluster in _cluster_by_elevation(group, el_tolerance):
            if len(cluster) < 3:
                continue
            azs = [float(f["mast_az_deg"]) for f in cluster]
            az_span = max(azs) - min(azs)
            if az_span < 15.0:
                continue
            els = [float(f["mast_el_deg"]) for f in cluster]
            candidates.append({
                "drive": int(drive),
                "frames": sorted(cluster, key=lambda f: float(f["mast_az_deg"])),
                "frame_count": len(cluster),
                "az_span_deg": round(az_span, 3),
                "az_min_deg": round(min(azs), 3),
                "az_max_deg": round(max(azs), 3),
                "el_center_deg": round(float(np.mean(els)), 3),
                "el_min_deg": round(min(els), 3),
                "el_max_deg": round(max(els), 3),
                "drive_match": station_drive is not None and int(drive) == int(station_drive),
            })

    if not candidates:
        raise RuntimeError("no usable horizontal Mastcam sequence after filtering")

    # Prefer the station's own drive, then the most frames, then the widest azimuth sweep.
    candidates.sort(key=lambda c: (not c["drive_match"], -c["frame_count"], -c["az_span_deg"]))
    chosen = candidates[0]
    report = {
        "usable_frames": len(usable),
        "dropped_frames": len(dropped),
        "dropped": [
            {
                "imageid": f["imageid"],
                "reasons": f.get("filter_reasons") or [],
                "errors": f.get("errors") or [],
            }
            for f in dropped
        ],
        "drive_candidates": [
            {
                "drive": c["drive"],
                "frames": c["frame_count"],
                "az_span_deg": c["az_span_deg"],
                "el_center_deg": c["el_center_deg"],
                "drive_match": c["drive_match"],
            }
            for c in candidates
        ],
        "selected_drive": chosen["drive"],
        "selected_frames": chosen["frame_count"],
        "selected_az_span_deg": chosen["az_span_deg"],
        "selected_el_center_deg": chosen["el_center_deg"],
        "el_cluster_tolerance_deg": el_tolerance,
    }
    return chosen["frames"], report


# ---------------------------------------------------------------------------
# Image download / crop
# ---------------------------------------------------------------------------

def load_cropped_frame(
    frame: dict[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
    offline: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Download the JPEG, crop the bottom metadata strip, return RGB array + hash info."""
    imageid = frame["imageid"]
    sol = int(frame["sol"])
    url = str(frame["url"] or RAW_IMAGE_BASE.format(sol=sol, imageid=imageid))
    cache_path = cache_dir / "images" / f"{sol}" / f"{imageid}.jpg"
    if offline and not cache_path.exists():
        raise RuntimeError(f"offline mode and image {imageid} is not cached")
    raw, source = _cached_fetch(url, cache_path, refresh=refresh)
    sha = _sha256(raw)
    with Image.open(io.BytesIO(raw)) as im:
        image = np.asarray(im.convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    crop_bottom = min(CROP_BOTTOM_PX, max(0, height - 8))
    cropped = image[: height - crop_bottom, :, :].copy()
    return cropped, {
        "url": url,
        "cache_source": source,
        "sha256_jpeg": sha,
        "source_width": int(width),
        "source_height": int(height),
        "crop_bottom_px": int(crop_bottom),
        "crop_bounds": [0, 0, int(width), int(height - crop_bottom)],
    }


# ---------------------------------------------------------------------------
# Equirectangular az/el projection
# ---------------------------------------------------------------------------

def _frame_fov(frame: dict[str, Any]) -> tuple[float, float]:
    width = int(frame["source_width"])
    height = int(frame["source_height"])
    hfov = FULL_FRAME_FOV_DEG[0] * (width / FULL_FRAME_SIZE[0])
    vfov = FULL_FRAME_FOV_DEG[1] * (height / FULL_FRAME_SIZE[1])
    return hfov, vfov


def _bilinear(channel: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Bilinear sample a 2-D array at floating pixel coordinates."""
    height, width = channel.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y1, 0, height - 1)
    fx = np.clip(xs - x0, 0.0, 1.0)
    fy = np.clip(ys - y0, 0.0, 1.0)
    v00 = channel[y0c, x0c]
    v10 = channel[y0c, x1c]
    v01 = channel[y1c, x0c]
    v11 = channel[y1c, x1c]
    top = v00 * (1 - fx) + v10 * fx
    bottom = v01 * (1 - fx) + v11 * fx
    return top * (1 - fy) + bottom * fy


def stitch_frames(frames: list[dict[str, Any]], cropped: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    """Project Mastcam az/el frames onto a shared equirectangular canvas."""
    if not frames:
        raise ValueError("no frames to stitch")

    geometry = []
    for frame in frames:
        hfov, vfov = _frame_fov(frame)
        az0 = float(frame["mast_az_deg"])
        el0 = float(frame["mast_el_deg"])
        crop_bottom = int(frame["crop_bottom_px"])
        source_h = int(frame["source_height"])
        cropped_h = source_h - crop_bottom
        # Cropping the bottom strip removes the lower part of the detector FOV.
        vfov_cropped = vfov * (cropped_h / source_h)
        el_top = el0 + vfov / 2.0
        el_bottom = el_top - vfov_cropped
        geometry.append({
            "frame": frame,
            "image": cropped[frame["imageid"]],
            "hfov": hfov,
            "vfov": vfov,
            "vfov_cropped": vfov_cropped,
            "az0": az0,
            "el0": el0,
            "el_top": el_top,
            "el_bottom": el_bottom,
            "az_left": az0 - hfov / 2.0,
            "az_right": az0 + hfov / 2.0,
        })

    az_min = min(g["az_left"] for g in geometry)
    az_max = max(g["az_right"] for g in geometry)
    el_min = min(g["el_bottom"] for g in geometry)
    el_max = max(g["el_top"] for g in geometry)
    az_span = az_max - az_min
    if az_span <= 0:
        raise ValueError("selected frames have no angular coverage")

    # Frame elevations often drift slightly across a mast sweep.  Using the
    # union rectangle would leave black stair-steps where only some frames
    # cover a pixel, so crop to the vertical band shared by every frame.  The
    # resulting mosaic is still the full azimuth union, but has no voids and
    # no fabricated pixels.
    el_top_common = min(g["el_top"] for g in geometry)
    el_bottom_common = max(g["el_bottom"] for g in geometry)
    common_band = el_top_common > el_bottom_common
    if common_band:
        el_top, el_bottom = el_top_common, el_bottom_common
    else:
        # Pathological scan with no shared elevation: keep the union and fill
        # residual voids from the nearest covered pixel below.
        el_top, el_bottom = el_max, el_min
    el_span = el_top - el_bottom
    if el_span <= 0:
        raise ValueError("selected frames have no angular coverage")

    # Match the source pixel scale (slightly oversampled so the panorama stays
    # crisp on wide displays); cap the canvas so the JPEG stays practical.
    src_deg_per_px = min(
        g["hfov"] / int(g["frame"]["source_width"]) for g in geometry
    )
    deg_per_px = src_deg_per_px / 1.18
    out_w = int(round(az_span / deg_per_px))
    out_h = int(round(el_span / deg_per_px))
    out_w = max(200, min(out_w, 4800))
    out_h = max(120, min(out_h, 2200))

    az = az_min + (np.arange(out_w, dtype=np.float64) + 0.5) * (az_span / out_w)
    el = el_top - (np.arange(out_h, dtype=np.float64) + 0.5) * (el_span / out_h)
    az_grid, el_grid = np.meshgrid(az, el)

    accumulator = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)
    contribution = {}

    for g in geometry:
        frame = g["frame"]
        image = g["image"].astype(np.float64)
        src_h, src_w = image.shape[:2]
        hfov = g["hfov"]
        vfov = g["vfov"]
        az0 = g["az0"]
        el0 = g["el0"]
        el_top = g["el_top"]

        src_w = int(frame["source_width"])
        xs = (az_grid - g["az_left"]) / hfov * src_w - 0.5
        ys = (el_top - el_grid) / vfov * int(frame["source_height"]) - 0.5
        # Pixel centres sit half a pixel inside the geometric frame bounds;
        # allow that half-pixel tolerance and clamp before bilinear sampling.
        valid = (xs >= -0.5) & (xs <= src_w - 0.5) & (ys >= -0.5) & (ys <= src_h - 0.5)
        if not valid.any():
            contribution[frame["imageid"]] = 0.0
            continue
        xs = np.clip(xs, 0.0, src_w - 1.0)
        ys = np.clip(ys, 0.0, src_h - 1.0)

        # Hann-like feather across the frame; a small floor keeps the outer
        # boundary from going transparent where only one frame covers a pixel.
        wx = 0.5 + 0.5 * np.cos(np.pi * np.clip(np.abs(az_grid - az0) / (hfov / 2.0), 0.0, 1.0))
        wy = 0.5 + 0.5 * np.cos(np.pi * np.clip(np.abs(el_grid - (el_top - g["vfov_cropped"] / 2.0)) / (g["vfov_cropped"] / 2.0), 0.0, 1.0))
        weight = np.where(valid, np.maximum(wx * wy, 0.04), 0.0)

        sampled = np.empty((out_h, out_w, 3), dtype=np.float64)
        for channel in range(3):
            sampled[..., channel] = _bilinear(image[..., channel], ys, xs)
        accumulator += weight[..., None] * sampled
        weight_sum += weight
        contribution[frame["imageid"]] = round(float(valid.mean()), 4)

    filled_pixels = 0
    if np.any(weight_sum <= 0):
        # Neighbour fill keeps an unusual scan from producing black holes.
        # The common vertical band above normally makes this a no-op.
        filled_pixels = int((weight_sum <= 0).sum())
        if np.any(weight_sum > 0):
            from scipy import ndimage

            _, nearest = ndimage.distance_transform_edt(weight_sum <= 0, return_indices=True)
            nearest = tuple(nearest)
            for channel in range(3):
                accumulator[..., channel] = np.where(
                    weight_sum <= 0,
                    accumulator[..., channel][nearest],
                    accumulator[..., channel],
                )
            weight_sum = np.where(weight_sum <= 0, 1.0, weight_sum)
        else:
            raise ValueError("projection left no covered output pixels")
    mosaic = accumulator / weight_sum[..., None]
    mosaic = np.clip(np.rint(mosaic), 0, 255).astype(np.uint8)

    metadata = {
        "projection": "equirectangular azimuth/elevation",
        "width": int(out_w),
        "height": int(out_h),
        "az_min_deg": round(float(az_min), 4),
        "az_max_deg": round(float(az_max), 4),
        "el_min_deg": round(float(el_bottom), 4),
        "el_max_deg": round(float(el_top), 4),
        "az_span_deg": round(float(az_span), 4),
        "el_span_deg": round(float(el_span), 4),
        "common_vertical_band": bool(common_band),
        "cropped_el_union_deg": [
            round(float(el_min), 4),
            round(float(el_max), 4),
        ],
        "nearest_filled_pixels": filled_pixels,
        "output_deg_per_px": round(float(deg_per_px), 6),
        "source_deg_per_px": round(float(src_deg_per_px), 6),
        "frame_contribution": contribution,
        "feather": "Hann-like az/el weight with 0.04 floor, normalised across overlapping frames",
    }
    return mosaic, metadata


# ---------------------------------------------------------------------------
# Per-station pipeline
# ---------------------------------------------------------------------------

def build_station_pano(
    station: dict[str, Any],
    *,
    cache_dir: Path,
    refresh: bool,
    offline: bool,
    quality: int = 92,
) -> dict[str, Any]:
    station_id = str(station["id"])
    sol = int(station["sol"])
    meta = STATION_DRIVE_SITE.get(station_id, {})
    drive = int(meta.get("drive", 0))
    filename = STATION_FILENAMES.get(station_id)
    if filename is None:
        raise ValueError(f"no output filename configured for {station_id}")

    print(f"  {station_id} sol {sol} drive {drive}: fetching MAST_LEFT items ...")
    items, api_report = fetch_mastcam_items(sol, cache_dir=cache_dir, refresh=refresh)
    c00_items = [
        item for item in items
        if "C00_DXXX" in str(item.get("imageid", "")).upper()
        and not _is_thumbnail_imageid(str(item.get("imageid", "")))
    ]
    print(f"    {len(items)} API items, {len(c00_items)} C00 frames; enriching PDS labels ...")

    frames: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(
                enrich_frame, item, cache_dir=cache_dir, refresh=refresh, offline=offline
            ): item
            for item in c00_items
        }
        for future in as_completed(futures):
            frames.append(future.result())
    frames.sort(key=lambda f: str(f.get("imageid", "")))

    selected, selection_report = select_sequence(frames, station_drive=drive)
    print(
        f"    selected drive {selection_report['selected_drive']}: "
        f"{selection_report['selected_frames']} frames, "
        f"az {selection_report['selected_az_span_deg']:.1f} deg, "
        f"el {selection_report['selected_el_center_deg']:.1f} deg"
    )

    cropped: dict[str, np.ndarray] = {}
    frame_meta: list[dict[str, Any]] = []
    for frame in selected:
        image, image_meta = load_cropped_frame(
            frame, cache_dir=cache_dir, refresh=refresh, offline=offline
        )
        cropped[frame["imageid"]] = image
        src_w = int(image_meta["source_width"])
        src_h = int(image_meta["source_height"])
        crop_bottom = int(image_meta["crop_bottom_px"])
        frame_meta.append({
            "imageid": frame["imageid"],
            "url": image_meta["url"],
            "pds_label_url": frame["pds_label_url"],
            "pds_product_id": _pds_urn(frame["imageid"]),
            "pds_product_type": frame.get("pds_product_type"),
            "pds_camera_product_id": frame.get("pds_camera_product_id"),
            "pds_spacecraft_clock_start": frame.get("pds_spacecraft_clock_start"),
            "pds_release_number": frame.get("pds_release_number"),
            "pds_label_sha256": frame.get("pds_label_sha256"),
            "sha256": image_meta["sha256_jpeg"],
            "sha256_jpeg": image_meta["sha256_jpeg"],
            "mast_az": frame.get("mast_az_deg"),
            "mast_el": frame.get("mast_el_deg"),
            "mast_az_deg": frame.get("mast_az_deg"),
            "mast_el_deg": frame.get("mast_el_deg"),
            "resolver_az_deg": frame.get("resolver_az_deg"),
            "resolver_el": frame.get("resolver_el_deg"),
            "resolver_el_deg": frame.get("resolver_el_deg"),
            "drive": str(frame.get("drive")) if frame.get("drive") is not None else None,
            "site": frame.get("site"),
            "dimensions": [src_w, src_h],
            "source_width": src_w,
            "source_height": src_h,
            "crop_bottom_px": crop_bottom,
            "crop_bounds": {
                "left": 0,
                "top": 0,
                "right": src_w,
                "bottom": max(0, src_h - crop_bottom),
            },
            "sequence_id": frame.get("sequence_id"),
        })

    mosaic, stitch_meta = stitch_frames(selected, cropped)
    ASSETS_PANO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ASSETS_PANO_DIR / filename
    Image.fromarray(mosaic).save(out_path, "JPEG", quality=quality, optimize=True)
    print(f"    wrote {out_path.relative_to(ROOT)}  ({out_path.stat().st_size/1024:.0f} KB)")

    width = int(stitch_meta["width"])
    height = int(stitch_meta["height"])
    entry = {
        "id": f"{station_id}_{station['name'].lower().replace(' ', '_')}_sol{sol}",
        "station_id": station_id,
        "station_name": station["name"],
        "sol": sol,
        "drive": str(drive),
        "site": int(selected[0].get("site") or 0),
        "filename": filename,
        "file_name": filename,
        "pano_url": f"assets/panos/{filename}",
        "file": f"assets/panos/{filename}",
        "frame_count": len(frame_meta),
        "az_range": [stitch_meta["az_min_deg"], stitch_meta["az_max_deg"]],
        "el_range": [stitch_meta["el_min_deg"], stitch_meta["el_max_deg"]],
        "dimensions": [width, height],
        "width": width,
        "height": height,
        "sha256": _sha256(out_path.read_bytes()),
        "az_min_deg": stitch_meta["az_min_deg"],
        "az_max_deg": stitch_meta["az_max_deg"],
        "el_min_deg": stitch_meta["el_min_deg"],
        "el_max_deg": stitch_meta["el_max_deg"],
        "az_span_deg": stitch_meta["az_span_deg"],
        "el_span_deg": stitch_meta["el_span_deg"],
        "projection": stitch_meta["projection"],
        "crop_bottom_px": CROP_BOTTOM_PX,
        "quality": quality,
        "frames": frame_meta,
        "selection": selection_report,
        "api": api_report,
        "stitch": stitch_meta,
        "credit": "NASA/JPL-Caltech/MSSS, MSL Mastcam raw images",
    }
    return entry


def main(argv: list[str] | None = None) -> int:
    global ASSETS_PANO_DIR
    parser = argparse.ArgumentParser(description="Build stitched Mastcam landmark panoramas.")
    parser.add_argument("--station", action="append", help="station id (default: ALT-05 ALT-10)")
    parser.add_argument("--offline", action="store_true", help="use cached API/PDS/JPEG data only")
    parser.add_argument("--refresh", action="store_true", help="ignore cached downloads")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=ASSETS_PANO_DIR)
    parser.add_argument("--quality", type=int, default=92)
    args = parser.parse_args(argv)

    station_ids = args.station or list(TARGET_STATIONS)
    stations = []
    for station_id in station_ids:
        match = next((s for s in DEFAULT_STATION_SPECS if s["id"] == station_id), None)
        if match is None:
            print(f"ERROR: unknown station {station_id}", file=sys.stderr)
            return 2
        stations.append(match)

    ASSETS_PANO_DIR = args.out_dir
    ASSETS_PANO_DIR.mkdir(parents=True, exist_ok=True)

    panos: list[dict[str, Any]] = []
    try:
        for station in stations:
            panos.append(build_station_pano(
                station,
                cache_dir=args.cache_dir,
                refresh=args.refresh,
                offline=args.offline,
                quality=args.quality,
            ))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Keep the consumer-facing ``landmarks`` mapping used by pds_ingest.py and a
    # versioned schema marker so the manifest is self-describing.
    manifest = {
        "schema": "ares-walk/mastcam-mosaic-manifest@1.0",
        "generator": "src/build_mastcam_panos.py",
        "description": "Multi-frame Mastcam landmark stitched panoramas with honest PDS metadata.",
        "generated_utc": _now_utc(),
        "source_api": IMAGES_API,
        "pds_volume": "volume_0040_raw",
        "pds_label_base": PDS_LABEL_BASE,
        "crop_bottom_px": CROP_BOTTOM_PX,
        "downward_mast_el_threshold_deg": DOWNWARD_MAST_EL_DEG,
        "thumbnail_max_width": THUMBNAIL_MAX_WIDTH,
        "el_cluster_tolerance_deg": EL_CLUSTER_TOLERANCE_DEG,
        "full_frame_fov_deg": list(FULL_FRAME_FOV_DEG),
        "credit": "NASA/JPL-Caltech/MSSS, MSL Mastcam raw images",
        "landmarks": {entry["station_id"]: entry for entry in panos},
        "panos": panos,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  wrote {manifest_path.relative_to(ROOT)}")
    print(f"  done: {len(panos)} stitched mosaic(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
