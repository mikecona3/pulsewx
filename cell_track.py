"""
Storm cell identification (from reflectivity) and frame-to-frame tracking.

v1 approach: threshold reflectivity to find storm cores, label connected
regions, compute centroids, then match centroids between consecutive scans.
Matching predicts where each track should be now (from its own motion, or
the typical motion of all storms when it has none yet) and assigns cells to
tracks all at once to minimize total distance, with a max-jump distance to
avoid matching unrelated cells. This is a simplified version of the same
idea behind SCIT/TITAN storm-tracking algorithms.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

KM_PER_DEG_LAT = 111.0


@dataclass
class StormCell:
    id: int
    lat: float
    lon: float
    max_dbz: float
    area_km2: float
    timestamp: "object"  # datetime


@dataclass
class TrackedCell:
    id: int
    history: list[StormCell] = field(default_factory=list)

    @property
    def latest(self) -> StormCell:
        return self.history[-1]

    def velocity_kmh(self, max_points: int = 4) -> tuple[float, float] | None:
        """Return (east_kmh, north_kmh) motion estimate, or None if there's
        only one observation. Fits a straight line through the last
        max_points positions, so one jumpy centroid doesn't swing the
        estimate the way a two-point difference would."""
        points = self.history[-max_points:]
        if len(points) < 2:
            return None
        t0 = points[0].timestamp
        hours = np.array([(p.timestamp - t0).total_seconds() / 3600.0 for p in points])
        if hours[-1] <= 0:
            return None
        km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(points[-1].lat))
        east_km = np.array([p.lon for p in points]) * km_per_deg_lon
        north_km = np.array([p.lat for p in points]) * KM_PER_DEG_LAT
        return (float(np.polyfit(hours, east_km, 1)[0]), float(np.polyfit(hours, north_km, 1)[0]))


def gate_areas_km2(ranges_m: np.ndarray, n_rays: int) -> np.ndarray:
    """Approximate area of each gate in a 360° sweep with n_rays radials,
    shaped (1, n_gates) to broadcast against a sweep's field data. Gates
    get wider with range, so counting gates says little about size."""
    gate_depth_km = float(ranges_m[1] - ranges_m[0]) / 1000.0
    beam_width_rad = 2 * np.pi / n_rays
    return (ranges_m / 1000.0 * beam_width_rad * gate_depth_km)[None, :]


def identify_cells(
    reflectivity: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    timestamp,
    dbz_threshold: float = 40.0,
    min_pixels: int = 4,
    gate_area_km2: np.ndarray | None = None,
    min_area_km2: float = 10.0,
) -> list[StormCell]:
    """Threshold + connected-component label reflectivity to find storm cores.

    Pass gate_area_km2 (see gate_areas_km2) to get real areas and drop
    regions smaller than min_area_km2 — fragments that small have centroids
    that jump around between scans and wreck tracking. Without it, area_km2
    falls back to a raw gate count and only min_pixels applies.
    """
    refl = np.ma.filled(reflectivity, fill_value=-999)
    mask = refl >= dbz_threshold
    labeled, n = ndimage.label(mask)
    areas = None if gate_area_km2 is None else np.broadcast_to(gate_area_km2, refl.shape)

    cells = []
    for label_id in range(1, n + 1):
        region = labeled == label_id
        if region.sum() < min_pixels:
            continue
        area = float(region.sum()) if areas is None else float(areas[region].sum())
        if areas is not None and area < min_area_km2:
            continue
        cells.append(
            StormCell(
                id=label_id,
                lat=float(lats[region].mean()),
                lon=float(lons[region].mean()),
                max_dbz=float(refl[region].max()),
                area_km2=area,
                timestamp=timestamp,
            )
        )
    return cells


def update_tracks(
    tracks: list[TrackedCell],
    new_cells: list[StormCell],
    max_jump_km: float = 15.0,
) -> list[TrackedCell]:
    """Match new_cells to existing tracks; start new tracks for unmatched
    cells; drop tracks that go unmatched (storm ended)."""
    next_id = max((t.id for t in tracks), default=0) + 1
    pairs = []
    if tracks and new_cells:
        typical = _median_velocity(tracks)
        if typical is None:
            # No track has motion yet: match once assuming storms stood
            # still, take the typical shift from those matches, and match
            # again with it. Storms in one area mostly move together.
            typical = _median_shift(_match(tracks, new_cells, [(0.0, 0.0)] * len(tracks), max_jump_km),
                                    tracks, new_cells) or (0.0, 0.0)
        predicted = [t.velocity_kmh() or typical for t in tracks]
        pairs = _match(tracks, new_cells, predicted, max_jump_km)

    matched_tracks, used = [], set()
    for ti, ci in pairs:
        tracks[ti].history.append(new_cells[ci])
        matched_tracks.append(tracks[ti])
        used.add(ci)
    # Unmatched tracks are dropped (storm cell ended/merged)

    new_tracks = []
    for ci, cell in enumerate(new_cells):
        if ci not in used:
            new_tracks.append(TrackedCell(id=next_id, history=[cell]))
            next_id += 1

    return matched_tracks + new_tracks


def _match(tracks, cells, velocities, max_jump_km) -> list[tuple[int, int]]:
    """Globally optimal track→cell assignment by distance from each track's
    predicted position; pairs farther apart than max_jump_km are left out."""
    t_new = cells[0].timestamp
    pred_lat, pred_lon = [], []
    for track, (east, north) in zip(tracks, velocities):
        last = track.latest
        hours = (t_new - last.timestamp).total_seconds() / 3600.0
        pred_lat.append(last.lat + north * hours / KM_PER_DEG_LAT)
        pred_lon.append(last.lon + east * hours / (KM_PER_DEG_LAT * np.cos(np.radians(last.lat))))
    dist = _haversine_km(
        np.array(pred_lat)[:, None], np.array(pred_lon)[:, None],
        np.array([c.lat for c in cells])[None, :], np.array([c.lon for c in cells])[None, :],
    )
    cost = np.where(dist <= max_jump_km, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if dist[r, c] <= max_jump_km]


def _median_velocity(tracks) -> tuple[float, float] | None:
    velocities = [v for v in (t.velocity_kmh() for t in tracks) if v]
    if not velocities:
        return None
    east, north = np.median(np.array(velocities), axis=0)
    return (float(east), float(north))


def _median_shift(pairs, tracks, cells) -> tuple[float, float] | None:
    """Median (east_kmh, north_kmh) implied by a set of track→cell matches."""
    shifts = []
    for ti, ci in pairs:
        a, b = tracks[ti].latest, cells[ci]
        hours = (b.timestamp - a.timestamp).total_seconds() / 3600.0
        if hours <= 0:
            continue
        km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians((a.lat + b.lat) / 2))
        shifts.append(((b.lon - a.lon) * km_per_deg_lon / hours, (b.lat - a.lat) * KM_PER_DEG_LAT / hours))
    if not shifts:
        return None
    east, north = np.median(np.array(shifts), axis=0)
    return (float(east), float(north))


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
