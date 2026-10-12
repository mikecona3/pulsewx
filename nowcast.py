"""
Short-term motion extrapolation with an uncertainty cone that shrinks as
lead time decreases (i.e., the projection for "5 minutes from now" is much
tighter than "30 minutes from now") and tightens overall as more scans
confirm a stable motion vector.

This does NOT replace NWS tornado warning polygons, which incorporate
storm-scale NWP guidance, environmental shear/instability, and forecaster
judgment. This is pure trajectory extrapolation from observed radar motion —
useful for short lead times (0-15 min), unreliable beyond that.
"""

from dataclasses import dataclass

import numpy as np

from cell_track import TrackedCell


@dataclass
class PathProjection:
    lead_time_min: float
    lat: float
    lon: float
    radius_km: float  # uncertainty radius at this lead time


def project_path(
    track: TrackedCell,
    lead_times_min: list[float] = (5, 10, 15, 20, 30),
    base_uncertainty_km: float = 1.5,
    growth_km_per_min: float = 0.6,
    velocity: tuple[float, float] | None = None,
) -> list[PathProjection]:
    """
    Extrapolate a tracked cell's position forward using its most recent
    motion vector, with uncertainty radius growing linearly with lead time.

    velocity: (east_kmh, north_kmh) to use instead of the track's own
    estimate, e.g. the typical motion of nearby storms when this track's
    looks unreliable.

    Tightening vs. a generic cone comes from two things this uses that a
    climatological polygon doesn't:
      1. The cell's *actual observed* velocity (not a average tornado speed)
      2. `growth_km_per_min` can be tuned down as track history lengthens
         (a motion vector confirmed across 4+ scans is more trustworthy
         than one from 2 scans) — see `confidence_factor` below.
    """
    if velocity is None:
        velocity = track.velocity_kmh()
    if velocity is None:
        return []

    east_kmh, north_kmh = velocity
    origin = track.latest

    confidence = confidence_factor(track)

    projections = []
    for t_min in lead_times_min:
        t_h = t_min / 60.0
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * np.cos(np.radians(origin.lat))

        d_lat = (north_kmh * t_h) / km_per_deg_lat
        d_lon = (east_kmh * t_h) / km_per_deg_lon

        radius = (base_uncertainty_km + growth_km_per_min * t_min) * confidence

        projections.append(
            PathProjection(
                lead_time_min=t_min,
                lat=origin.lat + d_lat,
                lon=origin.lon + d_lon,
                radius_km=radius,
            )
        )
    return projections


def confidence_factor(track: TrackedCell, full_confidence_scans: int = 5) -> float:
    """
    Scale factor applied to the uncertainty radius based on how many
    consecutive scans have confirmed this track's motion. More history =
    tighter cone (down to 1.0x at full_confidence_scans+), fresh tracks
    (2 scans, the minimum to compute velocity) get a wider 1.8x cone.
    """
    n = len(track.history)
    if n >= full_confidence_scans:
        return 1.0
    # linear interpolation from 1.8x (n=2) down to 1.0x (n=full_confidence_scans)
    return 1.8 - (0.8 * (n - 2) / (full_confidence_scans - 2))
