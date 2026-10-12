"""
Basic rotation (TVS/mesocyclone-style) detection from radar velocity data.

Approach:
  1. Dealias the radial velocity field. The radar can only measure speeds up
     to its Nyquist velocity; anything faster "folds" to the opposite sign,
     and an uncorrected fold looks exactly like a strong in/out couplet.
  2. Look for azimuthal shear: at the same range, a strong inbound gate
     (negative) with a strong outbound gate (positive) on a neighboring
     radial a little clockwise of it. That side-by-side pair across beams is
     the classic "couplet" signature of cyclonic rotation. Comparing gates
     along a single beam instead would find convergence/divergence, not
     rotation.
  3. Keep only couplets that show up on at least two elevation angles —
     persistence with height is a much stronger indicator than one tilt.

This is a simplified version of what NWS algorithms (like MDA/TDA) do; it's a
reasonable v1 that can be replaced or supplemented with an ML classifier
later. Only cyclonic (counterclockwise) rotation is detected, which covers
the vast majority of Northern Hemisphere tornadoes.

Caveat: velocity couplets are a NECESSARY but not SUFFICIENT condition for
tornadic rotation — plenty of non-tornadic shear produces similar
signatures. Treat detections as "candidate," not "confirmed."
"""

from dataclasses import dataclass, replace

import numpy as np
import pyart

DEALIASED_FIELD = "corrected_velocity"


@dataclass
class RotationCandidate:
    azimuth_deg: float
    range_km: float
    elevation_deg: float
    shear_ms: float  # velocity difference across the couplet, m/s
    lat: float
    lon: float
    tilts: int = 1  # how many distinct elevation angles the couplet appears on


def dealias_velocity(radar: "pyart.core.Radar") -> str:
    """Add a dealiased copy of the velocity field to the radar (once) and
    return its field name."""
    if DEALIASED_FIELD not in radar.fields:
        corrected = pyart.correct.dealias_region_based(radar, vel_field="velocity")
        radar.add_field(DEALIASED_FIELD, corrected)
    return DEALIASED_FIELD


def find_velocity_couplets(
    radar: "pyart.core.Radar",
    sweep: int = 0,
    shear_threshold_ms: float = 25.0,
    max_az_separation: int = 2,
    min_dbz: float = 20.0,
    max_range_km: float = 150.0,
    vel_field: str = "velocity",
) -> list[RotationCandidate]:
    """
    Scan one sweep's velocity field for cyclonic rotation couplets.

    shear_threshold_ms: minimum outbound-minus-inbound velocity across the
        couplet to flag. 25 m/s (~56 mph combined shear) is a starting
        point — tune against known cases for your radar/region.
    max_az_separation: how many radials apart the in/out pair can be and
        still count as one couplet (accounts for beam smearing).
    min_dbz: ignore couplets where neither gate has at least this much
        reflectivity — rotation lives inside storms, not in clear air.
    max_range_km: beyond ~150 km the beam is too high and too wide to
        resolve low-level rotation.
    vel_field: pass the dealiased field (see dealias_velocity); raw
        "velocity" will flag every aliasing fold.
    """
    sweep_slice = radar.get_slice(sweep)
    vel = radar.fields[vel_field]["data"][sweep_slice]
    if np.ma.getmaskarray(vel).all():
        return []  # surveillance-only sweep, no velocity data

    # Sort radials by azimuth so neighboring rows are neighboring beams; the
    # np.roll below then also wraps 359.5° around to 0°.
    azimuths = radar.get_azimuth(sweep)
    order = np.argsort(azimuths)
    azimuths = azimuths[order]
    v = np.ma.filled(vel[order].astype(float), np.nan)
    z = np.ma.filled(radar.fields["reflectivity"]["data"][sweep_slice][order].astype(float), np.nan)
    lats, lons, _ = radar.get_gate_lat_lon_alt(sweep)
    lats, lons = lats[order], lons[order]
    ranges_km = radar.range["data"] / 1000.0
    in_range = (ranges_km <= max_range_km)[None, :]

    best_shear = np.zeros_like(v)
    best_k = np.zeros(v.shape, dtype=int)
    for k in range(1, max_az_separation + 1):
        # The radial k steps clockwise. Cyclonic rotation puts outbound flow
        # clockwise of inbound flow, whichever side of the radar it's on.
        v_cw = np.roll(v, -k, axis=0)
        z_cw = np.roll(z, -k, axis=0)
        az_gap = (np.roll(azimuths, -k) - azimuths) % 360.0
        adjacent = (az_gap <= 1.5 * k)[:, None]  # skip gaps from missing radials
        shear = v_cw - v
        # NaN (masked) gates compare False, so they never qualify
        with np.errstate(invalid="ignore"):
            ok = (
                (v < 0) & (v_cw > 0) & (shear >= shear_threshold_ms)
                & (np.fmax(z, z_cw) >= min_dbz) & in_range & adjacent
            )
        better = ok & (shear > best_shear)
        best_shear[better] = shear[better]
        best_k[better] = k

    n_rays = v.shape[0]
    elevation = float(radar.fixed_angle["data"][sweep])
    candidates = []
    for ray, gate in zip(*np.nonzero(best_shear)):
        other = (ray + best_k[ray, gate]) % n_rays
        half_gap = ((azimuths[other] - azimuths[ray]) % 360.0) / 2
        candidates.append(
            RotationCandidate(
                azimuth_deg=float((azimuths[ray] + half_gap) % 360.0),
                range_km=float(ranges_km[gate]),
                elevation_deg=elevation,
                shear_ms=float(best_shear[ray, gate]),
                lat=float((lats[ray, gate] + lats[other, gate]) / 2),
                lon=float((lons[ray, gate] + lons[other, gate]) / 2),
            )
        )

    return _dedupe_nearby(candidates)


def _dedupe_nearby(candidates: list[RotationCandidate], min_sep_km: float = 2.0) -> list[RotationCandidate]:
    """Collapse clustered detections (same rotation, many adjacent gates) to local maxima."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda c: -c.shear_ms)
    kept: list[RotationCandidate] = []
    for c in candidates:
        if all(_haversine_km(c.lat, c.lon, k.lat, k.lon) > min_sep_km for k in kept):
            kept.append(c)
    return kept


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def candidates_across_sweeps(
    radar: "pyart.core.Radar",
    min_tilts: int = 2,
    cluster_km: float = 4.0,
    **kwargs,
) -> list[RotationCandidate]:
    """
    Dealias, run detection on every sweep, then group detections within
    cluster_km of each other across elevation angles. Each group is reported
    once (its strongest detection, with `tilts` set) and only if it appears
    on at least min_tilts distinct elevations.
    """
    vel_field = dealias_velocity(radar)
    pooled = []
    for sweep in range(radar.nsweeps):
        pooled.extend(find_velocity_couplets(radar, sweep=sweep, vel_field=vel_field, **kwargs))

    groups: list[tuple[RotationCandidate, set[float]]] = []
    for c in sorted(pooled, key=lambda c: -c.shear_ms):
        for strongest, elevations in groups:
            if _haversine_km(c.lat, c.lon, strongest.lat, strongest.lon) <= cluster_km:
                elevations.add(round(c.elevation_deg, 1))
                break
        else:
            groups.append((c, {round(c.elevation_deg, 1)}))

    return [
        replace(strongest, tilts=len(elevations))
        for strongest, elevations in groups
        if len(elevations) >= min_tilts
    ]
