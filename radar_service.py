"""
Live radar for the web app's /api/radar endpoint.

For the NEXRAD station nearest a location, this processes each new volume
scan as it appears — rotation detection, storm cell identification,
scan-to-scan tracking and path projection — and keeps the results in memory.

Work happens on request rather than in a background thread: the first
request for a station processes its three most recent scans (roughly 30
seconds), and later requests only process scans that have arrived since (a
new one lands every 4–10 minutes). The web page re-requests every 5 minutes,
which keeps tracks going for as long as someone has it open.
"""

import math
import os
import threading
import time
from datetime import timedelta

import numpy as np
import pyart

from cell_track import _haversine_km, gate_areas_km2, identify_cells, update_tracks
from fetch_forecast import get_radar_stations
from fetch_nexrad import DATA_DIR, download, list_recent_scans, scan_time
from nowcast import project_path
from rotation_detect import candidates_across_sweeps

BOOTSTRAP_SCANS = 3  # scans processed on a station's first request
MIN_SCANS_FOR_PATH = 3  # a storm needs this many scans before its path is projected
MAX_DEVIATION_KMH = 25.0  # own motion further than this from the area's typical motion isn't trusted
MAX_SCAN_GAP = timedelta(minutes=20)  # longer than this between scans and tracks start over
MAX_NEW_SCANS = 6  # most scans to catch up on in one request
RECHECK_SECONDS = 60  # how often to look for a new scan
RANGE_KM = 150.0  # matches rotation detection's range limit
MAX_STATION_DISTANCE_KM = 250.0  # farther than this from every radar and there's no useful coverage
# Faster than this and the "motion" is almost certainly two different cells
# matched across scans, so no path is projected.
MAX_STORM_SPEED_KMH = 120.0


def nearest_station(lat: float, lon: float) -> dict:
    return min(get_radar_stations(), key=lambda s: _haversine_km(lat, lon, s["lat"], s["lon"]))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Compass bearing from point 1 to point 2."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


class StationTracker:
    """Rolling state for one radar station: the latest rotation candidates
    and storm tracks built up across consecutive scans."""

    def __init__(self, station: dict):
        self.station = station
        self.lock = threading.Lock()
        self.tracks = []
        self.candidates = []
        self.last_key = None
        self.last_scan_time = None
        self.last_check = None

    def refresh(self):
        # Concurrent requests for the same station wait here rather than
        # processing the same scan twice.
        with self.lock:
            if self.last_check is not None and time.monotonic() - self.last_check < RECHECK_SECONDS:
                return
            keys = [k for k in list_recent_scans(self.station["id"]) if scan_time(k)]
            if not keys and self.last_key is None:
                raise RuntimeError(f"No recent scans found for radar {self.station['id']}")
            # Keys sort by time, so anything after the last one processed is new
            new = [k for k in keys if self.last_key is None or k > self.last_key]
            if new and (self.last_scan_time is None or scan_time(new[-1]) - self.last_scan_time > MAX_SCAN_GAP):
                new = new[-BOOTSTRAP_SCANS:]  # existing tracks are stale; start fresh
            for key in new[-MAX_NEW_SCANS:]:
                try:
                    self._process(key)
                except Exception as e:  # noqa: BLE001 — one bad file shouldn't stall the station
                    print(f"radar: skipping {key}: {e}")
                    self.last_key = key
            if self.last_scan_time is None:
                raise RuntimeError(f"Couldn't process any recent scans from radar {self.station['id']}")
            self.last_check = time.monotonic()

    def _process(self, key: str):
        t = scan_time(key)
        if self.last_scan_time is not None and t - self.last_scan_time > MAX_SCAN_GAP:
            self.tracks = []  # too long since the last scan to match storms across it

        already_local = os.path.exists(os.path.join(DATA_DIR, key.split("/")[-1]))
        path = download(key)
        try:
            radar = pyart.io.read_nexrad_archive(path)
        finally:
            # A volume is 5–15 MB and a new one lands every few minutes; only
            # keep files someone downloaded on purpose (e.g. with main.py)
            if not already_local:
                os.remove(path)

        candidates = candidates_across_sweeps(radar)
        lats, lons, _ = radar.get_gate_lat_lon_alt(0)
        areas = gate_areas_km2(radar.range["data"], int(radar.rays_per_sweep["data"][0]))
        cells = identify_cells(radar.get_field(0, "reflectivity"), lats, lons, t, gate_area_km2=areas)
        self.tracks = update_tracks(self.tracks, cells)
        self.candidates = candidates
        self.last_key, self.last_scan_time = key, t

    def snapshot(self, lat: float, lon: float) -> dict:
        """Current state as JSON-ready data, with distances and bearings
        measured from (lat, lon)."""
        with self.lock:
            station = self.station

            def place(plat, plon):
                return {
                    "lat": plat, "lon": plon,
                    "distance_km": round(float(_haversine_km(lat, lon, plat, plon)), 1),
                    "bearing_deg": round(_bearing_deg(lat, lon, plat, plon)),
                }

            rotation = [
                {**place(c.lat, c.lon), "shear_ms": round(c.shear_ms, 1), "tilts": c.tilts}
                for c in sorted(self.candidates, key=lambda c: -c.shear_ms)
            ]

            in_range = [
                t for t in self.tracks
                if _haversine_km(station["lat"], station["lon"], t.latest.lat, t.latest.lon) <= RANGE_KM
            ]
            # Motion is only trusted once a storm has been seen on a few
            # scans; two-scan estimates were mostly noise in testing.
            velocities = {}
            for t in in_range:
                v = t.velocity_kmh() if len(t.history) >= MIN_SCANS_FOR_PATH else None
                if v and math.hypot(*v) <= MAX_STORM_SPEED_KMH:
                    velocities[t.id] = v
            typical = None
            if len(velocities) >= 3:
                east, north = zip(*velocities.values())
                typical = (float(np.median(east)), float(np.median(north)))

            storms = []
            for track in in_range:
                cell = track.latest
                motion, path = None, []
                velocity, source = velocities.get(track.id), "own"
                # Merging and splitting make some centroids drift; storms in
                # one area mostly move together, so fall back to their
                # typical motion when this one's disagrees with it a lot.
                if velocity and typical and math.dist(velocity, typical) > MAX_DEVIATION_KMH:
                    velocity, source = typical, "area"
                if velocity:
                    east, north = velocity
                    motion = {
                        "speed_kmh": round(math.hypot(east, north)),
                        "toward_deg": round(math.degrees(math.atan2(east, north)) % 360),
                        "source": source,
                    }
                    path = project_path(track, velocity=velocity)
                # Treat the storm as a disk of its core's area; the projected
                # center's uncertainty circle plus that radius is where the
                # storm could be at each lead time.
                radius_km = math.sqrt(cell.area_km2 / math.pi)
                distance = float(_haversine_km(lat, lon, cell.lat, cell.lon))
                if distance <= radius_km:
                    reaches_in = 0  # already over (lat, lon)
                else:
                    # Only storms getting closer count: the uncertainty circle
                    # keeps growing, so for a storm moving away it can still
                    # end up covering (lat, lon) from behind.
                    reaches_in = next(
                        (p.lead_time_min for p in path
                         if (d := _haversine_km(lat, lon, p.lat, p.lon)) < distance
                         and d <= p.radius_km + radius_km),
                        None,
                    )
                storms.append({
                    "id": track.id,
                    **place(cell.lat, cell.lon),
                    "radius_km": round(radius_km, 1),
                    "max_dbz": round(cell.max_dbz),
                    "scans": len(track.history),
                    "motion": motion,
                    "path": [
                        {"minutes": p.lead_time_min, "lat": float(p.lat), "lon": float(p.lon),
                         "radius_km": round(p.radius_km, 1)}
                        for p in path
                    ],
                    "reaches_you_in_min": reaches_in,
                })
            storms.sort(key=lambda s: s["distance_km"])

            return {
                "station": {**station, "range_km": RANGE_KM},
                "scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
                "rotation": rotation,
                "storms": storms,
            }


_trackers: dict[str, StationTracker] = {}
_trackers_lock = threading.Lock()


def radar_snapshot(lat: float, lon: float) -> dict:
    station = nearest_station(lat, lon)
    if _haversine_km(lat, lon, station["lat"], station["lon"]) > MAX_STATION_DISTANCE_KM:
        raise ValueError("no NEXRAD radar near this location")
    with _trackers_lock:
        tracker = _trackers.setdefault(station["id"], StationTracker(station))
    tracker.refresh()
    return tracker.snapshot(lat, lon)
