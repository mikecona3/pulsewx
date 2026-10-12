#!/usr/bin/env python3
"""
CLI entry point for the tornado tracker pipeline.

Examples:
    python main.py --station KBGM --mode latest
    python main.py --station KTLX --mode latest --dbz-threshold 35 --shear-threshold 20
"""

import argparse
from datetime import timezone

import pyart

from fetch_nexrad import fetch_latest
from rotation_detect import candidates_across_sweeps
from cell_track import gate_areas_km2, identify_cells
from visualize import plot_scan


def main():
    parser = argparse.ArgumentParser(description="Tornado detection & nowcast pipeline")
    parser.add_argument("--station", required=True, help="4-letter NEXRAD station code, e.g. KBGM")
    parser.add_argument("--mode", choices=["latest"], default="latest")
    parser.add_argument("--dbz-threshold", type=float, default=40.0,
                         help="Reflectivity threshold (dBZ) for storm cell identification")
    parser.add_argument("--shear-threshold", type=float, default=25.0,
                         help="Velocity shear threshold (m/s) for rotation couplet detection")
    parser.add_argument("--out", default="scan_plot.png", help="Output plot path")
    args = parser.parse_args()

    print(f"Fetching latest scan for {args.station}...")
    scan_path = fetch_latest(args.station)
    print(f"  -> {scan_path}")

    print("Reading radar volume...")
    radar = pyart.io.read_nexrad_archive(scan_path)

    print("Dealiasing velocity and running rotation detection across all sweeps...")
    candidates = candidates_across_sweeps(radar, shear_threshold_ms=args.shear_threshold)
    if candidates:
        print(f"  -> {len(candidates)} rotation candidate(s) found:")
        for c in candidates:
            print(f"     az={c.azimuth_deg:.1f}deg range={c.range_km:.1f}km "
                  f"elev={c.elevation_deg:.1f}deg shear={c.shear_ms:.1f}m/s "
                  f"tilts={c.tilts} ({c.lat:.4f}, {c.lon:.4f})")
    else:
        print("  -> No rotation candidates above threshold.")

    print("Identifying storm cells from reflectivity...")
    refl = radar.get_field(0, "reflectivity")
    lats, lons, _ = radar.get_gate_lat_lon_alt(0)
    # Stamp cells with the scan's own start time, not the time we processed
    # it — motion estimates divide distance by the time between scans.
    scan_time = pyart.util.datetime_from_radar(
        radar, only_use_cftime_datetimes=False, only_use_python_datetimes=True,
    ).replace(tzinfo=timezone.utc)
    areas = gate_areas_km2(radar.range["data"], int(radar.rays_per_sweep["data"][0]))
    cells = identify_cells(refl, lats, lons, scan_time,
                            dbz_threshold=args.dbz_threshold, gate_area_km2=areas)
    print(f"  -> {len(cells)} storm cell(s) above {args.dbz_threshold} dBZ")

    print(f"Plotting to {args.out}...")
    plot_scan(radar, sweep=0, candidates=candidates, save_path=args.out)

    print("\nNote: this looks at a single scan. Storm tracking and path "
          "projection need consecutive scans; the web app's /api/radar "
          "(radar_service.py) does that.")


if __name__ == "__main__":
    main()
