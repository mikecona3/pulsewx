"""
Plot reflectivity and velocity from a radar volume, with detected rotation
candidates overlaid.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyart

from rotation_detect import DEALIASED_FIELD, RotationCandidate

MAX_LISTED = 10  # candidates listed in the corner box; all are still circled


def plot_scan(
    radar: "pyart.core.Radar",
    sweep: int = 0,
    candidates: list[RotationCandidate] | None = None,
    save_path: str | None = None,
    extent_km: float = 160.0,
):
    """
    extent_km: half-width of the plotted area around the radar. The default
    just covers rotation detection's 150 km range.
    """
    fig = plt.figure(figsize=(14, 6))
    display = pyart.graph.RadarDisplay(radar)
    limits = (-extent_km, extent_km)

    ax1 = fig.add_subplot(1, 2, 1)
    display.plot(
        "reflectivity", sweep=sweep, ax=ax1, vmin=-20, vmax=70,
        cmap="NWSRef", colorbar_label="dBZ",
    )
    display.set_limits(limits, limits, ax=ax1)

    # Modern scan strategies sample the lowest tilts twice: once for
    # reflectivity only (no velocity), then again for velocity. Plot velocity
    # from the first sweep at or above `sweep` that actually has it, and
    # prefer the dealiased field the detector used.
    vel_field = DEALIASED_FIELD if DEALIASED_FIELD in radar.fields else "velocity"
    vel_sweep = next(
        (s for s in range(sweep, radar.nsweeps)
         if not np.ma.getmaskarray(radar.get_field(s, vel_field)).all()),
        sweep,
    )
    ax2 = fig.add_subplot(1, 2, 2)
    display.plot(
        vel_field, sweep=vel_sweep, ax=ax2, vmin=-50, vmax=50,
        cmap="NWSVel", colorbar_label="m/s",
    )
    display.set_limits(limits, limits, ax=ax2)

    if candidates:
        ranked = sorted(candidates, key=lambda c: -c.shear_ms)
        for n, c in enumerate(ranked, start=1):
            # Convert range/azimuth to the display's x/y (km from radar)
            x = c.range_km * np.sin(np.radians(c.azimuth_deg))
            y = c.range_km * np.cos(np.radians(c.azimuth_deg))
            ax2.plot(x, y, "o", markersize=14, markerfacecolor="none",
                     markeredgecolor="black", markeredgewidth=2)
            # Just a number by each circle; details go in the box below, so
            # labels on clustered candidates don't pile up on each other
            ax2.annotate(str(n), (x, y), textcoords="offset points", xytext=(7, 7),
                         fontsize=8, fontweight="bold", color="black")

        lines = [f"{n}. {c.shear_ms:.0f} m/s, {c.tilts} tilts, {c.range_km:.0f} km"
                 for n, c in enumerate(ranked[:MAX_LISTED], start=1)]
        if len(ranked) > MAX_LISTED:
            lines.append(f"+{len(ranked) - MAX_LISTED} more")
        ax2.text(0.02, 0.98, "Rotation candidates\n" + "\n".join(lines),
                 transform=ax2.transAxes, va="top", ha="left", fontsize=7.5,
                 family="monospace",
                 bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85})

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)
