"""
Pull forecasts from the National Weather Service API (api.weather.gov).

Free, no API key required. Two-step flow because NWS forecasts are issued
per grid cell, not per lat/lon:
  1. /points/{lat},{lon}  -> tells you which forecast office + grid cell covers that point
  2. that response's `forecast` / `forecastHourly` URLs -> the actual forecast

Docs: https://www.weather.gov/documentation/services-web-api
"""

import os
import time
from functools import lru_cache

import requests

# NWS asks every app to identify itself with a way to reach its owner (an
# email or website), so they can get in touch before blocking misbehaving
# traffic. Set NWS_CONTACT in the environment rather than editing this file.
_CONTACT = os.environ.get("NWS_CONTACT", "").strip()
USER_AGENT = f"PulseWX ({_CONTACT})" if _CONTACT else "PulseWX"
BASE_URL = "https://api.weather.gov"


def _headers():
    # NWS API requires a descriptive User-Agent identifying the app; it does
    # not require an API key or auth token.
    return {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET from the NWS API, retrying once after a server error or timeout —
    api.weather.gov returns occasional 500s that succeed a moment later."""
    for attempt in (1, 2):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=10)
            if resp.status_code < 500 or attempt == 2:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 2:
                raise
        time.sleep(1)


@lru_cache(maxsize=256)
def get_grid_point(lat: float, lon: float) -> dict:
    """Resolve a lat/lon to its NWS forecast office + grid cell. Cached: the
    mapping from a point to its grid cell doesn't change, and every other
    call here starts with it. Treat the returned dict as read-only."""
    resp = _get(f"{BASE_URL}/points/{lat},{lon}")
    return resp.json()["properties"]


def get_forecast(lat: float, lon: float, hourly: bool = False) -> dict:
    """
    Get the forecast for a lat/lon.

    hourly=False -> daily forecast (today, tonight, tomorrow, ... ~7 days,
        each period has a short human-readable summary)
    hourly=True  -> hour-by-hour forecast, ~7 days out, no summary text
    """
    point = get_grid_point(lat, lon)
    url = point["forecastHourly"] if hourly else point["forecast"]
    resp = _get(url)
    return resp.json()["properties"]


def get_active_alerts(lat: float, lon: float) -> list[dict]:
    """Any active NWS alerts (watches/warnings/advisories) covering this point."""
    resp = _get(f"{BASE_URL}/alerts/active", params={"point": f"{lat},{lon}"})
    return resp.json()["features"]


@lru_cache(maxsize=1)
def get_radar_stations() -> tuple[dict, ...]:
    """All WSR-88D (NEXRAD) radar sites as {id, name, lat, lon}. Cached: the
    list only changes when a radar is built or retired."""
    resp = _get(f"{BASE_URL}/radar/stations", params={"stationType": "WSR-88D"})
    return tuple(
        {
            "id": f["properties"]["id"],
            "name": f["properties"]["name"],
            "lat": f["geometry"]["coordinates"][1],
            "lon": f["geometry"]["coordinates"][0],
        }
        for f in resp.json()["features"]
    )


def get_forecast_discussion(office_id: str) -> str:
    """
    The human-written 'Area Forecast Discussion' (AFD) from the local NWS
    office — forecasters explaining their reasoning in plain text. office_id
    is the 3-letter office code from get_grid_point()['cwa'], e.g. 'BGM' for
    Binghamton.
    """
    resp = _get(f"{BASE_URL}/products/types/AFD/locations/{office_id}")
    products = resp.json()["@graph"]
    if not products:
        return ""
    latest_id = products[0]["id"]
    detail = _get(f"{BASE_URL}/products/{latest_id}")
    return detail.json()["productText"]


def summarize_forecast(lat: float, lon: float) -> dict:
    """Convenience wrapper: everything the UI needs in one call."""
    point = get_grid_point(lat, lon)
    daily = get_forecast(lat, lon, hourly=False)
    return {
        "location": {
            "city": point.get("relativeLocation", {}).get("properties", {}).get("city"),
            "state": point.get("relativeLocation", {}).get("properties", {}).get("state"),
            "office": point.get("cwa"),
        },
        "periods": daily["periods"],  # list of ~14 half-day periods (day/night pairs)
        "updated": daily.get("updateTime") or daily.get("updated"),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Fetch a NOAA/NWS forecast for a lat/lon")
    parser.add_argument("lat", type=float)
    parser.add_argument("lon", type=float)
    parser.add_argument("--discussion", action="store_true", help="Also print the local forecast discussion")
    args = parser.parse_args()

    summary = summarize_forecast(args.lat, args.lon)
    print(json.dumps(summary, indent=2)[:2000])

    if args.discussion:
        print("\n--- Forecast Discussion ---\n")
        print(get_forecast_discussion(summary["location"]["office"]))
