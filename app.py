"""
Minimal backend: wraps fetch_forecast.py and serves it as JSON for the
frontend to consume. Also serves the frontend itself so you can run one
process locally.

Run:
    python app.py
Then open http://localhost:5050 (not 5000: macOS AirPlay Receiver holds that
port). Set PORT to use a different one.
"""

import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from fetch_forecast import summarize_forecast, get_forecast, get_forecast_discussion, get_active_alerts, get_grid_point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# No static folder: index.html sits next to the Python sources, and a static
# route would serve those too. "/" below serves just the page.
app = Flask(__name__, static_folder=None)
CORS(app)

# Default location: Binghamton, NY area — change or pass ?lat=&lon= per request
DEFAULT_LAT = 42.0987
DEFAULT_LON = -75.9180


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/forecast")
def api_forecast():
    lat = request.args.get("lat", DEFAULT_LAT, type=float)
    lon = request.args.get("lon", DEFAULT_LON, type=float)
    try:
        summary = summarize_forecast(lat, lon)
        hourly = get_forecast(lat, lon, hourly=True)
        return jsonify({
            "location": summary["location"],
            "updated": summary["updated"],
            "daily_periods": summary["periods"],
            "hourly_periods": hourly["periods"][:12],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/discussion")
def api_discussion():
    lat = request.args.get("lat", DEFAULT_LAT, type=float)
    lon = request.args.get("lon", DEFAULT_LON, type=float)
    try:
        office = get_grid_point(lat, lon)["cwa"]
        text = get_forecast_discussion(office)
        return jsonify({"office": office, "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/alerts")
def api_alerts():
    lat = request.args.get("lat", DEFAULT_LAT, type=float)
    lon = request.args.get("lon", DEFAULT_LON, type=float)
    try:
        alerts = get_active_alerts(lat, lon)
        return jsonify({"alerts": alerts})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/radar")
def api_radar():
    lat = request.args.get("lat", DEFAULT_LAT, type=float)
    lon = request.args.get("lon", DEFAULT_LON, type=float)
    try:
        # Imported here so the forecast endpoints still work without the
        # heavy radar dependencies (Py-ART etc.) installed
        from radar_service import radar_snapshot
        return jsonify(radar_snapshot(lat, lon))
    except Exception as e:
        return jsonify({"error": str(e)}), 502


if __name__ == "__main__":
    # Debug mode's in-browser debugger can run arbitrary code, so it's opt-in
    # (FLASK_DEBUG=1) and must never be on for a deployed server.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=int(os.environ.get("PORT", 5050)))
