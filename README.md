# PulseWX

A web weather app backed by NOAA / National Weather Service data, plus an
experimental tornado-detection pipeline that works on raw NEXRAD radar
scans.

## Forecast web app

- `app.py`: Flask backend. It serves `index.html` and three JSON endpoints,
  each taking `?lat=&lon=`:
  - `/api/forecast`: daily periods plus the next 12 hours
  - `/api/discussion`: the local NWS office's Area Forecast Discussion
  - `/api/alerts`: active watches, warnings and advisories
- `fetch_forecast.py`: api.weather.gov client used by the backend. It's free
  and needs no API key, but NWS asks every app to include a contact (email
  or website) in its User-Agent. Set it with the `NWS_CONTACT` environment
  variable.
- `/api/radar` (also `?lat=&lon=`): live radar from the nearest NEXRAD
  station. See "Radar on the page" below.
- `index.html`: the frontend, a single static page with no build step.

### Location

Click the city name at the top of the page to change location. You can
either:

- search for a US city or town (through the free Open-Meteo geocoding API,
  no key needed), or
- use your current location (through the browser's Geolocation API, which
  needs `localhost` or HTTPS).

The chosen location is saved in the browser's `localStorage`. The default
is Binghamton, NY. NWS only covers the US, so other locations show an
error with a button to pick again.

### Running

```
pip install -r requirements.txt
NWS_CONTACT="you@example.com" python app.py
```

Then open http://localhost:5050. It doesn't use port 5000 because macOS's
AirPlay Receiver already listens there. Set `PORT` to use a different port,
for example `PORT=8080 python app.py`.

`FLASK_DEBUG=1` turns on Flask's debug mode (auto-reload and an in-browser
debugger). Only use it on your own machine: the debugger can run arbitrary
code, so it must never be on for a deployed server.

### Radar on the page

The Radar section shows the nearest NEXRAD radar's latest results on a map:
possible rotation, storm cores (40+ dBZ, at least 10 km²), and each storm's
projected path for the next 30 minutes. Above the map it says whether
rotation was found, and whether a storm is over you or heading your way.
It refreshes every 5 minutes. It's experimental and is labeled as not an
official warning.

`radar_service.py` does the work on request, with no background job:

- The first request for a radar processes its 3 latest scans, so expect
  about 30 seconds. After that, requests only process scans that have
  arrived since. Results are kept in memory, so a restart starts over.
- Storms get a projected path only after they've been seen on 3 scans. If a
  storm's own motion disagrees with the typical motion of storms around it
  by more than 25 km/h, the typical motion is used instead, and the page
  says so. Merging and splitting storms make individual tracks jumpy.
- Locations more than 250 km from every radar get "no radar near this
  location".
- Downloaded scans are deleted after processing.

The map uses Leaflet (from cdnjs) with OpenStreetMap tiles. OSM's tile
servers are fine for personal use; a public deployment should switch to a
commercial tile provider per OSM's tile usage policy.

## Tornado detection pipeline (experimental, CLI only)

```
python main.py --station KBGM
```

1. `fetch_nexrad.py` downloads the latest Level II scan from the public
   `unidata-nexrad-level2` AWS bucket into `data/` in this folder. (The old
   `noaa-nexrad-level2` bucket now returns Access Denied.)
2. `rotation_detect.py` corrects velocity aliasing, then looks on every
   elevation angle for inbound and outbound gates side by side on
   neighboring beams (cyclonic rotation). It skips clear air (under 20 dBZ)
   and anything beyond 150 km, and keeps only rotation seen on at least two
   elevation angles. Results are rotation *candidates*, not confirmed
   tornadoes.
3. `cell_track.py` finds storm cells by thresholding reflectivity, and
   tracks them across scans (used by the web app).
4. `visualize.py` saves a reflectivity and velocity plot, zoomed to the 150
   km detection range, with the candidates circled, numbered and listed.

As a sanity check, on a clear-sky KTLX scan it finds nothing. On the Moore,
OK tornado (KTLX, 2013-05-20 20:12Z) its strongest candidate is in west
Moore on all 14 elevation angles. That scan can be fetched with
`fetch_nexrad.fetch_at()`.

`main.py` looks at one scan, so it doesn't track storms or project paths;
the web app's `/api/radar` does.
