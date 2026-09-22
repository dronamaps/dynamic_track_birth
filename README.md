# Dynamic Track Birth: An RGB UAV Framework for Crop-Row Recovery

An open-source web application for comparing the paper's three row-gap methods against a georeferenced RGB orthomosaic:

- Fourier / FFT baseline
- Strip tracking without dynamic birth
- Strip tracking with dynamic birth

The user uploads a GeoTIFF (up to 500 MB), selects the minimum gap distance,
starts one asynchronous analysis for all three methods, switches between their
orthomosaic overlays in Leaflet, and downloads the selected method's per-row
CSV.

## Key analysis parameters

The table below records the main parameters tuned in
[`row_detection_baselines.py`](backend/analysis/row_detection_baselines.py) and
[`row_gap_analysis_folder.py`](backend/analysis/row_gap_analysis_folder.py).
Values marked **CLI** can be changed when invoking the scripts; **internal**
values are shared experimental settings and require a code change. Keeping the
shared settings identical across methods is important: the comparison is meant
to isolate the row detector, not changes in vegetation segmentation or gap
scoring.

| Stage | Parameter | Default / tuned value | Scope | Rationale |
|---|---|---:|---|---|
| Spatial scale | `--gsd` / `gsd` | `0.05 m/px` | Both; **CLI** fallback | Converts pixel distances to metres. A valid projected GeoTIFF supplies its own GSD, so this value is used only for ungeoreferenced images or implausible raster metadata. |
| Gap definition | `--min-gap` / `min_gap_m` | `0.5 m` | Both; **CLI** | Rejects short soil runs caused by normal within-canopy texture while retaining agronomically meaningful missing-row segments. This is the minimum detected gap length exposed in the web UI. |
| DSM filtering | `--min-height` / `min_height_m` | `0.25 m` | Both; **CLI**, when `--dsm` is supplied | Requires ExG vegetation to have plausible canopy height, removing green but flat weeds/soil while avoiding an overly strict threshold for young crop. |
| Ground estimation | `ground_win_m` | `4.0 m` | Shared core; **internal** | Sets the morphological-opening window used to estimate ground from a DSM. It is deliberately wider than typical row spacing so inter-row soil anchors the local ground surface. |
| RGB/DSM score fusion | ExG : CHM weights | `0.6 : 0.4` | Both; **internal**, when DSM is used | Keeps spectral greenness as the primary signal while giving height enough influence to suppress low vegetation and colour artefacts. |
| Mask cleanup | closing/opening kernel | `3 x 3`, one pass each | Shared core; **internal** | Fills isolated holes and removes speckle without joining neighbouring crop rows at normal UAV resolution. |
| FFT orientation | sample and frequency band | Up to `1024 x 1024` px; radius `5` to `n/4`; `5°` circular smoothing | FFT and strip; **internal** | Caps FFT cost, removes the DC/field-boundary component and very high-frequency noise, and stabilizes the dominant row-angle estimate. |
| Expected row spacing | `expected_spacing_m` | `(0.6, 2.0) m` (`0.6 m` is the active minimum-distance bound) | FFT projection and both strip trackers; **internal** | Prevents multiple peaks from being assigned to one physical row. The tuple documents the intended crop-spacing range; the present peak finder uses its lower bound. |
| Row-peak strength | peak `prominence` | `0.15 x` local profile maximum | FFT projection and both strip trackers; **internal** | Suppresses weak vegetation/noise peaks while allowing partially planted or gapped rows to remain detectable. |
| Strip partitioning | `n_strips` / minimum strip width | `14` / `60 px` | Strip and strip-without-birth; **internal** | Provides enough longitudinal samples to follow curvature while keeping each strip wide enough for a stable projection profile. |
| Track association | match tolerance | Half the median local row spacing, at least `6 px` | Strip and strip-without-birth; **internal** | Allows gradual row drift between strips but limits jumps onto an adjacent row. |
| Missed-strip tolerance | `max_gap_strips` | `2` | Strip and strip-without-birth; **internal** | Lets a track survive short areas with weak vegetation without bridging long discontinuities. |
| Track retention | `min_strips_present` | `max(2, n_strips_actual / 5)` | Strip and strip-without-birth; **internal** | Removes very short noise tracks while preserving valid short rows and rows that enter partway through a multi-block field. |
| Centerline smoothing | `smooth` | `31 px` interpolation; `41 px` refinement | Strip methods; **internal** | Reduces strip-boundary and column-level jitter without forcing genuinely curved rows to become straight. |
| Row/furrow phase | `--row-phase` / `row_phase` | `auto`; candidates `0` and `+/- 0.5 x` spacing | Folder script: **CLI**; baseline strip methods: **internal** | Corrects the half-spacing ambiguity that occurs when periodic furrows are stronger than crop rows. Forced up/down modes are available for difficult fields. |
| Centerline refinement | search half-width | `max(3 px, 0.30 x spacing)` | Strip methods; **internal** | Snaps a coarse strip track to the local crop-likelihood centroid while keeping the search inside the expected row neighbourhood. |
| Gap scan band | `half_band` | `max(2 px, 0.35 x spacing)` | All methods; **internal** | Samples enough of the canopy around a centerline to tolerate small localization errors without routinely including adjacent rows. |
| Vegetation occupancy | `min_veg_frac` | `0.15` | All gap scoring; **internal** | Treats a column as occupied when at least 15% of its cross-row scan band is vegetation, making scoring tolerant of thin or discontinuous canopy. |
| Ground-truth matching | `--match-dist` / `--len-ratio` | `1.0 m` / `0.5` | Baseline `score`; **CLI** | Counts a detection as a match only when its midpoint is spatially close and its length differs by no more than 50% of the longer segment; this supports reproducible precision/recall/F1 evaluation. |

`row_detection_baselines.py` also exposes `--methods` (default
`fft,strip`) to select the comparison set; add `strip_nb` to run
the no-dynamic-birth ablation. File patterns, output paths, and DSM paths are
operational inputs rather than detection tuning parameters and are therefore
not included in the table.

## Architecture

```mermaid
flowchart LR
    U["Browser + Leaflet"] --> P["Caddy"]
    P --> F["Vinext frontend"]
    P --> A["FastAPI"]
    A --> R["Redis queue"]
    R --> W["Celery worker"]
    W --> D["Shared job storage"]
    A --> D
```

All runtime components are open source:

- Leaflet with a satellite imagery basemap for the browser map
- FastAPI for upload, status, GeoJSON, CSV, and raster tile endpoints
- Celery and Redis for background analysis
- Rasterio/GDAL, Rio-Tiler, OpenCV, SciPy, and the supplied research scripts
- Caddy as the same-origin reverse proxy

## Run

Requirements: Docker Engine with the Compose plugin and at least 8 GB RAM.
More memory is recommended for large or high-resolution orthomosaics.

```bash
docker compose up --build
```

Open `http://localhost:8080`.

The first build downloads the Python geospatial packages and JavaScript
dependencies. Submitted inputs and generated outputs remain in the Docker
volume `analysis_data`.

## Input requirements

- `.tif` or `.tiff`
- maximum size: 500 MB
- at least three RGB bands
- a valid CRS and affine geotransform

The pipeline derives GSD from a projected GeoTIFF. A geographic source is
reprojected to the local UTM zone by the supplied analysis code before metric
gap measurement.

## Outputs

Each successful job exposes:

- tiled orthomosaic for Leaflet
- per-method WGS 84 row-centerline GeoJSON
- per-method WGS 84 detected-gap GeoJSON
- per-method CSV with row length, number of gaps, gap length, and gap percentage
- per-method summary values for detected rows, gaps, total gap length, and
  overall gap rate

## Operations

Only one worker process is enabled by default because a single analysis can be
memory intensive. Increase `WORKER_CONCURRENCY` only when the host has enough
RAM for multiple orthomosaics.

For a remote Ubuntu server, place TLS and authentication in front of Caddy
before exposing the application publicly. Job retention and user accounts are
deliberately outside this first release.
