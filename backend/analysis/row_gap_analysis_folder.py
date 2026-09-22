"""
Sugarcane row-gap analysis for RGB orthomosaics.

Pipeline:
  1. ExG vegetation index + Otsu threshold -> vegetation mask
  2. FFT-based dominant row orientation -> rotate rows to horizontal
  3. Row detection via horizontal projection profile (peak finding)
  4. Along-row gap detection (runs of bare soil > min_gap length)
  5. Outputs: overlay PNG, per-row stats CSV, gap segments
     (GeoJSON in real coords if input is a GeoTIFF, else pixel coords)

Usage:
  python row_gap_analysis.py input.tif --gsd 0.05 --min-gap 0.5
  python row_gap_analysis.py screenshot.png --gsd 0.05   (pixel-based, GSD assumed)
"""

import os
# Fix for "PROJ: proj_identify ... DATABASE.LAYOUT.VERSION.MINOR" errors caused
# by a conflicting PostGIS/PostgreSQL PROJ install overriding rasterio's own
# bundled proj.db via a global PROJ_LIB/PROJ_DATA env var on Windows.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import argparse
import glob
import json
import math
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import binary_opening, binary_closing, binary_erosion


# ---------------------------------------------------------------- vegetation
def exg_mask(rgb, valid=None):
    """Excess Green index + Otsu -> binary vegetation mask.

    `valid` is a boolean array (True = real pixel data). Nodata/border
    pixels are excluded from the Otsu threshold computation so a black
    clipped border doesn't skew the vegetation/soil split, and are forced
    to 0 (non-vegetation) in the output mask.
    """
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    s = r + g + b + 1e-6
    exg = 2 * (g / s) - (r / s) - (b / s)
    exg_u8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if valid is None:
        valid = np.ones(rgb.shape[:2], dtype=bool)

    valid_vals = exg_u8[valid]
    thresh, _ = cv2.threshold(valid_vals.reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (exg_u8 > thresh) & valid
    mask = binary_closing(mask, np.ones((3, 3)), iterations=1)
    mask = binary_opening(mask, np.ones((3, 3)), iterations=1)
    mask = mask & valid
    return mask.astype(np.uint8), exg


def valid_data_mask(rgb, alpha=None, nodata_val=None):
    """Boolean mask of real (non-border/nodata) pixels.

    Priority: alpha band > declared nodata value > heuristic (pure black
    or pure white pixels, which clipped orthomosaic borders commonly use).
    """
    if alpha is not None:
        return alpha > 0
    if nodata_val is not None:
        return ~np.all(rgb == nodata_val, axis=-1)
    black = np.all(rgb <= 2, axis=-1)
    white = np.all(rgb >= 253, axis=-1)
    return ~(black | white)


# ---------------------------------------------------------------- height (DSM)
def chm_from_dsm(dsm, gsd, ground_win_m=4.0):
    """Canopy Height Model = DSM minus estimated ground surface.

    Ground is estimated from the DSM itself with a morphological opening
    (rolling minimum then smoothing) using a window larger than the row
    spacing, so inter-row soil pixels anchor the local ground level.
    Works well on flat/gently sloping fields; for terraced or steep
    terrain, supply a real DTM instead.
    """
    k = max(5, int(ground_win_m / gsd) | 1)  # odd kernel size
    dsm_f = dsm.astype(np.float32)
    finite = np.isfinite(dsm_f)
    fill = np.nanmedian(dsm_f[finite]) if finite.any() else 0.0
    dsm_f = np.where(finite, dsm_f, fill)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ground = cv2.morphologyEx(dsm_f, cv2.MORPH_OPEN, kernel)
    ground = cv2.GaussianBlur(ground, (k, k), 0)
    chm = dsm_f - ground
    chm[~finite] = 0.0
    return np.clip(chm, 0, None)


def fuse_masks(exg_mask_arr, chm, min_height_m, valid):
    """Vegetation = green (ExG) AND tall (CHM).

    The AND removes green-but-flat pixels (weeds, algae, green-tinged wet
    soil) and tall-but-not-green artifacts (DSM noise, poles), which are
    the two main failure modes of single-channel masks.
    """
    tall = chm >= min_height_m
    fused = exg_mask_arr.astype(bool) & tall & valid
    fused = binary_closing(fused, np.ones((3, 3)), iterations=1)
    fused = binary_opening(fused, np.ones((3, 3)), iterations=1)
    return (fused & valid).astype(np.uint8)


# ---------------------------------------------------------------- orientation
def dominant_row_angle(mask, valid=None):
    """Find dominant planting-row direction via FFT power spectrum.

    Rows are a quasi-periodic pattern; the FFT magnitude shows strongest
    energy perpendicular to the rows. Returns angle (deg) to rotate the
    image so rows become horizontal.

    If `valid` is given, the sample square is taken from well inside the
    eroded valid-data footprint so a clipped/rotated field boundary (black
    border) doesn't contaminate the frequency estimate.
    """
    h, w = mask.shape
    n = min(h, w, 1024)

    cy, cx = h // 2, w // 2
    if valid is not None and valid.any():
        # erode so the sample square (half-size n/2) fits fully inside data
        margin = n // 2 + 5
        er = binary_erosion(valid, np.ones((3, 3)), iterations=max(1, margin // 3))
        if er.any():
            ys, xs = np.where(er)
            cy, cx = int(ys.mean()), int(xs.mean())

    y0 = int(np.clip(cy - n // 2, 0, h - n))
    x0 = int(np.clip(cx - n // 2, 0, w - n))
    crop = mask[y0:y0 + n, x0:x0 + n].astype(np.float32)
    crop -= crop.mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    f = np.fft.fftshift(np.abs(np.fft.fft2(crop * win)))
    cy, cx = n // 2, n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    rad = np.hypot(yy - cy, xx - cx)
    ring = (rad > 5) & (rad < n // 4)          # ignore DC and very high freq
    ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 180
    bins = np.linspace(0, 180, 181)
    power, _ = np.histogram(ang[ring], bins=bins, weights=f[ring] ** 2)
    # smooth circularly
    k = np.ones(5) / 5
    power = np.convolve(np.r_[power[-2:], power, power[:2]], k, "same")[2:-2]
    peak_ang = bins[np.argmax(power)]
    # frequency peak lies perpendicular to row lines; rotating by
    # (90 - peak_ang) makes rows horizontal
    rot = (peak_ang - 90) % 180
    if rot > 90:
        rot -= 180
    return rot


def rotate_keep_all(img, angle, flags=cv2.INTER_NEAREST, border=0):
    h, w = img.shape[:2]
    c = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - c[0]
    M[1, 2] += nh / 2 - c[1]
    out = cv2.warpAffine(img, M, (nw, nh), flags=flags, borderValue=border)
    return out, M


# ---------------------------------------------------------------- rows & gaps
def detect_rows(mask_rot, gsd, expected_spacing_m=(0.6, 2.0)):
    """Detect row centrelines from the horizontal projection profile.

    This gives the INITIAL row count/spacing estimate only. Because a
    single global rotation angle rarely holds across an entire field
    (real fields drift/curve a few degrees from one side to the other),
    the actual per-row path is refined afterwards by `track_rows_in_strips`.
    """
    profile = mask_rot.sum(axis=1).astype(np.float32)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()
    min_dist = max(3, int(expected_spacing_m[0] / gsd))
    peaks, _ = find_peaks(profile, distance=min_dist,
                          prominence=0.15 * profile.max())
    if len(peaks) > 2:
        spacing_px = np.median(np.diff(peaks))
    else:
        spacing_px = min_dist
    return peaks, spacing_px, profile


def track_rows_in_strips(mask_rot, gsd, n_strips=14, expected_spacing_m=(0.6, 2.0),
                         max_gap_strips=2):
    """Detect row positions independently in narrow vertical strips, then
    link matching rows strip-to-strip into piecewise centerlines.

    Processes strips left-to-right as a multi-object tracking problem:
    each strip's peaks are matched to the nearest active track (nearest-
    neighbor gating by `tol`, a local-spacing-based distance), and any
    peak that doesn't match an active track SPAWNS A NEW ONE right there.

    That birth step matters for fields made of multiple sub-blocks with
    different row orientation/spacing (e.g. a narrower strip of a
    different crop or planting direction along one edge) — a single
    seed-and-grow-outward tracker only ever extends tracks from the block
    it happened to seed in, so every track dies at the block boundary and
    the other block's rows are silently dropped. Track birth lets each
    block grow its own independent set of rows.

    Tracks are also allowed to skip up to `max_gap_strips` consecutive
    strips with no match (e.g. a strip with too little vegetation to find
    a clean peak) before being considered ended, so a single noisy strip
    doesn't fragment an otherwise-continuous row.
    """
    h, w = mask_rot.shape
    strip_w = max(60, w // n_strips)
    xedges = list(range(0, w, strip_w))
    if xedges[-1] != w:
        xedges.append(w)
    strips = [(xedges[i], xedges[i + 1]) for i in range(len(xedges) - 1)]
    centers = [(x0 + x1) // 2 for x0, x1 in strips]
    n = len(strips)

    min_dist = max(3, int(expected_spacing_m[0] / gsd))

    def strip_peaks(x0, x1):
        profile = mask_rot[:, x0:x1].sum(axis=1).astype(np.float32)
        profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 9), 0).ravel()
        if profile.max() <= 0:
            return np.array([], dtype=int)
        pk, _ = find_peaks(profile, distance=min_dist, prominence=0.15 * profile.max())
        return pk

    strip_peak_list = [strip_peaks(x0, x1) for x0, x1 in strips]

    all_diffs = np.concatenate([np.diff(np.sort(p)) for p in strip_peak_list if len(p) > 1]) \
        if any(len(p) > 1 for p in strip_peak_list) else np.array([min_dist * 2])
    global_tol = max(6, int(0.5 * np.median(all_diffs))) if len(all_diffs) else min_dist

    # active_tracks: track_id -> {'y': last y, 's': last strip idx}
    # histories: track_id -> list of (strip_idx, y)
    active_tracks = {}
    histories = {}
    next_id = 0

    for s in range(n):
        peaks = strip_peak_list[s]
        matched_peak_idx = set()

        # local tolerance from this strip's own peak spacing where possible
        if len(peaks) > 1:
            tol = max(6, int(0.5 * np.median(np.diff(np.sort(peaks)))))
        else:
            tol = global_tol

        # match existing active tracks to nearest peak in this strip
        for tid in list(active_tracks.keys()):
            info = active_tracks[tid]
            if s - info["s"] > max_gap_strips:
                del active_tracks[tid]   # track has ended; history is kept
                continue
            if len(peaks) == 0:
                continue
            d = np.abs(peaks - info["y"])
            j = int(np.argmin(d))
            if d[j] <= tol and j not in matched_peak_idx:
                matched_peak_idx.add(j)
                active_tracks[tid] = {"y": int(peaks[j]), "s": s}
                histories[tid].append((s, int(peaks[j])))

        # spawn new tracks for unmatched peaks (new block / new row entering view)
        for j, y in enumerate(peaks):
            if j not in matched_peak_idx:
                active_tracks[next_id] = {"y": int(y), "s": s}
                histories[next_id] = [(s, int(y))]
                next_id += 1

    row_lines = []
    min_strips_present = max(2, n // 5)  # looser than before -- short blocks are valid
    for tid, pts in histories.items():
        if len(pts) < min_strips_present:
            continue
        xs = np.array([centers[si] for si, _ in pts])
        ys = np.array([y for _, y in pts], dtype=np.float32)
        row_lines.append((xs, ys))
    row_lines.sort(key=lambda r: r[1].mean())
    return row_lines


def centerline_for_columns(xs, ys, width, smooth=31):
    """Interpolate a tracked row's sparse (xs, ys) strip samples into a
    per-column y value, smoothed to avoid strip-boundary jitter."""
    all_x = np.arange(width)
    y_full = np.interp(all_x, xs, ys, left=ys[0], right=ys[-1])
    if smooth > 1:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        y_full = np.convolve(np.pad(y_full, pad, mode="edge"), k, mode="same")[pad:pad + width]
    return y_full


def line_band_score(score_rot, y_of_x, x0, x1, half_band):
    """Mean crop-likelihood sampled along a centerline."""
    h = score_rot.shape[0]
    vals = []
    for x in range(x0, x1 + 1):
        y = int(round(y_of_x[x]))
        yb0, yb1 = max(0, y - half_band), min(h, y + half_band + 1)
        if yb1 > yb0:
            vals.append(float(score_rot[yb0:yb1, x].mean()))
    return float(np.mean(vals)) if vals else 0.0


def center_contrast_score(score_rot, y_of_x, x0, x1, spacing_px, half_band):
    """Score whether crop-likelihood is centered on the line, not beside it."""
    center = line_band_score(score_rot, y_of_x, x0, x1, half_band)
    side_band = max(1, half_band)
    upper = line_band_score(score_rot, np.clip(y_of_x - 0.35 * spacing_px, 0, score_rot.shape[0] - 1),
                            x0, x1, side_band)
    lower = line_band_score(score_rot, np.clip(y_of_x + 0.35 * spacing_px, 0, score_rot.shape[0] - 1),
                            x0, x1, side_band)
    return center - 0.5 * (upper + lower)


def correct_centerline_phase(score_rot, y_of_x, x0, x1, spacing_px, half_band,
                             row_phase="auto"):
    """Move a detected periodic line from furrow phase to crop-row phase.

    In sparse or newly sown cane, the strongest periodic lines can be furrows
    instead of plant rows. The grid spacing is still correct, but the phase is
    shifted by about half a row. Pick the center/upper/lower half-spacing
    candidate with the highest crop-likelihood support.
    """
    h = score_rot.shape[0]
    if row_phase == "none":
        return y_of_x, 0.0

    forced = {
        "shift-up": -0.5 * spacing_px,
        "shift-down": 0.5 * spacing_px,
    }
    if row_phase in forced:
        off = forced[row_phase]
        return np.clip(y_of_x + off, 0, h - 1), off

    offsets = [0.0, -0.5 * spacing_px, 0.5 * spacing_px]
    candidates = []
    for off in offsets:
        y_shift = np.clip(y_of_x + off, 0, h - 1)
        support = line_band_score(score_rot, y_shift, x0, x1, half_band)
        contrast = center_contrast_score(score_rot, y_shift, x0, x1, spacing_px, half_band)
        candidates.append((support + contrast, off))
    best_score, best_off = max(candidates, key=lambda item: item[0])
    base_score = candidates[0][0]
    if best_off != 0.0 and best_score > base_score:
        return np.clip(y_of_x + best_off, 0, h - 1), best_off
    return y_of_x, 0.0


def refine_centerline(weight_rot, y_of_x, x0, x1, search_half, smooth=41):
    """Snap the centerline to the local crop-likelihood centroid, column by column.

    The strip-tracked centerline is only as precise as the strip width;
    residual offsets of a few px put the scan band partly on the inter-row
    furrow, which both misses real gaps and flags false ones. For each
    column, recompute y as the crop-likelihood-weighted centroid within
    +/- search_half of the current estimate (skipping empty columns so
    real gaps don't drag the line), then smooth.
    """
    h = weight_rot.shape[0]
    y_ref = y_of_x.copy()
    for x in range(x0, x1 + 1):
        yc = int(round(y_of_x[x]))
        a, b = max(0, yc - search_half), min(h, yc + search_half + 1)
        col = weight_rot[a:b, x].astype(np.float32)
        s = col.sum()
        if s > 1e-6:
            idx = np.arange(a, b)
            y_ref[x] = (idx * col).sum() / s
    # smooth only within [x0, x1]
    seg = y_ref[x0:x1 + 1]
    if smooth > 1 and len(seg) > smooth:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        seg = np.convolve(np.pad(seg, pad, mode="edge"), k, mode="same")[pad:pad + len(seg)]
        y_ref[x0:x1 + 1] = seg
    return y_ref


def gaps_along_centerline(mask_rot, y_of_x, x0, x1, half_band, gsd,
                          min_gap_m, min_veg_frac=0.15):
    """Scan a (possibly curved) row centerline for vegetation gaps.

    `y_of_x` is a per-column y array (from centerline_for_columns);
    the scan band follows it instead of a fixed row y, so curvature or
    drift in the row doesn't throw off the veg/soil classification.
    """
    h, w = mask_rot.shape
    xs_range = np.arange(x0, x1)
    veg = np.empty(len(xs_range), dtype=np.float32)
    for i, x in enumerate(xs_range):
        y = int(round(y_of_x[x]))
        yb0, yb1 = max(0, y - half_band), min(h, y + half_band + 1)
        veg[i] = mask_rot[yb0:yb1, x].mean() if yb1 > yb0 else 0.0

    cols = np.where(veg > min_veg_frac)[0]
    if len(cols) < 10:
        return [], (0, 0)
    xs, xe = cols[0], cols[-1]
    inrow = veg[xs:xe + 1] > min_veg_frac
    gaps, run = [], None
    min_gap_px = int(min_gap_m / gsd)
    for i, v in enumerate(inrow):
        if not v and run is None:
            run = i
        elif v and run is not None:
            if i - run >= min_gap_px:
                gaps.append((x0 + xs + run, x0 + xs + i))
            run = None
    if run is not None and len(inrow) - run >= min_gap_px:
        gaps.append((x0 + xs + run, x0 + xs + len(inrow)))
    return gaps, (x0 + xs, x0 + xe)


def gaps_in_row(mask_rot, y, half_band, gsd, min_gap_m, min_veg_frac=0.15):
    """Straight-line fallback (used only if strip-tracking degenerates)."""
    y0, y1 = max(0, y - half_band), min(mask_rot.shape[0], y + half_band + 1)
    band = mask_rot[y0:y1, :]
    veg = band.mean(axis=0)
    cols = np.where(veg > min_veg_frac)[0]
    if len(cols) < 10:
        return [], (0, 0)
    xs, xe = cols[0], cols[-1]
    inrow = veg[xs:xe + 1] > min_veg_frac
    gaps, run = [], None
    min_gap_px = int(min_gap_m / gsd)
    for i, v in enumerate(inrow):
        if not v and run is None:
            run = i
        elif v and run is not None:
            if i - run >= min_gap_px:
                gaps.append((xs + run, xs + i))
            run = None
    if run is not None and len(inrow) - run >= min_gap_px:
        gaps.append((xs + run, xs + len(inrow)))
    return gaps, (xs, xe)


def choose_global_row_phase(score_rot, row_lines, width, spacing_px, half_band,
                            row_phase):
    """Pick one row/furrow phase for the whole field.

    Per-row phase decisions can alternate when vegetation is sparse. A field
    should use one consistent lattice phase, so auto mode scores the original
    grid and the two half-spacing shifted grids globally.
    """
    if row_phase != "auto" or not row_lines:
        return row_phase, 0.0

    offsets = [0.0, -0.5 * spacing_px, 0.5 * spacing_px]
    scores = []
    for off in offsets:
        vals = []
        for xs_track, ys_track in row_lines:
            y_of_x = centerline_for_columns(xs_track, ys_track, width)
            x0, x1 = int(xs_track.min()), int(xs_track.max())
            y_shift = np.clip(y_of_x + off, 0, score_rot.shape[0] - 1)
            vals.append(center_contrast_score(score_rot, y_shift, x0, x1,
                                              spacing_px, half_band))
        scores.append(float(np.median(vals)) if vals else -np.inf)

    best_idx = int(np.argmax(scores))
    best_off = offsets[best_idx]
    if best_off == 0.0:
        return "none", 0.0
    return ("shift-up" if best_off < 0 else "shift-down"), best_off


# ---------------------------------------------------------------- main
def analyse(path, gsd=0.05, min_gap_m=0.5, out_prefix="out",
           dsm_path=None, min_height_m=0.25, row_phase="auto"):
    cli_gsd = gsd  # fallback if a projected gsd can't be recovered
    # -- load (GeoTIFF-aware) --
    transform = None
    crs = None
    valid = None
    if path.lower().endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        with rasterio.open(path) as src:
            n_bands = src.count
            src_crs = src.crs
            nodata_val = src.nodata

            if src_crs is not None and src_crs.is_geographic:
                # Pixel size is in degrees, not metres (e.g. clipped/exported
                # as EPSG:4326). Reproject to the local UTM zone so gsd,
                # row spacing, and gap lengths come out in real metres.
                lon = (src.bounds.left + src.bounds.right) / 2
                lat = (src.bounds.top + src.bounds.bottom) / 2
                zone = int((lon + 180) // 6) + 1
                epsg = 32600 + zone if lat >= 0 else 32700 + zone
                dst_crs = f"EPSG:{epsg}"
                print(f"[diagnostics] source CRS {src_crs} is geographic "
                      f"(degrees) -> reprojecting to {dst_crs} for metric analysis")
                dst_transform, dst_w, dst_h = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds)

                def _reproj(band_idx, resampling):
                    src_band = src.read(band_idx)
                    dst_band = np.zeros((dst_h, dst_w), dtype=src_band.dtype)
                    reproject(source=src_band, destination=dst_band,
                              src_transform=src.transform, src_crs=src_crs,
                              dst_transform=dst_transform, dst_crs=dst_crs,
                              resampling=resampling)
                    return dst_band

                r = _reproj(1, Resampling.bilinear)
                g = _reproj(2, Resampling.bilinear)
                b = _reproj(3, Resampling.bilinear)
                arr = np.dstack([r, g, b])
                alpha = _reproj(4, Resampling.nearest) if n_bands >= 4 else None
                transform, crs = dst_transform, rasterio.crs.CRS.from_string(dst_crs)
            else:
                arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                transform, crs = src.transform, src_crs
                alpha = src.read(4) if n_bands >= 4 else None

            gsd = abs(transform.a)

        if not (1e-3 < gsd < 10):
            print(f"[diagnostics] WARNING: computed gsd={gsd:.6f} m/px looks "
                  f"implausible for a drone ortho; falling back to --gsd={cli_gsd}")
            gsd = cli_gsd

        rgb = arr
        valid = valid_data_mask(
            rgb, alpha=alpha,
            nodata_val=(nodata_val, nodata_val, nodata_val) if nodata_val is not None else None,
        )
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid = valid_data_mask(rgb)

    valid_frac = valid.mean()
    print(f"[diagnostics] image size: {rgb.shape[1]}x{rgb.shape[0]} px, "
          f"gsd={gsd:.4f} m/px, valid data: {100*valid_frac:.1f}%")
    if valid_frac < 0.05:
        print("[diagnostics] WARNING: <5% of pixels flagged valid — "
              "nodata detection heuristic likely failed on this file. "
              "Check src.nodata / alpha band manually.")

    mask, exg = exg_mask(rgb, valid=valid)
    score = cv2.normalize(exg, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    score = np.where(valid, score, 0.0)
    print(f"[diagnostics] vegetation fraction (of valid pixels): "
          f"{100*mask.sum()/max(1,valid.sum()):.1f}%")

    # -- optional DSM fusion --
    if dsm_path is not None:
        import rasterio
        from rasterio.warp import reproject, Resampling
        with rasterio.open(dsm_path) as dsrc:
            dsm_native = dsrc.read(1).astype(np.float32)
            if dsrc.nodata is not None:
                dsm_native[dsm_native == dsrc.nodata] = np.nan
            # resample DSM onto the (possibly reprojected) ortho grid
            dsm = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
            if transform is not None and crs is not None:
                reproject(source=dsm_native, destination=dsm,
                          src_transform=dsrc.transform, src_crs=dsrc.crs,
                          dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.bilinear,
                          src_nodata=np.nan, dst_nodata=np.nan)
            else:
                # no georeferencing on the RGB input: sizes must match
                if dsm_native.shape == rgb.shape[:2]:
                    dsm = dsm_native
                else:
                    dsm = cv2.resize(dsm_native, (rgb.shape[1], rgb.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        chm = chm_from_dsm(dsm, gsd)
        veg_heights = chm[mask.astype(bool)]
        print(f"[diagnostics] CHM over ExG-vegetation pixels: "
              f"p25={np.percentile(veg_heights,25):.2f} m, "
              f"median={np.percentile(veg_heights,50):.2f} m, "
              f"p75={np.percentile(veg_heights,75):.2f} m")
        mask = fuse_masks(mask, chm, min_height_m, valid)
        chm_score = cv2.normalize(chm, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        score = np.where(valid, 0.6 * score + 0.4 * chm_score, 0.0)
        print(f"[diagnostics] vegetation fraction after ExG+height fusion "
              f"(min height {min_height_m} m): "
              f"{100*mask.sum()/max(1,valid.sum()):.1f}%")

    angle = dominant_row_angle(mask, valid=valid)
    mask_rot, M = rotate_keep_all(mask, angle)
    valid_rot, _ = rotate_keep_all(valid.astype(np.uint8), angle)
    valid_rot = valid_rot.astype(bool)
    mask_rot = (mask_rot.astype(bool) & valid_rot).astype(np.uint8)
    rgb_rot, _ = rotate_keep_all(rgb, angle, flags=cv2.INTER_LINEAR)
    score_rot, _ = rotate_keep_all(score, angle, flags=cv2.INTER_LINEAR, border=0)
    score_rot = np.where(valid_rot, score_rot, 0.0).astype(np.float32)

    global_rows, spacing_px, profile = detect_rows(mask_rot, gsd)
    half_band = max(2, int(spacing_px * 0.35))
    Minv = cv2.invertAffineTransform(M)

    row_lines = track_rows_in_strips(mask_rot, gsd)
    used_tracking = len(row_lines) >= max(2, len(global_rows) * 0.5)
    print(f"[diagnostics] row tracking: {'strip-tracked curved centerlines' if used_tracking else 'straight-line fallback'} "
          f"({len(row_lines) if used_tracking else len(global_rows)} rows)")
    effective_row_phase, global_phase_shift = choose_global_row_phase(
        score_rot, row_lines, mask_rot.shape[1], spacing_px,
        max(1, int(spacing_px * 0.12)), row_phase)
    if used_tracking and row_phase == "auto":
        print(f"[diagnostics] row/furrow global phase: {effective_row_phase} "
              f"({global_phase_shift:.1f} px)")

    def to_orig(x, y):
        return Minv @ np.array([x, y, 1.0])

    def draw_polyline(img, xs_col, y_of_x, color, thickness, step=25):
        pts = []
        for x in range(xs_col[0], xs_col[1] + 1, step):
            p = to_orig(x, y_of_x[x])
            pts.append(np.int32(p))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts)], False, color, thickness)
        return pts

    overlay = rgb.copy()
    all_gaps, row_stats, row_polylines = [], [], []

    if used_tracking:
        phase_shifts = []
        for ri, (xs_track, ys_track) in enumerate(row_lines):
            y_of_x = centerline_for_columns(xs_track, ys_track, mask_rot.shape[1])
            x0, x1 = int(xs_track.min()), int(xs_track.max())
            y_of_x, phase_shift = correct_centerline_phase(
                score_rot, y_of_x, x0, x1, spacing_px, max(1, int(spacing_px * 0.12)),
                row_phase=effective_row_phase)
            phase_shifts.append(phase_shift)
            y_of_x = refine_centerline(score_rot, y_of_x, x0, x1,
                                       search_half=max(3, int(spacing_px * 0.3)))
            gaps, (xs, xe) = gaps_along_centerline(mask_rot, y_of_x, x0, x1, half_band, gsd, min_gap_m)
            row_len_m = (xe - xs) * gsd
            gap_len_m = sum((b - a) for a, b in gaps) * gsd
            row_stats.append({
                "row": ri + 1, "row_length_m": round(row_len_m, 2),
                "n_gaps": len(gaps), "gap_length_m": round(gap_len_m, 2),
                "gap_pct": round(100 * gap_len_m / row_len_m, 1) if row_len_m else 0,
            })
            for a, b in gaps:
                pts = [to_orig(x, y_of_x[x]) for x in range(a, b + 1, max(1, (b - a) // 10 or 1))]
                for p1, p2 in zip(pts[:-1], pts[1:]):
                    cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)),
                             (255, 40, 40), max(2, int(spacing_px * 0.3)))
                all_gaps.append({"row": ri + 1, "len_m": round((b - a) * gsd, 2),
                                 "p1": pts[0].tolist(), "p2": pts[-1].tolist()})
            draw_polyline(overlay, (xs, xe), y_of_x, (255, 235, 60), 1)
            pts = [to_orig(x, y_of_x[x]) for x in range(xs, xe + 1, 15)]
            row_polylines.append({"row": ri + 1, "points": [p.tolist() for p in pts]})
        shifted = sum(1 for s in phase_shifts if abs(s) > 1e-6)
        if shifted:
            print(f"[diagnostics] row/furrow phase correction shifted "
                  f"{shifted}/{len(phase_shifts)} tracked rows by ~half spacing")
    else:
        # Degenerate field (too few strips tracked) -- fall back to the
        # original straight-row approach rather than producing nothing.
        for ri, y in enumerate(global_rows):
            gaps, (xs, xe) = gaps_in_row(mask_rot, y, half_band, gsd, min_gap_m)
            row_len_m = (xe - xs) * gsd
            gap_len_m = sum((b - a) for a, b in gaps) * gsd
            row_stats.append({
                "row": ri + 1, "row_length_m": round(row_len_m, 2),
                "n_gaps": len(gaps), "gap_length_m": round(gap_len_m, 2),
                "gap_pct": round(100 * gap_len_m / row_len_m, 1) if row_len_m else 0,
            })
            for a, b in gaps:
                p1, p2 = to_orig(a, y), to_orig(b, y)
                all_gaps.append({"row": ri + 1, "len_m": round((b - a) * gsd, 2),
                                 "p1": p1.tolist(), "p2": p2.tolist()})
                cv2.line(overlay, tuple(np.int32(p1)), tuple(np.int32(p2)),
                         (255, 40, 40), max(2, int(spacing_px * 0.3)))
            q1, q2 = to_orig(xs, y), to_orig(xe, y)
            cv2.line(overlay, tuple(np.int32(q1)), tuple(np.int32(q2)),
                     (255, 235, 60), 1)
            pts = [to_orig(x, y) for x in range(xs, xe + 1, 15)]
            row_polylines.append({"row": ri + 1, "points": [p.tolist() for p in pts]})

    blended = cv2.addWeighted(rgb, 0.45, overlay, 0.55, 0)
    cv2.imwrite(f"{out_prefix}_overlay.png",
                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    # -- rows-only export (no gap markers) --
    rows_overlay = rgb.copy()
    for rp in row_polylines:
        pts = np.array([np.int32(p) for p in rp["points"]])
        if len(pts) >= 2:
            cv2.polylines(rows_overlay, [pts], False, (255, 235, 60), 2)
    rows_blended = cv2.addWeighted(rgb, 0.5, rows_overlay, 0.5, 0)
    cv2.imwrite(f"{out_prefix}_rows_only.png",
                cv2.cvtColor(rows_blended, cv2.COLOR_RGB2BGR))

    row_feats = []
    for rp in row_polylines:
        if transform is not None:
            coords = [list(transform * (p[0], p[1])) for p in rp["points"]]
        else:
            coords = [[p[0], p[1]] for p in rp["points"]]
        if len(coords) >= 2:
            row_feats.append({
                "type": "Feature",
                "properties": {"row": rp["row"]},
                "geometry": {"type": "LineString", "coordinates": coords},
            })
    rows_gj = {"type": "FeatureCollection", "features": row_feats}
    if crs is not None:
        rows_gj["crs"] = {"type": "name", "properties": {"name": str(crs)}}
    with open(f"{out_prefix}_rows.geojson", "w") as f:
        json.dump(rows_gj, f)

    # -- stats + vectors --
    import csv
    with open(f"{out_prefix}_row_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row_stats[0].keys())
        w.writeheader(); w.writerows(row_stats)

    feats = []
    for g in all_gaps:
        if transform is not None:
            gx1, gy1 = transform * (g["p1"][0], g["p1"][1])
            gx2, gy2 = transform * (g["p2"][0], g["p2"][1])
        else:
            gx1, gy1 = g["p1"][0], g["p1"][1]
            gx2, gy2 = g["p2"][0], g["p2"][1]
        feats.append({
            "type": "Feature",
            "properties": {"row": g["row"], "gap_length_m": g["len_m"]},
            "geometry": {"type": "LineString",
                         "coordinates": [[gx1, gy1], [gx2, gy2]]},
        })
    gj = {"type": "FeatureCollection", "features": feats}
    if crs is not None:
        gj["crs"] = {"type": "name", "properties": {"name": str(crs)}}
    with open(f"{out_prefix}_gaps.geojson", "w") as f:
        json.dump(gj, f)

    total_gap = sum(r["gap_length_m"] for r in row_stats)
    total_row = sum(r["row_length_m"] for r in row_stats)
    summary = {
        "row_angle_deg": round(angle, 1),
        "n_rows": len(row_stats),
        "median_row_spacing_m": round(spacing_px * gsd, 2),
        "total_row_length_m": round(total_row, 1),
        "total_gap_length_m": round(total_gap, 1),
        "overall_gap_pct": round(100 * total_gap / total_row, 1) if total_row else 0,
        "n_gap_segments": len(all_gaps),
    }
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- batch (folder)
def run_batch(input_dir, out_dir, gsd, min_gap_m, dsm_arg, min_height_m, patterns,
              row_phase):
    """Run analyse() on every matching field file in a folder.

    One bad/corrupt field file does NOT abort the whole batch -- it's
    logged and skipped, so a village-scale run of many fields survives a
    handful of problem files rather than dying partway through.

    --dsm can be:
      - a folder, in which case each field's DSM is matched by filename
        stem (e.g. field7.tif -> looks for field7.* in that folder)
      - a single file, applied to every field (only sensible if all
        fields share one village-wide DSM, already the case for us)
      - omitted, in which case no height fusion is used for any field
    """
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat.strip())))
    files = sorted(set(files))

    print(f"[batch] found {len(files)} field file(s) in {input_dir} "
          f"(patterns: {patterns})")
    if not files:
        print("[batch] no matching files -- check --pattern (comma-separated "
              "globs, e.g. '*.tif,*.tiff')")
        return

    dsm_dir = dsm_arg if (dsm_arg and os.path.isdir(dsm_arg)) else None
    dsm_single = dsm_arg if (dsm_arg and not os.path.isdir(dsm_arg)) else None

    results = []
    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = os.path.join(out_dir, stem)
        print(f"\n[batch] ({i}/{len(files)}) {os.path.basename(path)} ...")

        dsm_path = None
        if dsm_dir:
            cands = glob.glob(os.path.join(dsm_dir, stem + ".*"))
            if cands:
                dsm_path = cands[0]
            else:
                print(f"[batch]   no matching DSM for '{stem}' in {dsm_dir} "
                      f"-- proceeding without height fusion for this field")
        elif dsm_single:
            dsm_path = dsm_single

        try:
            summary = analyse(path, gsd=gsd, min_gap_m=min_gap_m,
                              out_prefix=prefix, dsm_path=dsm_path,
                              min_height_m=min_height_m,
                              row_phase=row_phase)
            summary["field"] = stem
            summary["status"] = "ok"
            results.append(summary)
        except Exception as e:
            print(f"[batch]   *** ERROR on '{stem}': {e}")
            results.append({"field": stem, "status": "error", "error": str(e)})

    all_keys = []
    for r in results:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    agg_path = os.path.join(out_dir, "batch_summary.csv")
    import csv as _csv
    with open(agg_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(results)

    n_ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n[batch] DONE. {n_ok}/{len(results)} fields succeeded. "
          f"Per-field outputs in {out_dir}/, aggregate stats -> {agg_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="a single field ortho file, OR a folder "
                    "containing multiple field files (batch mode)")
    ap.add_argument("--gsd", type=float, default=0.05,
                    help="metres/pixel (ignored for GeoTIFF; read from file)")
    ap.add_argument("--min-gap", type=float, default=0.5,
                    help="minimum along-row gap length in metres")
    ap.add_argument("--out", default="out",
                    help="output prefix (single-file mode only)")
    ap.add_argument("--out-dir", default="batch_out",
                    help="output directory (folder/batch mode only) -- one "
                         "set of outputs per field, named after each file, "
                         "plus an aggregate batch_summary.csv")
    ap.add_argument("--pattern", default="*.tif,*.tiff",
                    help="comma-separated glob patterns to match in folder "
                         "mode, e.g. '*.tif,*.tiff,*.png'")
    ap.add_argument("--dsm", default=None,
                    help="optional DSM/DTM GeoTIFF for height-fused vegetation "
                         "masking. Single-file mode: one file. Folder mode: "
                         "either a folder of per-field DSMs matched by "
                         "filename stem, or one shared village-wide DSM file")
    ap.add_argument("--min-height", type=float, default=0.25,
                    help="minimum canopy height (m) to count as vegetation "
                         "when --dsm is used")
    ap.add_argument("--row-phase", choices=["auto", "none", "shift-up", "shift-down"],
                    default="auto",
                    help="row/furrow phase handling. auto picks a global phase; "
                         "none keeps detected lines; shift-up/shift-down force "
                         "a half-row-spacing correction for inverted outputs")
    args = ap.parse_args()

    if os.path.isdir(args.input):
        run_batch(args.input, args.out_dir, args.gsd, args.min_gap,
                 args.dsm, args.min_height, args.pattern.split(","),
                 args.row_phase)
    else:
        analyse(args.input, gsd=args.gsd, min_gap_m=args.min_gap,
               out_prefix=args.out, dsm_path=args.dsm, min_height_m=args.min_height,
               row_phase=args.row_phase)
