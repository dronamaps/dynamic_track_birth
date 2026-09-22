"""
Row-detection BASELINES for the row-gap paper.

Purpose
-------
Run three row-detection methods on the *same* imagery, through the *same*
segmentation and the *same* gap scorer, so the only thing that differs is the
row-detection algorithm. This is what makes the comparison fair and what a
reviewer will look for:

    FFT     : single global row angle (FFT spectrum) -> straight parallel rows
    Hough   : straight line voting (HoughLinesP) -> clustered straight rows
              (unconstrained -- vulnerable to grid-aliasing misorientation
              on dense, regularly spaced rows; see detect_hough docstring)
    Hough_c : same, but the orientation vote is constrained to a window
              around an FFT prior -- isolates misorientation from Hough's
              other, structural weaknesses (curves, gaps, block boundaries)
    strip   : your strip-based tracker with track-birth (the proposed method)

Every method reuses YOUR code for:
    - loading / reprojection / GSD                (load_field, mirrors analyse)
    - ExG + Otsu vegetation mask (+ optional DSM)  (core.exg_mask / fuse_masks)
    - along-row gap detection & metrics            (core.gaps_along_centerline)

...so any difference in the output numbers is attributable to row detection
alone. Do NOT change the mask or gap parameters between methods.

What it produces
----------------
Per field, per method:
    {prefix}_{method}_overlay.png     rows (yellow) + gaps (red) on the ortho
    {prefix}_{method}_rows.geojson    detected row centerlines (one LineString
                                     per row -- straight for fft/hough, curved
                                     for strip/strip_nb), world coords if GeoTIFF
    {prefix}_{method}_row_stats.csv   per-row length/gap-count/gap-% table
    {prefix}_{method}_gaps.geojson    detected gap vectors, world coords if GeoTIFF
Aggregate: 
    comparison.csv                  one row per (field, method) with counts,
                                    total gap length, overall gap %, spacing

Note: *_rows.geojson is the row-DETECTION output itself (how each method
traced the rows) -- distinct from *_gaps.geojson, which is the downstream
gap measurement. Comparing rows.geojson across methods on a curved/multi-
block field is the direct visual evidence for the paper's "conventional
methods lose track" claim (fft/hough draw straight lines that drift or stop
short; strip/strip_nb follow the true curve).

Precision / recall
------------------
This harness produces each method's DETECTED gaps. To get precision/recall/F1
you also need manual ground-truth gap annotations (a GeoJSON of LineStrings).
When you have them, run:

    python row_detection_baselines.py score \
        --detected comparison_out/field7_fft_gaps.geojson \
        --truth    annotations/field7_truth.geojson

and it will report TP/FP/FN and P/R/F1 for that method+field. Repeat per
method to fill the paper's baseline table. See score_against_truth() for the
expected annotation format and the matching rule (tune to your schema).

Usage
-----
    # one field, all three methods:
    python row_detection_baselines.py run field7.tif --gsd 0.03 --min-gap 0.5

    # a folder of fields (village-scale), all three methods -> comparison.csv:
    python row_detection_baselines.py batch fields/ --out-dir comparison_out \
        --gsd 0.03 --min-gap 0.5 --dsm village_dsm.tif

    # point at your core module explicitly if the filename differs:
    python row_detection_baselines.py batch fields/ --core row_gap_analysis.py
"""

import os
os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)

import argparse
import csv
import glob
import importlib.util
import json
import sys
import numpy as np
import cv2
from scipy.signal import find_peaks


# --------------------------------------------------------------- core import
def load_core(path_hint=None):
    """Import the main pipeline module so we reuse its exact functions.

    Tries an explicit --core path first, then common filenames next to this
    script. We only need its module-level helper functions (all importable).
    """
    candidates = []
    if path_hint:
        candidates.append(path_hint)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "row_gap_analysis.py"),
        os.path.join(here, "row_gap_analysis_folder.py"),
        "row_gap_analysis.py",
        "row_gap_analysis_folder.py",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            spec = importlib.util.spec_from_file_location("rowgap_core", c)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"[core] using pipeline module: {c}")
            return mod
    raise FileNotFoundError(
        "Could not find your pipeline module. Pass it with --core "
        "(e.g. --core row_gap_analysis.py)."
    )


# ------------------------------------------------------------------- loading
def load_field(core, path, gsd_cli=0.05, dsm_path=None, min_height_m=0.25):
    """Load + segment a field EXACTLY as analyse() does, up to the mask.

    Returns everything the detectors need, so all three share one identical
    vegetation mask and score field. This mirrors the front half of
    core.analyse(); keep it in sync if you change the loader there. (Ideally,
    factor analyse()'s loader into a shared function and call it here.)
    """
    transform = crs = valid = None
    if path.lower().endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
        with rasterio.open(path) as src:
            n_bands, src_crs, nodata_val = src.count, src.crs, src.nodata
            if src_crs is not None and src_crs.is_geographic:
                lon = (src.bounds.left + src.bounds.right) / 2
                lat = (src.bounds.top + src.bounds.bottom) / 2
                zone = int((lon + 180) // 6) + 1
                epsg = (32600 if lat >= 0 else 32700) + zone
                dst_crs = f"EPSG:{epsg}"
                dt, dw, dh = calculate_default_transform(
                    src_crs, dst_crs, src.width, src.height, *src.bounds)

                def _rp(bi, rs):
                    sb = src.read(bi)
                    db = np.zeros((dh, dw), dtype=sb.dtype)
                    reproject(source=sb, destination=db,
                              src_transform=src.transform, src_crs=src_crs,
                              dst_transform=dt, dst_crs=dst_crs, resampling=rs)
                    return db
                r = _rp(1, Resampling.bilinear)
                g = _rp(2, Resampling.bilinear)
                b = _rp(3, Resampling.bilinear)
                arr = np.dstack([r, g, b])
                alpha = _rp(4, Resampling.nearest) if n_bands >= 4 else None
                transform = dt
                crs = rasterio.crs.CRS.from_string(dst_crs)
            else:
                arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                transform, crs = src.transform, src_crs
                alpha = src.read(4) if n_bands >= 4 else None
            gsd = abs(transform.a)
        if not (1e-3 < gsd < 10):
            gsd = gsd_cli
        rgb = arr
        nd = (nodata_val,) * 3 if nodata_val is not None else None
        valid = core.valid_data_mask(rgb, alpha=alpha, nodata_val=nd)
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gsd = gsd_cli
        valid = core.valid_data_mask(rgb)

    mask, exg = core.exg_mask(rgb, valid=valid)
    score = cv2.normalize(exg, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    score = np.where(valid, score, 0.0)

    if dsm_path is not None:
        import rasterio
        from rasterio.warp import reproject, Resampling
        with rasterio.open(dsm_path) as dsrc:
            dsm_native = dsrc.read(1).astype(np.float32)
            if dsrc.nodata is not None:
                dsm_native[dsm_native == dsrc.nodata] = np.nan
            dsm = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
            if transform is not None and crs is not None:
                reproject(source=dsm_native, destination=dsm,
                          src_transform=dsrc.transform, src_crs=dsrc.crs,
                          dst_transform=transform, dst_crs=crs,
                          resampling=Resampling.bilinear,
                          src_nodata=np.nan, dst_nodata=np.nan)
            elif dsm_native.shape == rgb.shape[:2]:
                dsm = dsm_native
            else:
                dsm = cv2.resize(dsm_native, (rgb.shape[1], rgb.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        chm = core.chm_from_dsm(dsm, gsd)
        mask = core.fuse_masks(mask, chm, min_height_m, valid)
        chm_s = cv2.normalize(chm, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        score = np.where(valid, 0.6 * score + 0.4 * chm_s, 0.0).astype(np.float32)

    return {"rgb": rgb, "mask": mask, "score": score, "valid": valid,
            "gsd": gsd, "transform": transform, "crs": crs}


# ------------------------------------------------------------ detector output
class Frame:
    """Rotated working frame + straight/curved row centerlines.

    rows: list of (x0, x1, y_of_x) in the rotated frame, where y_of_x is a
    full-width per-column float array. Straight-line methods just use a
    constant y_of_x; the strip tracker uses a curved one.
    """
    def __init__(self, mask_rot, score_rot, M, gsd, spacing_px, rows):
        self.mask_rot = mask_rot
        self.score_rot = score_rot
        self.M = M
        self.gsd = gsd
        self.spacing_px = spacing_px
        self.rows = rows


def _rotate_all(core, field, angle):
    mask_rot, M = core.rotate_keep_all(field["mask"], angle)
    valid_rot, _ = core.rotate_keep_all(field["valid"].astype(np.uint8), angle)
    valid_rot = valid_rot.astype(bool)
    mask_rot = (mask_rot.astype(bool) & valid_rot).astype(np.uint8)
    score_rot, _ = core.rotate_keep_all(field["score"], angle,
                                        flags=cv2.INTER_LINEAR, border=0)
    score_rot = np.where(valid_rot, score_rot, 0.0).astype(np.float32)
    return mask_rot, score_rot, M


# ------------------------------------------------------------- FFT baseline
def detect_fft(core, field):
    """Global FFT row angle + horizontal projection peaks -> straight rows.

    This is the 'conventional' baseline: one angle and one spacing for the
    whole field, rows drawn as straight parallel lines. No per-row path, no
    curvature, no re-initialisation. Fails where the field drifts/curves or
    contains multiple blocks with different geometry.
    """
    angle = core.dominant_row_angle(field["mask"], valid=field["valid"])
    mask_rot, score_rot, M = _rotate_all(core, field, angle)
    peaks, spacing_px, _ = core.detect_rows(mask_rot, field["gsd"])
    w = mask_rot.shape[1]
    rows = [(0, w - 1, np.full(w, float(y), dtype=np.float32)) for y in peaks]
    rows.sort(key=lambda r: r[2][0])
    return Frame(mask_rot, score_rot, M, field["gsd"], spacing_px, rows)


# ------------------------------------------------------------ Hough baseline
def _detect_hough_core(core, field, ang_tol_deg=10.0, orientation_prior_deg=None,
                       prior_window_deg=25.0):
    """Shared Hough logic. If orientation_prior_deg is given, the dominant-
    angle vote is restricted to orientations within prior_window_deg of the
    prior BEFORE picking the winner -- see detect_hough_constrained for why.
    """
    mask = field["mask"]
    gsd = field["gsd"]
    m255 = (mask * 255).astype(np.uint8)
    m255 = cv2.dilate(m255, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(m255, 40, 120)

    h, w = mask.shape
    min_len = max(20, int(0.15 * w))
    max_gap = max(5, int(0.02 * w))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(30, min_len // 2),
                            minLineLength=min_len, maxLineGap=max_gap)
    if lines is None or len(lines) == 0:
        return Frame(mask, field["score"], np.eye(2, 3, dtype=np.float32),
                     gsd, max(3, int(0.6 / gsd)), [])

    segs = lines[:, 0, :]  # x1,y1,x2,y2
    dx = segs[:, 2] - segs[:, 0]
    dy = segs[:, 3] - segs[:, 1]
    ang = (np.degrees(np.arctan2(dy, dx)) + 180) % 180  # 0..180
    orient = np.where(ang > 90, ang - 180, ang)
    lengths = np.hypot(dx, dy)

    bin_edges = np.arange(-90, 91, 2)
    hist, edges_b = np.histogram(orient, bins=bin_edges, weights=lengths)

    if orientation_prior_deg is not None:
        # Zero out histogram bins outside the prior window BEFORE picking the
        # winner. This is the fix for grid-aliasing: on dense, regularly
        # spaced rows, a spurious diagonal that happens to clip through many
        # rows' edges at evenly-spaced intervals can rack up more total
        # length than the true row direction and win the unconstrained vote
        # outright (the failure visible in the field photo -- cyan lines
        # running in a completely different direction from the crop rows).
        # Restricting candidate angles to a window around an independent
        # orientation estimate (here, the FFT angle) removes that failure
        # mode while leaving Hough's other real weaknesses -- fragmentation
        # on curves, scatter at block boundaries and gaps -- untouched.
        bin_centers = bin_edges[:-1] + 1.0
        d = np.abs(((bin_centers - orientation_prior_deg + 90) % 180) - 90)
        hist = np.where(d <= prior_window_deg, hist, 0.0)
        if hist.sum() == 0:
            # nothing survived the window -- fall back to the unconstrained
            # winner rather than silently returning nothing (a telling
            # result in itself: report it, don't hide it)
            hist, edges_b = np.histogram(orient, bins=bin_edges, weights=lengths)

    dom = edges_b[int(np.argmax(hist))] + 1.0

    keep = np.abs(((orient - dom + 90) % 180) - 90) <= ang_tol_deg
    segs = segs[keep]
    if len(segs) == 0:
        return Frame(mask, field["score"], np.eye(2, 3, dtype=np.float32),
                     gsd, max(3, int(0.6 / gsd)), [])

    mask_rot, score_rot, M = _rotate_all(core, field, dom)
    ones = np.ones((len(segs), 1), dtype=np.float32)
    p1 = np.hstack([segs[:, 0:2].astype(np.float32), ones])  # (N,3)
    p2 = np.hstack([segs[:, 2:4].astype(np.float32), ones])
    r1 = (M @ p1.T).T  # (N,2) rotated coords
    r2 = (M @ p2.T).T
    ymid = 0.5 * (r1[:, 1] + r2[:, 1])
    xa = np.minimum(r1[:, 0], r2[:, 0])
    xb = np.maximum(r1[:, 0], r2[:, 0])

    spacing_px = max(3, int(0.6 / gsd))
    order = np.argsort(ymid)
    ys = ymid[order]; xa = xa[order]; xb = xb[order]
    wpx = mask_rot.shape[1]
    rows = []
    cy, cxa, cxb = [ys[0]], [xa[0]], [xb[0]]
    for k in range(1, len(ys)):
        if ys[k] - cy[-1] <= 0.5 * spacing_px:
            cy.append(ys[k]); cxa.append(xa[k]); cxb.append(xb[k])
        else:
            yv = float(np.median(cy))
            x0 = int(np.clip(min(cxa), 0, wpx - 1))
            x1 = int(np.clip(max(cxb), 0, wpx - 1))
            if x1 - x0 > spacing_px:
                rows.append((x0, x1, np.full(wpx, yv, dtype=np.float32)))
            cy, cxa, cxb = [ys[k]], [xa[k]], [xb[k]]
    yv = float(np.median(cy))
    x0 = int(np.clip(min(cxa), 0, wpx - 1)); x1 = int(np.clip(max(cxb), 0, wpx - 1))
    if x1 - x0 > spacing_px:
        rows.append((x0, x1, np.full(wpx, yv, dtype=np.float32)))
    rows.sort(key=lambda r: r[2][0])
    # refine spacing estimate from the clustered rows if possible
    if len(rows) > 2:
        yy = np.array([r[2][0] for r in rows])
        spacing_px = float(np.median(np.diff(yy))) or spacing_px
    return Frame(mask_rot, score_rot, M, gsd, spacing_px, rows)


def detect_hough(core, field, ang_tol_deg=10.0):
    """Probabilistic Hough line voting -> clustered straight rows. UNCONSTRAINED:
    the dominant orientation is whichever angle bin wins the length-weighted
    vote, with no independent check on plausibility.

    Detects straight segments, takes their dominant orientation, rotates rows
    horizontal, then clusters the detected segments' y-positions into rows.
    Because it depends on long continuous straight edges, it fragments on
    curved rows and scatters/merges across block boundaries and gaps -- the
    classic Hough failure this experiment is meant to expose.

    On DENSE, REGULARLY spaced rows there is a second, sharper failure mode:
    grid-aliasing. A diagonal that happens to clip through many rows' edges
    at evenly-spaced intervals (an artefact of the rows' own periodicity --
    the same phenomenon as moire patterns / false diagonals in a picket
    fence photo) can accumulate more total edge length than the true row
    direction and win the vote outright, producing lines running in a
    completely wrong direction across the whole field rather than merely
    imprecise ones. This is the unconstrained baseline "as normally used";
    see detect_hough_constrained for the orientation-prior mitigation.
    """
    return _detect_hough_core(core, field, ang_tol_deg=ang_tol_deg,
                              orientation_prior_deg=None)


def detect_hough_constrained(core, field, ang_tol_deg=10.0, prior_window_deg=25.0):
    """Hough line voting, but the dominant-angle vote is restricted to a
    window around an independent orientation prior (the FFT angle) BEFORE
    picking the winner.

    This directly fixes grid-aliasing (see detect_hough's docstring): a
    spurious diagonal can no longer out-vote the true row direction just by
    accumulating more length, because it is excluded from the vote entirely
    if it falls outside the prior window. Hough's OTHER real weaknesses --
    fragmentation on curved rows, scatter/loss at block boundaries and gaps
    -- are untouched, since those come from the straight-line/continuity
    assumption itself, not from the orientation vote.

    Include this alongside plain 'hough' in the comparison table: it isolates
    how much of Hough's failure on this field was catastrophic misorientation
    (fixed here) versus the structural curved/fragmented-field weakness
    (not fixed here, and not fixable by an orientation prior).
    """
    prior = core.dominant_row_angle(field["mask"], valid=field["valid"])
    return _detect_hough_core(core, field, ang_tol_deg=ang_tol_deg,
                              orientation_prior_deg=prior,
                              prior_window_deg=prior_window_deg)


# ------------------------------------------------- proposed strip tracker
def detect_strip(core, field, row_phase="auto"):
    """Your strip-based tracker with track-birth (the proposed method).

    Mirrors the strip branch of analyse(): FFT orientation -> strip tracking
    -> per-column centerline -> global phase choice -> phase correct -> refine.
    Included so the comparison table is generated in one pass, through the
    identical gap scorer used for FFT/Hough.
    """
    angle = core.dominant_row_angle(field["mask"], valid=field["valid"])
    mask_rot, score_rot, M = _rotate_all(core, field, angle)
    gsd = field["gsd"]
    _, spacing_px, _ = core.detect_rows(mask_rot, gsd)
    half_hint = max(1, int(spacing_px * 0.12))

    row_lines = core.track_rows_in_strips(mask_rot, gsd)
    eff_phase, _ = core.choose_global_row_phase(
        score_rot, row_lines, mask_rot.shape[1], spacing_px, half_hint, row_phase)

    rows = []
    W = mask_rot.shape[1]
    for xs_track, ys_track in row_lines:
        y_of_x = core.centerline_for_columns(xs_track, ys_track, W)
        x0, x1 = int(xs_track.min()), int(xs_track.max())
        y_of_x, _ = core.correct_centerline_phase(
            score_rot, y_of_x, x0, x1, spacing_px, half_hint, row_phase=eff_phase)
        y_of_x = core.refine_centerline(
            score_rot, y_of_x, x0, x1, search_half=max(3, int(spacing_px * 0.3)))
        rows.append((x0, x1, y_of_x))
    rows.sort(key=lambda r: r[2][int((r[0] + r[1]) / 2)])
    return Frame(mask_rot, score_rot, M, gsd, spacing_px, rows)


# --------------------------------------------- ABLATION: strip WITHOUT birth
def track_rows_nobirth(mask_rot, gsd, n_strips=14, expected_spacing_m=(0.6, 2.0),
                       max_gap_strips=2):
    """Ablation of track_rows_in_strips: births DISABLED after the seed strip.

    This is a line-for-line copy of core.track_rows_in_strips EXCEPT that
    unmatched peaks spawn a new track ONLY in the first strip that has any
    peaks (the seed). After that, unmatched peaks are dropped -- i.e. the
    classic 'seed once and grow outward' tracker the docstring warns about.
    Every other parameter (strip width, peak finding, tolerance, skip
    tolerance) is identical, so any metric difference vs the full method is
    attributable to track-birth alone.

    NOTE: if you change the core tracker, re-sync this copy so the ablation
    stays a true one-variable comparison.
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

    active_tracks, histories, next_id = {}, {}, 0
    seeded = False   # <-- the only behavioural change vs the full tracker

    for s in range(n):
        peaks = strip_peak_list[s]
        matched_peak_idx = set()
        if len(peaks) > 1:
            tol = max(6, int(0.5 * np.median(np.diff(np.sort(peaks)))))
        else:
            tol = global_tol

        for tid in list(active_tracks.keys()):
            info = active_tracks[tid]
            if s - info["s"] > max_gap_strips:
                del active_tracks[tid]
                continue
            if len(peaks) == 0:
                continue
            d = np.abs(peaks - info["y"])
            j = int(np.argmin(d))
            if d[j] <= tol and j not in matched_peak_idx:
                matched_peak_idx.add(j)
                active_tracks[tid] = {"y": int(peaks[j]), "s": s}
                histories[tid].append((s, int(peaks[j])))

        # births ONLY at the seed strip; afterwards unmatched peaks are dropped
        if not seeded and len(peaks) > 0:
            for j, y in enumerate(peaks):
                if j not in matched_peak_idx:
                    active_tracks[next_id] = {"y": int(y), "s": s}
                    histories[next_id] = [(s, int(y))]
                    next_id += 1
            seeded = True

    row_lines = []
    min_strips_present = max(2, n // 5)
    for tid, pts in histories.items():
        if len(pts) < min_strips_present:
            continue
        xs = np.array([centers[si] for si, _ in pts])
        ys = np.array([y for _, y in pts], dtype=np.float32)
        row_lines.append((xs, ys))
    row_lines.sort(key=lambda r: r[1].mean())
    return row_lines


def detect_strip_nobirth(core, field, row_phase="auto"):
    """Proposed method with track-birth turned off (the ablation detector).

    Identical to detect_strip() except it calls track_rows_nobirth() for the
    tracking step. Run alongside 'strip' to quantify what birth buys you --
    especially on multi-block fields, where the un-seeded block's rows should
    vanish here but survive under the full method.
    """
    angle = core.dominant_row_angle(field["mask"], valid=field["valid"])
    mask_rot, score_rot, M = _rotate_all(core, field, angle)
    gsd = field["gsd"]
    _, spacing_px, _ = core.detect_rows(mask_rot, gsd)
    half_hint = max(1, int(spacing_px * 0.12))

    row_lines = track_rows_nobirth(mask_rot, gsd)
    eff_phase, _ = core.choose_global_row_phase(
        score_rot, row_lines, mask_rot.shape[1], spacing_px, half_hint, row_phase)

    rows = []
    W = mask_rot.shape[1]
    for xs_track, ys_track in row_lines:
        y_of_x = core.centerline_for_columns(xs_track, ys_track, W)
        x0, x1 = int(xs_track.min()), int(xs_track.max())
        y_of_x, _ = core.correct_centerline_phase(
            score_rot, y_of_x, x0, x1, spacing_px, half_hint, row_phase=eff_phase)
        y_of_x = core.refine_centerline(
            score_rot, y_of_x, x0, x1, search_half=max(3, int(spacing_px * 0.3)))
        rows.append((x0, x1, y_of_x))
    rows.sort(key=lambda r: r[2][int((r[0] + r[1]) / 2)])
    return Frame(mask_rot, score_rot, M, gsd, spacing_px, rows)


# ------------------------------------------------------------- shared scorer
def measure(core, field, frame, method, min_gap_m, row_vertex_step=15):
    """Run the SAME gap scorer on whatever rows a detector produced.

    Returns (summary, row_stats, gap_features, row_features). gap_features
    and row_features are in world coords if the input was a GeoTIFF, else
    pixel coords -- same convention as analyse()'s *_gaps.geojson /
    *_rows.geojson.

    row_features vectorizes each detected row's FULL centerline (straight
    for fft/hough, curved for strip/strip_nb) as one LineString, sampled
    every `row_vertex_step` columns -- this is the row-DETECTION output
    itself (as opposed to gap_features, which is the downstream gap
    measurement), so you can inspect/compare how each method actually
    traced the rows, independent of the gap scoring.
    """
    gsd = frame.gsd
    mask_rot = frame.mask_rot
    half_band = max(2, int(frame.spacing_px * 0.35))
    Minv = cv2.invertAffineTransform(frame.M)
    transform = field["transform"]

    def to_orig(x, y):
        return Minv @ np.array([x, y, 1.0])

    def to_world(x, y):
        p = to_orig(x, y)
        if transform is not None:
            wx, wy = transform * (p[0], p[1])
            return float(wx), float(wy)
        return float(p[0]), float(p[1])

    row_stats, gap_feats, row_feats = [], [], []
    total_gap = total_row = 0.0
    n_gap_segments = 0
    for ri, (x0, x1, y_of_x) in enumerate(frame.rows, 1):
        gaps, (xs, xe) = core.gaps_along_centerline(
            mask_rot, y_of_x, x0, x1, half_band, gsd, min_gap_m)
        row_len_m = (xe - xs) * gsd
        gap_len_m = sum((b - a) for a, b in gaps) * gsd
        total_row += row_len_m
        total_gap += gap_len_m
        n_gap_segments += len(gaps)
        row_stats.append({
            "row": ri, "row_length_m": round(row_len_m, 2),
            "n_gaps": len(gaps), "gap_length_m": round(gap_len_m, 2),
            "gap_pct": round(100 * gap_len_m / row_len_m, 1) if row_len_m else 0,
        })
        for a, b in gaps:
            wx1, wy1 = to_world(a, y_of_x[a])
            wx2, wy2 = to_world(b, y_of_x[b])
            gap_feats.append({
                "type": "Feature",
                "properties": {"row": ri, "gap_length_m": round((b - a) * gsd, 2),
                               "method": method},
                "geometry": {"type": "LineString",
                             "coordinates": [[wx1, wy1], [wx2, wy2]]},
            })

        # -- row centerline itself (the detection output, not the gaps) --
        coords = []
        for x in range(int(xs), int(xe) + 1, row_vertex_step):
            wx, wy = to_world(x, y_of_x[x])
            coords.append([wx, wy])
        if int(xe) not in range(int(xs), int(xe) + 1, row_vertex_step):
            wx, wy = to_world(int(xe), y_of_x[int(xe)])
            coords.append([wx, wy])  # keep the true endpoint
        if len(coords) >= 2:
            row_feats.append({
                "type": "Feature",
                "properties": {"row": ri, "row_length_m": round(row_len_m, 2),
                               "n_gaps": len(gaps), "method": method},
                "geometry": {"type": "LineString", "coordinates": coords},
            })

    summary = {
        "method": method,
        "n_rows": len(frame.rows),
        "median_row_spacing_m": round(frame.spacing_px * gsd, 2),
        "total_row_length_m": round(total_row, 1),
        "total_gap_length_m": round(total_gap, 1),
        "overall_gap_pct": round(100 * total_gap / total_row, 1) if total_row else 0,
        "n_gap_segments": n_gap_segments,
    }
    return summary, row_stats, gap_feats, row_feats


def write_overlay(core, field, frame, out_png):
    """Draw rows (yellow) + gaps (red) on the original ortho for eyeballing."""
    rgb = field["rgb"]
    overlay = rgb.copy()
    Minv = cv2.invertAffineTransform(frame.M)
    gsd = frame.gsd
    half_band = max(2, int(frame.spacing_px * 0.35))
    thick = max(2, int(frame.spacing_px * 0.3))

    def to_orig(x, y):
        return Minv @ np.array([x, y, 1.0])

    for (x0, x1, y_of_x) in frame.rows:
        gaps, (xs, xe) = core.gaps_along_centerline(
            frame.mask_rot, y_of_x, x0, x1, half_band, gsd, 0.5)
        pts = [np.int32(to_orig(x, y_of_x[x])) for x in range(xs, xe + 1, 15)]
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.array(pts)], False, (255, 235, 60), 1)
        for a, b in gaps:
            p1 = np.int32(to_orig(a, y_of_x[a])); p2 = np.int32(to_orig(b, y_of_x[b]))
            cv2.line(overlay, tuple(p1), tuple(p2), (255, 40, 40), thick)
    blended = cv2.addWeighted(rgb, 0.45, overlay, 0.55, 0)
    cv2.imwrite(out_png, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------- run helpers
DETECTORS = {"fft": detect_fft, "hough": detect_hough,
             "hough_c": detect_hough_constrained,
             "strip": detect_strip, "strip_nb": detect_strip_nobirth}


def run_one(core, path, out_prefix, gsd, min_gap_m, dsm_path, min_height_m,
            methods, write_png=True):
    field = load_field(core, path, gsd_cli=gsd, dsm_path=dsm_path,
                       min_height_m=min_height_m)
    print(f"[field] {os.path.basename(path)}  {field['rgb'].shape[1]}x"
          f"{field['rgb'].shape[0]} px  gsd={field['gsd']:.4f} m/px")

    # independent orientation reference for the Hough misorientation diagnostic
    # (grid-aliasing check) -- computed once, regardless of which methods run
    fft_ref_angle = core.dominant_row_angle(field["mask"], valid=field["valid"])

    results = []
    for m in methods:
        try:
            frame = DETECTORS[m](core, field)
            summary, row_stats, gap_feats, row_feats = measure(
                core, field, frame, m, min_gap_m)

            # -- Hough misorientation diagnostic: how far did Hough's chosen
            # rotation angle deviate from the independent FFT estimate?
            # A large deviation on 'hough' (uncorrected) with a small one on
            # 'hough_c' (same field, same code, orientation-prior applied)
            # is the direct, quantified evidence for the grid-aliasing
            # failure -- rather than an unexplained near-zero P/R.
            if m in ("hough", "hough_c"):
                hough_angle = float(np.degrees(np.arctan2(frame.M[1, 0], frame.M[0, 0])))
                dev = abs(((hough_angle - fft_ref_angle + 90) % 180) - 90)
                summary["angle_dev_from_fft_deg"] = round(dev, 1)
                summary["likely_misoriented"] = bool(dev > 15.0)

            if write_png:
                write_overlay(core, field, frame, f"{out_prefix}_{m}_overlay.png")

            crs_block = ({"type": "name", "properties": {"name": str(field["crs"])}}
                        if field["crs"] is not None else None)

            gj = {"type": "FeatureCollection", "features": gap_feats}
            if crs_block:
                gj["crs"] = crs_block
            with open(f"{out_prefix}_{m}_gaps.geojson", "w") as f:
                json.dump(gj, f)

            # -- row-detection output: geometry (one LineString per row) --
            rj = {"type": "FeatureCollection", "features": row_feats}
            if crs_block:
                rj["crs"] = crs_block
            with open(f"{out_prefix}_{m}_rows.geojson", "w") as f:
                json.dump(rj, f)

            # -- row-detection output: per-row stats table --
            with open(f"{out_prefix}_{m}_row_stats.csv", "w", newline="") as f:
                if row_stats:
                    w = csv.DictWriter(f, fieldnames=row_stats[0].keys())
                    w.writeheader(); w.writerows(row_stats)
                else:
                    f.write("row,row_length_m,n_gaps,gap_length_m,gap_pct\n")

            dev_note = (f"  angle_dev={summary['angle_dev_from_fft_deg']:5.1f}deg"
                       + (" (MISORIENTED)" if summary["likely_misoriented"] else "")
                       if "angle_dev_from_fft_deg" in summary else "")
            print(f"   [{m:5s}] rows={summary['n_rows']:3d}  "
                  f"gap%={summary['overall_gap_pct']:5.1f}  "
                  f"segs={summary['n_gap_segments']:4d}  "
                  f"spacing={summary['median_row_spacing_m']}m{dev_note}")
            results.append(summary)
        except Exception as e:
            print(f"   [{m:5s}] *** ERROR: {e}")
            results.append({"method": m, "error": str(e)})
    return results


def run_batch(core, input_dir, out_dir, gsd, min_gap_m, dsm_arg, min_height_m,
              patterns, methods):
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat.strip())))
    files = sorted(set(files))
    print(f"[batch] {len(files)} field(s) x {len(methods)} method(s)")
    if not files:
        print("[batch] no files matched --pattern"); return

    dsm_dir = dsm_arg if (dsm_arg and os.path.isdir(dsm_arg)) else None
    dsm_single = dsm_arg if (dsm_arg and not os.path.isdir(dsm_arg)) else None

    rows_out = []
    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        prefix = os.path.join(out_dir, stem)
        print(f"\n[batch] ({i}/{len(files)}) {stem}")
        dsm_path = None
        if dsm_dir:
            c = glob.glob(os.path.join(dsm_dir, stem + ".*"))
            dsm_path = c[0] if c else None
        elif dsm_single:
            dsm_path = dsm_single
        res = run_one(core, path, prefix, gsd, min_gap_m, dsm_path,
                      min_height_m, methods)
        for r in res:
            r = dict(r); r["field"] = stem
            rows_out.append(r)

    keys = []
    for r in rows_out:
        for k in ("field", "method"):
            if k not in keys:
                keys.append(k)
        for k in r:
            if k not in keys:
                keys.append(k)
    agg = os.path.join(out_dir, "comparison.csv")
    with open(agg, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows_out)
    print(f"\n[batch] DONE -> {agg}")
    print("[batch] Pivot on (field, method) to build the paper's baseline "
          "table. For precision/recall, score each *_gaps.geojson against "
          "your manual annotations with the 'score' subcommand.")

    # -- Hough misorientation roll-up: the "N of M fields" number for the
    # paper's grid-aliasing sentence. Reported per hough variant present.
    for hm in ("hough", "hough_c"):
        rows_hm = [r for r in rows_out if r.get("method") == hm
                  and "angle_dev_from_fft_deg" in r]
        if not rows_hm:
            continue
        n_mis = sum(1 for r in rows_hm if r.get("likely_misoriented"))
        devs = [r["angle_dev_from_fft_deg"] for r in rows_hm]
        print(f"[batch] {hm}: misoriented on {n_mis}/{len(rows_hm)} fields "
              f"(>15deg from FFT angle)  "
              f"median dev={sorted(devs)[len(devs)//2]:.1f}deg  "
              f"max dev={max(devs):.1f}deg")


# ------------------------------------------------------- precision / recall
def _load_gap_segments(geojson_path):
    with open(geojson_path) as f:
        gj = json.load(f)
    segs = []
    for ft in gj.get("features", []):
        g = ft.get("geometry", {})
        if g.get("type") == "LineString" and len(g["coordinates"]) >= 2:
            (x1, y1), (x2, y2) = g["coordinates"][0], g["coordinates"][-1]
            segs.append(((x1, y1), (x2, y2),
                         ft.get("properties", {}).get("row")))
    return segs


def _seg_midpoint(s):
    (x1, y1), (x2, y2), _ = s
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _seg_len(s):
    (x1, y1), (x2, y2), _ = s
    return float(np.hypot(x2 - x1, y2 - y1))


def score_against_truth(detected_geojson, truth_geojson,
                        match_dist_m=1.0, len_ratio=0.5):
    """Precision / recall / F1 of detected gaps vs manual ground truth.

    EXPECTED ANNOTATION FORMAT (tune to your schema):
      truth = a GeoJSON FeatureCollection of LineString features, one per
      manually digitised gap, in the SAME CRS as the detected gaps (i.e. the
      georeferenced outputs from this harness). A "row" property is optional.

    MATCHING RULE (greedy, one-to-one):
      A detected gap matches a truth gap if their midpoints are within
      match_dist_m AND their lengths agree within len_ratio (|Ld-Lt| <=
      len_ratio*max(Ld,Lt)). This is a pragmatic geometric match; if your
      annotations carry row ids and along-row extents, a per-row interval-IoU
      match is stricter and better -- adjust here before quoting numbers.

    Reports TP, FP, FN and P/R/F1. NOTE this is a coarse matcher meant to get
    you a defensible first number; confirm the rule fits your annotations
    before it goes in the paper.
    """
    det = _load_gap_segments(detected_geojson)
    tru = _load_gap_segments(truth_geojson)
    used = set()
    tp = 0
    for d in det:
        dm, dl = _seg_midpoint(d), _seg_len(d)
        best, bestd = -1, 1e18
        for j, t in enumerate(tru):
            if j in used:
                continue
            tm, tl = _seg_midpoint(t), _seg_len(t)
            dist = np.hypot(dm[0] - tm[0], dm[1] - tm[1])
            if dist <= match_dist_m and abs(dl - tl) <= len_ratio * max(dl, tl, 1e-6):
                if dist < bestd:
                    bestd, best = dist, j
        if best >= 0:
            used.add(best); tp += 1
    fp = len(det) - tp
    fn = len(tru) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(json.dumps({
        "detected": len(det), "truth": len(tru),
        "TP": tp, "FP": fp, "FN": fn,
        "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(f1, 3),
        "match_dist_m": match_dist_m, "len_ratio": len_ratio,
    }, indent=2))
    return {"precision": prec, "recall": rec, "f1": f1, "TP": tp, "FP": fp, "FN": fn}


# -------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--core", default=None, help="path to your pipeline module")
    common.add_argument("--gsd", type=float, default=0.05, help="m/px fallback")
    common.add_argument("--min-gap", type=float, default=0.5, help="min gap length (m)")
    common.add_argument("--dsm", default=None, help="optional DSM (file or folder)")
    common.add_argument("--min-height", type=float, default=0.25)
    common.add_argument("--methods", default="fft,hough,hough_c,strip",
                        help="comma list from: fft,hough,hough_c,strip,strip_nb "
                             "(hough_c = Hough with an FFT orientation prior, "
                             "fixing grid-aliasing misorientation on dense "
                             "regular rows; strip_nb = strip tracker with "
                             "track-birth disabled, for the ablation study)")

    p_run = sub.add_parser("run", parents=[common], help="one field, all methods")
    p_run.add_argument("input")
    p_run.add_argument("--out", default="baseline_out")

    p_bat = sub.add_parser("batch", parents=[common], help="folder of fields")
    p_bat.add_argument("input")
    p_bat.add_argument("--out-dir", default="comparison_out")
    p_bat.add_argument("--pattern", default="*.tif,*.tiff")

    p_sc = sub.add_parser("score", help="precision/recall vs annotations")
    p_sc.add_argument("--detected", required=True)
    p_sc.add_argument("--truth", required=True)
    p_sc.add_argument("--match-dist", type=float, default=1.0)
    p_sc.add_argument("--len-ratio", type=float, default=0.5)

    args = ap.parse_args()

    if args.cmd == "score":
        score_against_truth(args.detected, args.truth,
                            match_dist_m=args.match_dist, len_ratio=args.len_ratio)
        return

    core = load_core(args.core)
    methods = [m.strip() for m in args.methods.split(",") if m.strip() in DETECTORS]
    if not methods:
        sys.exit("no valid --methods (choose from fft,hough,strip)")

    if args.cmd == "run":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        run_one(core, args.input, args.out, args.gsd, args.min_gap,
                args.dsm, args.min_height, methods)
    elif args.cmd == "batch":
        run_batch(core, args.input, args.out_dir, args.gsd, args.min_gap,
                  args.dsm, args.min_height, args.pattern.split(","), methods)


if __name__ == "__main__":
    main()
