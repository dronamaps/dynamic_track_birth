import os
import shutil
from pathlib import Path

from celery import current_task

from backend.celery_app import celery_app
from backend.geo import geojson_to_wgs84, prepare_cog


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data/jobs"))
METHOD_LABELS = {
    "fft": "Fourier / FFT",
    "strip_nb": "Strip without birth",
    "strip": "Strip + dynamic birth",
}


def _progress(percent: int, message: str) -> None:
    current_task.update_state(
        state="PROGRESS",
        meta={"progress": percent, "message": message},
    )


@celery_app.task(bind=True, name="backend.tasks.analyse_orthomosaic")
def analyse_orthomosaic(
    self,
    job_id: str,
    input_name: str,
    min_gap_m: float,
) -> dict:
    from backend.analysis import row_detection_baselines as baselines

    job_dir = DATA_ROOT / job_id
    input_path = job_dir / input_name
    output_prefix = job_dir / "analysis"
    cog_path = job_dir / "orthomosaic.cog.tif"

    _progress(12, "Validating GeoTIFF and preparing map tiles")
    bounds = prepare_cog(input_path, cog_path)

    _progress(28, "Segmenting vegetation")
    core_path = Path(__file__).parent / "analysis" / "row_gap_analysis_folder.py"
    core = baselines.load_core(str(core_path))

    _progress(42, "Running Fourier and strip-tracking methods")
    method_keys = list(METHOD_LABELS)
    results = baselines.run_one(
        core=core,
        path=str(input_path),
        out_prefix=str(output_prefix),
        gsd=0.05,
        min_gap_m=min_gap_m,
        dsm_path=None,
        min_height_m=0.25,
        methods=method_keys,
        write_png=False,
    )

    _progress(82, "Preparing web map overlays")
    method_outputs = {}
    for method, summary in zip(method_keys, results, strict=True):
        if summary.get("error"):
            raise RuntimeError(f"{METHOD_LABELS[method]} failed: {summary['error']}")

        rows_source = job_dir / f"analysis_{method}_rows.geojson"
        gaps_source = job_dir / f"analysis_{method}_gaps.geojson"
        rows_web = job_dir / f"{method}_rows.wgs84.geojson"
        gaps_web = job_dir / f"{method}_gaps.wgs84.geojson"
        geojson_to_wgs84(rows_source, rows_web)
        geojson_to_wgs84(gaps_source, gaps_web)

        csv_source = job_dir / f"analysis_{method}_row_stats.csv"
        csv_web = job_dir / f"{method}_row_gap_statistics.csv"
        shutil.copyfile(csv_source, csv_web)

        method_outputs[method] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "summary": summary,
            "rows_path": f"/api/jobs/{job_id}/methods/{method}/rows.geojson",
            "gaps_path": f"/api/jobs/{job_id}/methods/{method}/gaps.geojson",
            "csv_path": (
                f"/api/jobs/{job_id}/methods/{method}/row-gap-statistics.csv"
            ),
        }

    _progress(96, "Finalizing analysis")
    return {
        "default_method": "strip",
        "bounds": bounds,
        "tile_path": f"/api/jobs/{job_id}/tiles/{{z}}/{{x}}/{{y}}.png",
        "methods": method_outputs,
    }
