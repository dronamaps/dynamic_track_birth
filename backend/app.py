import json
import os
import re
import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from rio_tiler.io import COGReader
from rio_tiler.utils import render

from backend.celery_app import celery_app
from backend.tasks import DATA_ROOT, analyse_orthomosaic


MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))
CHUNK_SIZE = 1024 * 1024
SAFE_JOB_ID = re.compile(r"^[a-f0-9-]{36}$")
METHOD_KEYS = {"fft", "strip_nb", "strip"}

app = FastAPI(title="Crop Row Gap Analyzer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _job_dir(job_id: str) -> Path:
    if not SAFE_JOB_ID.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    path = DATA_ROOT / job_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Unknown job")
    return path


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "methods": sorted(METHOD_KEYS)}


@app.post("/api/jobs", status_code=202)
async def create_job(
    orthomosaic: UploadFile = File(...),
    min_gap_m: float = Form(..., gt=0.0, le=20.0),
) -> dict:
    filename = Path(orthomosaic.filename or "orthomosaic.tif").name
    if Path(filename).suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(status_code=415, detail="Upload a .tif or .tiff file")

    job_id = str(uuid.uuid4())
    job_dir = DATA_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_name = f"input{Path(filename).suffix.lower()}"
    destination = job_dir / input_name

    size = 0
    try:
        with destination.open("wb") as stream:
            while chunk := await orthomosaic.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="GeoTIFF exceeds the 500 MB upload limit",
                    )
                stream.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        if job_dir.exists():
            job_dir.rmdir()
        raise
    finally:
        await orthomosaic.close()

    analyse_orthomosaic.apply_async(
        args=[job_id, input_name, min_gap_m],
        task_id=job_id,
    )
    return {"job_id": job_id, "status": "queued", "size_bytes": size}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    _job_dir(job_id)
    task = AsyncResult(job_id, app=celery_app)
    state = task.state
    if state == "SUCCESS":
        return {
            "job_id": job_id,
            "status": "succeeded",
            "progress": 100,
            "message": "Analysis complete",
            "result": task.result,
        }
    if state == "FAILURE":
        return {
            "job_id": job_id,
            "status": "failed",
            "progress": 100,
            "message": str(task.result),
        }
    if state == "PROGRESS":
        meta = task.info if isinstance(task.info, dict) else {}
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": meta.get("progress", 30),
            "message": meta.get("message", "Processing orthomosaic"),
        }
    return {
        "job_id": job_id,
        "status": "queued",
        "progress": 5,
        "message": "Waiting for an analysis worker",
    }


def _method_path(job_id: str, method: str, suffix: str) -> Path:
    if method not in METHOD_KEYS:
        raise HTTPException(status_code=404, detail="Unknown analysis method")
    return _job_dir(job_id) / f"{method}_{suffix}"


@app.get("/api/jobs/{job_id}/methods/{method}/rows.geojson")
def rows_geojson(job_id: str, method: str) -> FileResponse:
    path = _method_path(job_id, method, "rows.wgs84.geojson")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Rows are not ready")
    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/jobs/{job_id}/methods/{method}/gaps.geojson")
def gaps_geojson(job_id: str, method: str) -> FileResponse:
    path = _method_path(job_id, method, "gaps.wgs84.geojson")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Gaps are not ready")
    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/jobs/{job_id}/methods/{method}/row-gap-statistics.csv")
def statistics_csv(job_id: str, method: str) -> FileResponse:
    path = _method_path(job_id, method, "row_gap_statistics.csv")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Statistics are not ready")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{job_id}_{method}_row_gap_statistics.csv",
    )


@app.get("/api/jobs/{job_id}/tiles/{z}/{x}/{y}.png")
def raster_tile(job_id: str, z: int, x: int, y: int) -> Response:
    path = _job_dir(job_id) / "orthomosaic.cog.tif"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Orthomosaic tiles are not ready")
    try:
        with COGReader(str(path)) as cog:
            image = cog.tile(x, y, z, tilesize=256)
        content = render(image.data, mask=image.mask, img_format="PNG")
    except Exception as exc:
        raise HTTPException(status_code=404, detail="Tile is outside raster bounds") from exc
    return Response(content=content, media_type="image/png")
