import json
from pathlib import Path
from typing import Any

import rasterio
from pyproj import CRS, Transformer
from rasterio.shutil import copy as rio_copy
from rasterio.warp import transform_bounds


def prepare_cog(source: Path, destination: Path) -> list[list[float]]:
    """Create a tiled COG and return its EPSG:4326 Leaflet bounds."""
    with rasterio.open(source) as dataset:
        if dataset.crs is None:
            raise ValueError("The uploaded GeoTIFF has no CRS/georeferencing.")
        west, south, east, north = transform_bounds(
            dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21
        )

    rio_copy(
        source,
        destination,
        driver="COG",
        compress="DEFLATE",
        blocksize=512,
        overview_resampling="bilinear",
    )
    return [[south, west], [north, east]]


def _map_coordinates(value: Any, transform: Transformer) -> Any:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        x, y = transform.transform(value[0], value[1])
        return [float(x), float(y), *value[2:]]
    if isinstance(value, list):
        return [_map_coordinates(item, transform) for item in value]
    return value


def geojson_to_wgs84(source: Path, destination: Path) -> None:
    """Convert the pipeline's projected GeoJSON to RFC 7946 coordinates."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    crs_name = (
        payload.get("crs", {})
        .get("properties", {})
        .get("name")
    )
    if not crs_name:
        raise ValueError(f"Missing CRS metadata in {source.name}")

    source_crs = CRS.from_user_input(crs_name)
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    for feature in payload.get("features", []):
        geometry = feature.get("geometry")
        if geometry and geometry.get("coordinates") is not None:
            geometry["coordinates"] = _map_coordinates(
                geometry["coordinates"], transformer
            )

    payload.pop("crs", None)
    destination.write_text(json.dumps(payload), encoding="utf-8")
