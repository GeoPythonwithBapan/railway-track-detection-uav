"""
Generate binary rail masks from GeoPackage polygon annotations.

Example:
    python mask_generation.py \
        --tiles-dir "D:/data/tiles" \
        --annotations "D:/data/Rail_Annotation.gpkg" \
        --output-dir "D:/data/masks"

Expected annotation fields by default:
    tile_name : tile filename/stem linked to the annotation
    class_id  : class value (rail = 1)

Output:
    One PNG mask per annotated tile. Background = 0, rail = 1.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize


def normalize_tile_name(value) -> str:
    """Normalize a tile reference to its filename stem."""
    return Path(str(value)).stem.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rasterize railway rail polygons into binary PNG masks."
    )
    parser.add_argument("--tiles-dir", required=True, help="Folder containing raster tiles; subfolders are searched too.")
    parser.add_argument("--annotations", required=True, help="GeoPackage containing polygon annotations.")
    parser.add_argument("--output-dir", required=True, help="Folder for generated PNG masks.")
    parser.add_argument("--tile-field", default="tile_name", help="Annotation field identifying the tile (default: tile_name).")
    parser.add_argument("--class-field", default="class_id", help="Annotation class field (default: class_id).")
    parser.add_argument("--rail-class", type=int, default=1, help="Class ID for rail polygons (default: 1).")
    parser.add_argument("--all-touched", action="store_true", help="Include every pixel touched by a polygon boundary.")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir).expanduser().resolve()
    annotation_path = Path(args.annotations).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not tiles_dir.is_dir():
        raise SystemExit(f"Tile directory does not exist: {tiles_dir}")
    if not annotation_path.is_file():
        raise SystemExit(f"Annotation GeoPackage does not exist: {annotation_path}")

    try:
        annotations = gpd.read_file(annotation_path)
    except Exception as exc:
        raise SystemExit(f"Could not read annotation file: {exc}") from exc

    for field in (args.tile_field, args.class_field):
        if field not in annotations.columns:
            raise SystemExit(
                f"Required field '{field}' was not found. "
                f"Available fields: {list(annotations.columns)}"
            )

    if annotations.empty:
        raise SystemExit("The annotation layer contains no features.")

    annotations = annotations[
        annotations[args.class_field].astype(str) == str(args.rail_class)
    ].copy()
    annotations = annotations[
        annotations.geometry.notna() & ~annotations.geometry.is_empty
    ].copy()

    if annotations.empty:
        raise SystemExit(
            f"No valid annotation geometries found for class {args.rail_class}."
        )

    annotations["_tile_key"] = annotations[args.tile_field].map(normalize_tile_name)
    annotations_by_tile = {
        tile_key: group
        for tile_key, group in annotations.groupby("_tile_key")
    }

    tile_paths = sorted(
        p for p in tiles_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not tile_paths:
        raise SystemExit(f"No .tif or .tiff tiles found under: {tiles_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    total_rail_pixels = 0

    print(f"Raster tiles found: {len(tile_paths)}")
    print(f"Rail annotation features: {len(annotations)}")
    print(f"Rail class ID: {args.rail_class}")

    for tile_path in tile_paths:
        tile_key = tile_path.stem
        tile_annotations = annotations_by_tile.get(tile_key)

        # A tile without annotations still gets an all-background mask.
        with rasterio.open(tile_path) as src:
            height, width = src.height, src.width
            tile_crs = src.crs
            transform = src.transform

            if tile_annotations is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            else:
                shapes = tile_annotations
                if tile_crs is not None and shapes.crs is not None and shapes.crs != tile_crs:
                    shapes = shapes.to_crs(tile_crs)
                elif shapes.crs is None and tile_crs is not None:
                    raise ValueError(
                        f"Annotations have no CRS but tile '{tile_path.name}' "
                        f"uses {tile_crs}. Assign the correct annotation CRS first."
                    )

                geometry_values = [
                    (geom, 1)
                    for geom in shapes.geometry
                    if geom is not None and not geom.is_empty
                ]
                if geometry_values:
                    mask = rasterize(
                        geometry_values,
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        default_value=1,
                        all_touched=args.all_touched,
                        dtype=np.uint8,
                    )
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)

        mask_path = output_dir / f"{tile_key}.png"
        Image.fromarray(mask, mode="L").save(mask_path)

        rail_pixels = int(np.count_nonzero(mask))
        total_rail_pixels += rail_pixels
        processed += 1

        if tile_annotations is None:
            skipped += 1
        else:
            print(f"{tile_key}: annotations={len(tile_annotations)}, rail pixels={rail_pixels}")

    print("\nMask generation completed.")
    print(f"Tiles processed: {processed}")
    print(f"Tiles without matching rail annotations (all-background masks): {skipped}")
    print(f"Total rail pixels: {total_rail_pixels}")
    print(f"Output masks: {output_dir}")


if __name__ == "__main__":
    main()
