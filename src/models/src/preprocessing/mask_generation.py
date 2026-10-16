import os
import glob
import numpy as np
import geopandas as gpd
import rasterio

from rasterio.features import rasterize


# ============================================================
# PATHS
# ============================================================

TILES_DIR = r"D:\Bapan\auto_digi\track\tiles"

ANNOTATION_GPKG = r"D:\Bapan\auto_digi\track\annotation\Rail_Annotation.gpkg"

OUTPUT_MASK_DIR = r"D:\Bapan\auto_digi\track\mask_generation"


# ============================================================
# FIELD NAMES
# ============================================================

# Field containing tile name
TILE_FIELD = "tile_name"

# Field containing class ID
CLASS_FIELD = "class_id"


# ============================================================
# CLASS VALUE
# ============================================================

# Background = 0
# Rail      = 1

RAIL_CLASS = 1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_MASK_DIR,
    exist_ok=True
)


# ============================================================
# READ ANNOTATION
# ============================================================

print("\n==============================================")
print("READING ANNOTATION")
print("==============================================")

gdf = gpd.read_file(
    ANNOTATION_GPKG
)

print(f"Total annotation features: {len(gdf)}")

print("\nAvailable fields:")
print(gdf.columns.tolist())


# ============================================================
# CHECK REQUIRED FIELDS
# ============================================================

if TILE_FIELD not in gdf.columns:

    raise ValueError(
        f"\nERROR: Field '{TILE_FIELD}' was not found.\n"
        f"Available fields: {gdf.columns.tolist()}"
    )


if CLASS_FIELD not in gdf.columns:

    raise ValueError(
        f"\nERROR: Field '{CLASS_FIELD}' was not found.\n"
        f"Available fields: {gdf.columns.tolist()}"
    )


# ============================================================
# CHECK CRS
# ============================================================

print("\nAnnotation CRS:")
print(gdf.crs)


# ============================================================
# FIND ALL TILE FILES
# ============================================================

tile_files = glob.glob(
    os.path.join(
        TILES_DIR,
        "*.tif"
    )
)

print(f"\nTotal TIFF tiles found: {len(tile_files)}")


# ============================================================
# PROCESS EACH TILE THAT HAS ANNOTATION
# ============================================================

processed = 0
skipped = 0


for tile_path in tile_files:

    tile_filename = os.path.basename(
        tile_path
    )

    tile_name = os.path.splitext(
        tile_filename
    )[0]

    print("\n----------------------------------------------")
    print(f"Processing: {tile_name}")
    print("----------------------------------------------")


    # --------------------------------------------------------
    # Find annotations belonging to this tile
    # --------------------------------------------------------

    tile_gdf = gdf[
        gdf[TILE_FIELD].astype(str).str.strip()
        == tile_name
    ].copy()


    # --------------------------------------------------------
    # If no annotation exists
    # --------------------------------------------------------

    if tile_gdf.empty:

        print("No annotation found → SKIPPED")

        skipped += 1

        continue


    print(
        f"Annotations found: {len(tile_gdf)}"
    )


    # --------------------------------------------------------
    # Open tile
    # --------------------------------------------------------

    with rasterio.open(tile_path) as src:

        width = src.width
        height = src.height

        transform = src.transform

        raster_crs = src.crs


        print(
            f"Tile size: {width} × {height}"
        )

        print(
            f"Tile CRS: {raster_crs}"
        )


        # ----------------------------------------------------
        # Reproject annotation if CRS differs
        # ----------------------------------------------------

        if tile_gdf.crs != raster_crs:

            print(
                "CRS differs → Reprojecting annotations..."
            )

            tile_gdf = tile_gdf.to_crs(
                raster_crs
            )


        # ----------------------------------------------------
        # Select Rail polygons
        # ----------------------------------------------------

        rail_gdf = tile_gdf[
            tile_gdf[CLASS_FIELD].astype(int)
            == RAIL_CLASS
        ].copy()


        if rail_gdf.empty:

            print(
                "No Rail polygons found → SKIPPED"
            )

            skipped += 1

            continue


        # ----------------------------------------------------
        # Prepare geometries
        # ----------------------------------------------------

        shapes = [
            (
                geom,
                RAIL_CLASS
            )
            for geom in rail_gdf.geometry
            if geom is not None
            and not geom.is_empty
        ]


        # ----------------------------------------------------
        # Rasterize polygons
        # ----------------------------------------------------

        mask = rasterize(

            shapes=shapes,

            out_shape=(
                height,
                width
            ),

            transform=transform,

            fill=0,

            dtype=np.uint8
        )


        # ----------------------------------------------------
        # Save mask as PNG
        # ----------------------------------------------------

        output_name = (
            f"{tile_name}_mask.png"
        )

        output_path = os.path.join(
            OUTPUT_MASK_DIR,
            output_name
        )


        # Use PIL to save PNG
        from PIL import Image

        Image.fromarray(
            mask,
            mode="L"
        ).save(
            output_path
        )


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        background_pixels = np.sum(
            mask == 0
        )

        rail_pixels = np.sum(
            mask == 1
        )

        total_pixels = mask.size

        rail_percentage = (
            rail_pixels /
            total_pixels
        ) * 100


        print(
            f"Mask saved: {output_name}"
        )

        print(
            f"Background pixels: {background_pixels}"
        )

        print(
            f"Rail pixels:       {rail_pixels}"
        )

        print(
            f"Rail coverage:     {rail_percentage:.2f}%"
        )


        processed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n==============================================")
print("MASK GENERATION COMPLETED")
print("==============================================")

print(
    f"Tiles processed : {processed}"
)

print(
    f"Tiles skipped   : {skipped}"
)

print(
    f"Output folder   : {OUTPUT_MASK_DIR}"
)

print("\nMask values:")
print("0 = Background")
print("1 = Rail")

print("\nNext step:")
print("Visually inspect several image/mask pairs.")
