import os

import numpy as np
import rasterio
import geopandas as gpd

from shapely.geometry import LineString

from skan import Skeleton


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_SKELETON = (
    r"D:\Bapan\auto_digi\track\postprocess_rail\rail_skeleton_v3.tif"
)

OUTPUT_DIR = (
    r"D:\Bapan\auto_digi\track\final_output"
)

OUTPUT_GPKG = os.path.join(
    OUTPUT_DIR,
    "rail_lines.gpkg"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Minimum vector segment length in metres.
#
# Very short skeleton branches are usually noise.
#
# We deliberately keep this relatively small for now.
# We will inspect the result before using a stronger filter.
# ============================================================

MIN_LENGTH_M = 5.0


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SKELETON → GEOSPATIAL RAIL LINES")
print("=" * 70)

print()
print("Input skeleton:")
print(INPUT_SKELETON)

print()
print("Output:")
print(OUTPUT_GPKG)


# ============================================================
# READ SKELETON
# ============================================================

with rasterio.open(
    INPUT_SKELETON
) as src:

    skeleton = src.read(1)

    transform = src.transform

    crs = src.crs

    width = src.width

    height = src.height


print()
print("Raster size:")
print(
    f"{width} x {height}"
)

print(
    "CRS:",
    crs
)

print(
    "Skeleton pixels:",
    int(
        np.count_nonzero(
            skeleton
        )
    )
)


# ============================================================
# BINARY SKELETON
# ============================================================

skeleton_binary = (
    skeleton > 0
)


# ============================================================
# CREATE SKAN SKELETON
# ============================================================

print()
print(
    "Building skeleton graph..."
)

# Pixel spacing is approximately 3 cm.
#
# We obtain it directly from the GeoTIFF transform.
# ============================================================

pixel_size_x = abs(
    transform.a
)

pixel_size_y = abs(
    transform.e
)

print(
    "Pixel size X:",
    pixel_size_x,
    "m"
)

print(
    "Pixel size Y:",
    pixel_size_y,
    "m"
)


# skan accepts spacing so path lengths
# can be calculated in real-world units.
# ============================================================

sk = Skeleton(
    skeleton_binary,
    spacing=(
        pixel_size_y,
        pixel_size_x
    )
)


print(
    "Skeleton paths:",
    sk.n_paths
)


# ============================================================
# EXTRACT PATHS
# ============================================================

records = []

accepted = 0

rejected = 0


for path_id in range(
    sk.n_paths
):

    # --------------------------------------------------------
    # Path length
    # --------------------------------------------------------

    length_m = float(
        sk.path_lengths()[
            path_id
        ]
    )

    # --------------------------------------------------------
    # Remove very short branches
    # --------------------------------------------------------

    if length_m < MIN_LENGTH_M:

        rejected += 1

        continue

    # --------------------------------------------------------
    # Get ordered pixel coordinates
    #
    # path_coordinates returns:
    #
    # [[row, col],
    #  [row, col],
    #  ...]
    # --------------------------------------------------------

    coordinates = sk.path_coordinates(
        path_id
    )

    if coordinates is None:

        rejected += 1

        continue

    if len(coordinates) < 2:

        rejected += 1

        continue


    # ========================================================
    # PIXEL → MAP COORDINATES
    # ========================================================

    line_coordinates = []

    for row, col in coordinates:

        # rasterio transform
        # converts pixel coordinates
        # to map coordinates.
        #
        # offset='center' gives the pixel center.
        # ----------------------------------------------------

        x, y = rasterio.transform.xy(
            transform,
            int(row),
            int(col),
            offset="center"
        )

        line_coordinates.append(
            (x, y)
        )


    # ========================================================
    # CREATE LINESTRING
    # ========================================================

    line = LineString(
        line_coordinates
    )


    # --------------------------------------------------------
    # Remove invalid/zero-length geometry
    # --------------------------------------------------------

    if line.is_empty:

        rejected += 1

        continue

    if not line.is_valid:

        line = line.buffer(
            0
        )

        if line.is_empty:

            rejected += 1

            continue


    # ========================================================
    # SAVE RECORD
    # ========================================================

    records.append(
        {
            "segment_id": accepted + 1,
            "path_id": path_id,
            "length_m": float(
                line.length
            ),
            "n_points": len(
                coordinates
            ),
            "geometry": line
        }
    )

    accepted += 1


# ============================================================
# CREATE GEODATAFRAME
# ============================================================

print()
print(
    "Accepted segments:",
    accepted
)

print(
    "Rejected short segments:",
    rejected
)


if len(records) == 0:

    raise RuntimeError(
        "No valid rail segments were created. "
        "Try reducing MIN_LENGTH_M."
    )


gdf = gpd.GeoDataFrame(
    records,
    geometry="geometry",
    crs=crs
)


# ============================================================
# SORT BY LENGTH
# ============================================================

gdf = gdf.sort_values(
    "length_m",
    ascending=False
).reset_index(
    drop=True
)

# Reassign segment IDs
gdf["segment_id"] = (
    np.arange(
        1,
        len(gdf) + 1
    )
)


# ============================================================
# SAVE GEOPACKAGE
# ============================================================

print()
print(
    "Saving GeoPackage..."
)

gdf.to_file(
    OUTPUT_GPKG,
    layer="rail_segments",
    driver="GPKG"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("VECTOR EXTRACTION COMPLETED")
print("=" * 70)

print()
print(
    "Total rail segments:",
    len(gdf)
)

print(
    "Longest segment:",
    f"{gdf['length_m'].max():.2f} m"
)

print(
    "Shortest saved segment:",
    f"{gdf['length_m'].min():.2f} m"
)

print()
print(
    "GeoPackage:"
)

print(
    OUTPUT_GPKG
)

print()
print(
    "Layer:"
)

print(
    "rail_segments"
)

print()
print(
    "Open the GeoPackage in QGIS and inspect the"
)

print(
    "rail_segments layer before continuing."
)
