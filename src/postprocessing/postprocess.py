import os
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from skimage.morphology import skeletonize


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_MASK = (
    r"D:\Bapan\auto_digi\track\prediction\full_rail_mask.tif"
)

OUTPUT_DIR = (
    r"D:\Bapan\auto_digi\track\postprocess_rail"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

MIN_COMPONENT_SIZE = 300

CLOSING_SIZE = 3
OPENING_SIZE = 3

# Tile size
TILE_SIZE = 2048

# Overlap
OVERLAP = 128


# ============================================================
# OUTPUT FILES
# ============================================================

CLEANED_MASK_TIF = os.path.join(
    OUTPUT_DIR,
    "cleaned_rail_mask_v3.tif"
)

SKELETON_TIF = os.path.join(
    OUTPUT_DIR,
    "rail_skeleton_v3.tif"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RAIL POST-PROCESSING V3 - FULLY TILED")
print("=" * 70)

print()
print("Input:")
print(INPUT_MASK)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# OPEN INPUT
# ============================================================

with rasterio.open(INPUT_MASK) as src:

    width = src.width
    height = src.height

    profile = src.profile.copy()

    crs = src.crs
    transform = src.transform

    print()
    print("Raster size:")
    print(f"{width} x {height}")

    print("CRS:")
    print(crs)

    print()
    print("Tile size:")
    print(TILE_SIZE)

    print("Overlap:")
    print(OVERLAP)


# ============================================================
# TILE SETTINGS
# ============================================================

STEP = TILE_SIZE - OVERLAP

x_positions = list(
    range(
        0,
        max(width - TILE_SIZE + 1, 1),
        STEP
    )
)

y_positions = list(
    range(
        0,
        max(height - TILE_SIZE + 1, 1),
        STEP
    )
)


# Make sure final tile reaches image boundary
final_x = width - TILE_SIZE

if final_x > 0:

    if x_positions[-1] != final_x:
        x_positions.append(final_x)


final_y = height - TILE_SIZE

if final_y > 0:

    if y_positions[-1] != final_y:
        y_positions.append(final_y)


total_tiles = (
    len(x_positions) *
    len(y_positions)
)


print()
print("X tiles:", len(x_positions))
print("Y tiles:", len(y_positions))
print("Total tiles:", total_tiles)


# ============================================================
# OUTPUT PROFILE
# ============================================================

output_profile = profile.copy()

output_profile.update(
    {
        "driver": "GTiff",
        "count": 1,
        "dtype": "uint8",
        "nodata": 0,
        "compress": "DEFLATE",
        "zlevel": 6,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "YES"
    }
)

output_profile.pop("photometric", None)
output_profile.pop("interleave", None)


# ============================================================
# CREATE OUTPUT FILES
# ============================================================

print()
print("Creating output GeoTIFFs...")

with rasterio.open(
    CLEANED_MASK_TIF,
    "w",
    **output_profile
) as cleaned_dst:

    with rasterio.open(
        SKELETON_TIF,
        "w",
        **output_profile
    ) as skeleton_dst:

        # ====================================================
        # PROCESS TILES
        # ====================================================

        tile_number = 0

        total_cleaned_pixels = 0
        total_skeleton_pixels = 0

        with rasterio.open(INPUT_MASK) as src:

            for y0 in y_positions:

                for x0 in x_positions:

                    tile_number += 1

                    # ----------------------------------------
                    # TILE BOUNDARIES
                    # ----------------------------------------

                    y1 = min(
                        y0 + TILE_SIZE,
                        height
                    )

                    x1 = min(
                        x0 + TILE_SIZE,
                        width
                    )

                    tile_height = y1 - y0
                    tile_width = x1 - x0

                    print()
                    print(
                        f"[{tile_number}/{total_tiles}] "
                        f"Tile x={x0}:{x1}, "
                        f"y={y0}:{y1}"
                    )

                    # ----------------------------------------
                    # READ TILE
                    # ----------------------------------------

                    tile = src.read(
                        1,
                        window=Window(
                            x0,
                            y0,
                            tile_width,
                            tile_height
                        )
                    )

                    binary = tile > 0

                    # ----------------------------------------
                    # EMPTY TILE
                    # ----------------------------------------

                    if not np.any(binary):

                        print(
                            "  Empty tile - writing zeros"
                        )

                        empty = np.zeros(
                            (
                                tile_height,
                                tile_width
                            ),
                            dtype=np.uint8
                        )

                        cleaned_dst.write(
                            empty,
                            1,
                            window=Window(
                                x0,
                                y0,
                                tile_width,
                                tile_height
                            )
                        )

                        skeleton_dst.write(
                            empty,
                            1,
                            window=Window(
                                x0,
                                y0,
                                tile_width,
                                tile_height
                            )
                        )

                        continue

                    # ========================================
                    # STEP 1 - MORPHOLOGICAL CLOSING
                    # ========================================

                    closing_structure = np.ones(
                        (
                            CLOSING_SIZE,
                            CLOSING_SIZE
                        ),
                        dtype=bool
                    )

                    closed = ndimage.binary_closing(
                        binary,
                        structure=closing_structure
                    )

                    # ========================================
                    # STEP 2 - MORPHOLOGICAL OPENING
                    # ========================================

                    opening_structure = np.ones(
                        (
                            OPENING_SIZE,
                            OPENING_SIZE
                        ),
                        dtype=bool
                    )

                    cleaned = ndimage.binary_opening(
                        closed,
                        structure=opening_structure
                    )

                    # ========================================
                    # STEP 3 - CONNECTED COMPONENT FILTER
                    # ========================================

                    connectivity = np.ones(
                        (3, 3),
                        dtype=np.uint8
                    )

                    labeled, number_of_components = (
                        ndimage.label(
                            cleaned,
                            structure=connectivity
                        )
                    )

                    component_sizes = np.bincount(
                        labeled.ravel()
                    )

                    keep = (
                        component_sizes >=
                        MIN_COMPONENT_SIZE
                    )

                    keep[0] = False

                    cleaned_filtered = keep[
                        labeled
                    ]

                    cleaned_mask = (
                        cleaned_filtered.astype(
                            np.uint8
                        )
                    )

                    cleaned_pixels = int(
                        np.count_nonzero(
                            cleaned_mask
                        )
                    )

                    total_cleaned_pixels += (
                        cleaned_pixels
                    )

                    print(
                        "  Components:",
                        number_of_components
                    )

                    print(
                        "  Cleaned pixels:",
                        cleaned_pixels
                    )

                    # ========================================
                    # TILE INTERIOR
                    #
                    # Keep only central area.
                    # This prevents duplicate overlap
                    # and reduces tile-boundary artifacts.
                    # ========================================

                    top_cut = 0
                    bottom_cut = 0
                    left_cut = 0
                    right_cut = 0

                    if y0 > 0:
                        top_cut = OVERLAP // 2

                    if y1 < height:
                        bottom_cut = OVERLAP // 2

                    if x0 > 0:
                        left_cut = OVERLAP // 2

                    if x1 < width:
                        right_cut = OVERLAP // 2

                    central_cleaned = cleaned_mask[
                        top_cut:
                        tile_height - bottom_cut
                        if bottom_cut > 0
                        else tile_height,

                        left_cut:
                        tile_width - right_cut
                        if right_cut > 0
                        else tile_width
                    ]

                    # ========================================
                    # STEP 4 - SKELETONIZE TILE
                    # ========================================

                    print(
                        "  Skeletonizing tile..."
                    )

                    tile_skeleton = skeletonize(
                        cleaned_mask > 0
                    )

                    # ========================================
                    # CENTRAL SKELETON
                    # ========================================

                    central_skeleton = tile_skeleton[
                        top_cut:
                        tile_height - bottom_cut
                        if bottom_cut > 0
                        else tile_height,

                        left_cut:
                        tile_width - right_cut
                        if right_cut > 0
                        else tile_width
                    ]

                    central_skeleton = (
                        central_skeleton.astype(
                            np.uint8
                        )
                    )

                    skeleton_pixels = int(
                        np.count_nonzero(
                            central_skeleton
                        )
                    )

                    total_skeleton_pixels += (
                        skeleton_pixels
                    )

                    # ========================================
                    # OUTPUT POSITION
                    # ========================================

                    output_x = (
                        x0 + left_cut
                    )

                    output_y = (
                        y0 + top_cut
                    )

                    output_width = (
                        central_cleaned.shape[1]
                    )

                    output_height = (
                        central_cleaned.shape[0]
                    )

                    output_window = Window(
                        output_x,
                        output_y,
                        output_width,
                        output_height
                    )

                    # ========================================
                    # WRITE CLEANED MASK
                    # ========================================

                    cleaned_dst.write(
                        central_cleaned.astype(
                            np.uint8
                        ),
                        1,
                        window=output_window
                    )

                    # ========================================
                    # WRITE SKELETON
                    # ========================================

                    skeleton_dst.write(
                        central_skeleton,
                        1,
                        window=output_window
                    )

                    print(
                        "  Skeleton pixels:",
                        skeleton_pixels
                    )

                    print(
                        "  Tile completed."
                    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("V3 POST-PROCESSING COMPLETED")
print("=" * 70)

print()
print("Cleaned mask:")
print(CLEANED_MASK_TIF)

print()
print("Skeleton:")
print(SKELETON_TIF)

print()
print("Total cleaned pixels:")
print(total_cleaned_pixels)

print()
print("Total skeleton pixels:")
print(total_skeleton_pixels)

print()
print("Next step:")
print(
    "Use rail_skeleton_v3.tif "
    "for vectorization."
)

print()
print("=" * 70)
