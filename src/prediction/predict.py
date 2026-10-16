import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image
import torch

from models.lite_rsnet import LiteRSNet


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    r"D:\Bapan\VS\rail_track_detection\runs\lite_rsnet_cldice_v1\checkpoints\best_model.pth"
)

INPUT_IMAGE = (
    r"D:\Bapan\auto_digi\track\Input\input2\input_image2.tif"
)

OUTPUT_DIR = (
    r"D:\Bapan\auto_digi\track\prediction"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TILE_SIZE = 1024
OVERLAP = 0.25
STRIDE = int(TILE_SIZE * (1.0 - OVERLAP))

THRESHOLD = 0.45

# Small preview only
PREVIEW_SCALE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE INFORMATION
# ============================================================

print("=" * 70)
print("LITE-RSNET LARGE IMAGE MEMORY-SAFE PREDICTION")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading model...")

model = LiteRSNet(
    in_channels=3,
    num_classes=1
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")

print(
    "Checkpoint epoch:",
    checkpoint.get("epoch", "N/A")
)

print(
    "Validation Dice:",
    checkpoint.get("val_dice", "N/A")
)


# ============================================================
# TILE POSITIONS
# ============================================================

def get_positions(length, tile_size, stride):

    positions = list(
        range(
            0,
            max(
                length - tile_size + 1,
                1
            ),
            stride
        )
    )

    final_position = length - tile_size

    if final_position > 0:

        if positions[-1] != final_position:
            positions.append(final_position)

    return positions


# ============================================================
# MAIN
# ============================================================

with rasterio.open(INPUT_IMAGE) as src:

    width = src.width
    height = src.height

    profile = src.profile.copy()

    crs = src.crs

    print()
    print("Input image:")
    print(INPUT_IMAGE)

    print(
        "Size:",
        width,
        "x",
        height
    )

    print("CRS:", crs)

    print("Tile size:", TILE_SIZE)

    print(
        "Overlap:",
        f"{OVERLAP * 100:.0f}%"
    )

    print("Stride:", STRIDE)


    # ========================================================
    # TILE POSITIONS
    # ========================================================

    x_positions = get_positions(
        width,
        TILE_SIZE,
        STRIDE
    )

    y_positions = get_positions(
        height,
        TILE_SIZE,
        STRIDE
    )

    total_tiles = (
        len(x_positions) *
        len(y_positions)
    )

    print()
    print("X positions:", len(x_positions))
    print("Y positions:", len(y_positions))
    print("Total inference tiles:", total_tiles)


    # ========================================================
    # OUTPUT MASK
    # ========================================================

    mask_path = os.path.join(
        OUTPUT_DIR,
        "full_rail_mask.tif"
    )

    mask_profile = profile.copy()

    mask_profile.update(
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

    # Remove settings that may conflict with 1-band output
    mask_profile.pop("photometric", None)
    mask_profile.pop("interleave", None)


    # ========================================================
    # MEMORY-SAFE OUTPUT
    #
    # Only one vertical strip is kept in RAM.
    # ========================================================

    probability_sum = np.zeros(
        (TILE_SIZE, width),
        dtype=np.float32
    )

    weight_sum = np.zeros(
        (TILE_SIZE, width),
        dtype=np.uint8
    )


    total_rail_pixels = 0
    tile_number = 0


    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    with rasterio.open(
        mask_path,
        "w",
        **mask_profile
    ) as dst:

        # ----------------------------------------------------
        # PROCESS Y ROWS
        # ----------------------------------------------------

        for y_index, y in enumerate(y_positions):

            print()
            print(
                f"Processing Y row "
                f"{y_index + 1}/{len(y_positions)} "
                f"(Y={y})"
            )


            # ------------------------------------------------
            # For rows after the first:
            #
            # Keep the overlap from previous Y row.
            # ------------------------------------------------

            if y_index > 0:

                previous_y = y_positions[
                    y_index - 1
                ]

                rows_already_written = (
                    y - previous_y
                )

                overlap_rows = (
                    TILE_SIZE -
                    rows_already_written
                )

                if overlap_rows > 0:

                    probability_sum[
                        :overlap_rows
                    ] = probability_sum[
                        rows_already_written:
                        rows_already_written + overlap_rows
                    ]

                    weight_sum[
                        :overlap_rows
                    ] = weight_sum[
                        rows_already_written:
                        rows_already_written + overlap_rows
                    ]

                probability_sum[
                    overlap_rows:
                ] = 0

                weight_sum[
                    overlap_rows:
                ] = 0


            # ------------------------------------------------
            # PROCESS X TILES
            # ------------------------------------------------

            for x in x_positions:

                tile_number += 1

                window = Window(
                    x,
                    y,
                    TILE_SIZE,
                    TILE_SIZE
                )

                # --------------------------------------------
                # READ RGB TILE
                # --------------------------------------------

                image = src.read(
                    [1, 2, 3],
                    window=window
                )

                actual_height = image.shape[1]
                actual_width = image.shape[2]


                # --------------------------------------------
                # PAD
                # --------------------------------------------

                padded = np.zeros(
                    (
                        3,
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    dtype=np.uint8
                )

                padded[
                    :,
                    :actual_height,
                    :actual_width
                ] = image


                # --------------------------------------------
                # NORMALIZE
                # --------------------------------------------

                image_float = (
                    padded.astype(
                        np.float32
                    ) / 255.0
                )


                # --------------------------------------------
                # TENSOR
                # --------------------------------------------

                tensor = torch.from_numpy(
                    image_float
                ).unsqueeze(
                    0
                ).to(
                    DEVICE
                )


                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                with torch.no_grad():

                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=(
                            DEVICE.type == "cuda"
                        )
                    ):

                        logits = model(
                            tensor
                        )

                        probabilities = torch.sigmoid(
                            logits
                        )


                prediction = (
                    probabilities[
                        0,
                        0
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )


                # --------------------------------------------
                # REMOVE PADDING
                # --------------------------------------------

                prediction = prediction[
                    :actual_height,
                    :actual_width
                ]


                # --------------------------------------------
                # ADD TO STRIP
                # --------------------------------------------

                local_y_start = 0

                local_y_end = actual_height

                probability_sum[
                    local_y_start:local_y_end,
                    x:x + actual_width
                ] += prediction

                weight_sum[
                    local_y_start:local_y_end,
                    x:x + actual_width
                ] += 1


                # --------------------------------------------
                # PROGRESS
                # --------------------------------------------

                print(
                    f"\rTiles: "
                    f"{tile_number}/{total_tiles} "
                    f"({100 * tile_number / total_tiles:.1f}%)",
                    end=""
                )


                # --------------------------------------------
                # GPU MEMORY CLEANUP
                # --------------------------------------------

                del tensor
                del logits
                del probabilities
                del prediction


            print()


            # =================================================
            # WRITE COMPLETED ROWS
            # =================================================

            if y_index < len(y_positions) - 1:

                next_y = y_positions[
                    y_index + 1
                ]

                rows_to_write = (
                    next_y - y
                )

            else:

                rows_to_write = (
                    height - y
                )


            # -------------------------------------------------
            # AVERAGE
            # -------------------------------------------------

            avg_probability = (
                probability_sum[
                    :rows_to_write
                ]
                /
                np.maximum(
                    weight_sum[
                        :rows_to_write
                    ],
                    1
                )
            )


            # -------------------------------------------------
            # THRESHOLD
            # -------------------------------------------------

            mask_strip = (
                avg_probability >= THRESHOLD
            ).astype(
                np.uint8
            )


            # -------------------------------------------------
            # STATISTICS
            # -------------------------------------------------

            rail_pixels = int(
                mask_strip.sum()
            )

            total_rail_pixels += (
                rail_pixels
            )


            # -------------------------------------------------
            # WRITE MASK STRIP
            # -------------------------------------------------

            dst.write(
                mask_strip,
                1,
                window=Window(
                    0,
                    y,
                    width,
                    rows_to_write
                )
            )


            print(
                "Written rows:",
                y,
                "to",
                y + rows_to_write
            )

            print(
                "Rail pixels in strip:",
                rail_pixels
            )


    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    rail_percentage = (
        100.0 *
        total_rail_pixels /
        (width * height)
    )

    print()
    print("=" * 70)
    print("PREDICTION MASK COMPLETED")
    print("=" * 70)

    print(
        "Threshold:",
        THRESHOLD
    )

    print(
        "Total rail pixels:",
        total_rail_pixels
    )

    print(
        "Rail percentage:",
        f"{rail_percentage:.3f}%"
    )

    print()
    print("Mask saved:")
    print(mask_path)


# ============================================================
# CREATE LOW-RESOLUTION PREVIEW
# ============================================================

print()
print("Creating low-resolution preview...")

with rasterio.open(INPUT_IMAGE) as src:

    preview_height = max(
        1,
        height // PREVIEW_SCALE
    )

    preview_width = max(
        1,
        width // PREVIEW_SCALE
    )

    rgb_small = src.read(
        [1, 2, 3],
        out_shape=(
            3,
            preview_height,
            preview_width
        ),
        resampling=Resampling.bilinear
    )

    rgb_small = np.transpose(
        rgb_small,
        (1, 2, 0)
    ).astype(
        np.uint8
    )


with rasterio.open(mask_path) as mask_src:

    mask_small = mask_src.read(
        1,
        out_shape=(
            preview_height,
            preview_width
        ),
        resampling=Resampling.nearest
    )

    mask_small = (
        mask_small > 0
    )


# ============================================================
# SAVE PREVIEW
# ============================================================

mask_preview_path = os.path.join(
    OUTPUT_DIR,
    "full_rail_mask_preview.png"
)

overlay_path = os.path.join(
    OUTPUT_DIR,
    "full_rail_overlay_preview.jpg"
)


Image.fromarray(
    mask_small.astype(
        np.uint8
    ) * 255
).save(
    mask_preview_path
)


overlay = rgb_small.copy()

overlay[
    mask_small
] = [
    255,
    0,
    0
]


Image.fromarray(
    overlay
).save(
    overlay_path,
    quality=95
)


print()
print("Mask preview:")
print(mask_preview_path)

print()
print("Overlay preview:")
print(overlay_path)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("LARGE IMAGE PREDICTION COMPLETED")
print("=" * 70)

print()
print("Output folder:")
print(OUTPUT_DIR)
