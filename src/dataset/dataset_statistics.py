import os
import numpy as np
import rasterio
from PIL import Image


# ============================================================
# DATASET PATH
# ============================================================

DATASET_DIR = r"D:\Bapan\auto_digi\track\LiteRSNet_Dataset"


# ============================================================
# SPLITS
# ============================================================

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# MAIN CHECK
# ============================================================

print("\n==============================================")
print("LITE-RSNET DATASET QUALITY CHECK")
print("==============================================")


grand_total_images = 0
grand_total_masks = 0

grand_rail_pixels = 0
grand_background_pixels = 0


# ============================================================
# PROCESS EACH SPLIT
# ============================================================

for split in SPLITS:

    print("\n")
    print("==============================================")
    print(f"{split.upper()} DATASET")
    print("==============================================")

    image_dir = os.path.join(
        DATASET_DIR,
        "images",
        split
    )

    mask_dir = os.path.join(
        DATASET_DIR,
        "masks",
        split
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not os.path.exists(image_dir):

        print(
            f"ERROR: Image directory not found:\n"
            f"{image_dir}"
        )

        continue

    if not os.path.exists(mask_dir):

        print(
            f"ERROR: Mask directory not found:\n"
            f"{mask_dir}"
        )

        continue

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    image_files = sorted([
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(".tif")
    ])

    mask_files = sorted([
        f
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    ])

    print(
        f"\nImages: {len(image_files)}"
    )

    print(
        f"Masks : {len(mask_files)}"
    )

    grand_total_images += len(image_files)
    grand_total_masks += len(mask_files)

    # --------------------------------------------------------
    # Match images and masks
    # --------------------------------------------------------

    image_names = {
        os.path.splitext(f)[0]
        for f in image_files
    }

    mask_names = {
        f.replace("_mask.png", "")
        for f in mask_files
    }

    missing_masks = sorted(
        image_names - mask_names
    )

    missing_images = sorted(
        mask_names - image_names
    )

    print(
        f"Missing masks : {len(missing_masks)}"
    )

    print(
        f"Missing images: {len(missing_images)}"
    )

    if missing_masks:

        print("\nImages without masks:")

        for name in missing_masks[:20]:

            print(
                "  ",
                name
            )

    if missing_images:

        print("\nMasks without images:")

        for name in missing_images[:20]:

            print(
                "  ",
                name
            )

    # --------------------------------------------------------
    # Mask quality
    # --------------------------------------------------------

    bad_dimensions = []
    bad_values = []

    rail_pixels_total = 0
    background_pixels_total = 0

    valid_masks = 0

    for mask_file in mask_files:

        mask_path = os.path.join(
            mask_dir,
            mask_file
        )

        try:

            mask = np.array(
                Image.open(mask_path)
            )

        except Exception as e:

            print(
                f"ERROR reading mask: "
                f"{mask_file}"
            )

            print(e)

            continue

        # ----------------------------------------------------
        # Dimension check
        # ----------------------------------------------------

        height, width = mask.shape[:2]

        if width != 1024 or height != 1024:

            bad_dimensions.append(
                (
                    mask_file,
                    width,
                    height
                )
            )

        # ----------------------------------------------------
        # Value check
        # ----------------------------------------------------

        unique_values = np.unique(mask)

        allowed_values = {
            0,
            1
        }

        if not set(
            unique_values
        ).issubset(
            allowed_values
        ):

            bad_values.append(
                (
                    mask_file,
                    unique_values.tolist()
                )
            )

        # ----------------------------------------------------
        # Pixel statistics
        # ----------------------------------------------------

        rail_pixels = np.sum(
            mask == 1
        )

        background_pixels = np.sum(
            mask == 0
        )

        rail_pixels_total += int(
            rail_pixels
        )

        background_pixels_total += int(
            background_pixels
        )

        valid_masks += 1

    # --------------------------------------------------------
    # Print mask statistics
    # --------------------------------------------------------

    print(
        f"\nValid masks: {valid_masks}"
    )

    print(
        f"Incorrect dimensions: "
        f"{len(bad_dimensions)}"
    )

    print(
        f"Invalid pixel values: "
        f"{len(bad_values)}"
    )

    if bad_dimensions:

        print("\nBad dimensions:")

        for item in bad_dimensions[:20]:

            print(
                "  ",
                item
            )

    if bad_values:

        print("\nBad pixel values:")

        for item in bad_values[:20]:

            print(
                "  ",
                item
            )

    # --------------------------------------------------------
    # Pixel totals
    # --------------------------------------------------------

    total_pixels = (
        rail_pixels_total
        +
        background_pixels_total
    )

    if total_pixels > 0:

        rail_percentage = (
            rail_pixels_total
            /
            total_pixels
        ) * 100

    else:

        rail_percentage = 0

    print(
        "\nRail pixels:",
        f"{rail_pixels_total:,}"
    )

    print(
        "Background pixels:",
        f"{background_pixels_total:,}"
    )

    print(
        "Rail coverage:",
        f"{rail_percentage:.4f}%"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("==============================================")
print("FINAL DATASET SUMMARY")
print("==============================================")

print(
    f"\nTotal images: "
    f"{grand_total_images}"
)

print(
    f"Total masks : "
    f"{grand_total_masks}"
)

print(
    f"Total Rail pixels: "
    f"{grand_rail_pixels:,}"
)

print(
    f"Total Background pixels: "
    f"{grand_background_pixels:,}"
)


# Recalculate global totals directly
# from the three split folders

global_rail = 0
global_background = 0

for split in SPLITS:

    mask_dir = os.path.join(
        DATASET_DIR,
        "masks",
        split
    )

    if not os.path.exists(mask_dir):
        continue

    for mask_file in os.listdir(mask_dir):

        if not mask_file.endswith(".png"):
            continue

        mask_path = os.path.join(
            mask_dir,
            mask_file
        )

        mask = np.array(
            Image.open(mask_path)
        )

        global_rail += int(
            np.sum(mask == 1)
        )

        global_background += int(
            np.sum(mask == 0)
        )


global_total = (
    global_rail
    +
    global_background
)

if global_total > 0:

    global_rail_percentage = (
        global_rail
        /
        global_total
    ) * 100

else:

    global_rail_percentage = 0


print(
    "\nGlobal Rail pixels:",
    f"{global_rail:,}"
)

print(
    "Global Background pixels:",
    f"{global_background:,}"
)

print(
    "Global Rail coverage:",
    f"{global_rail_percentage:.4f}%"
)


print("\n==============================================")
print("DATASET CHECK COMPLETED")
print("==============================================")
