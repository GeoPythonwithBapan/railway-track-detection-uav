"""
Lite-RSNet Dataset Quality Check

Checks:
1. Image and mask counts
2. Missing image-mask pairs
3. Mask dimensions
4. Binary mask values (0 = background, 1 = rail)
5. Rail/background pixel statistics
6. Rail coverage percentage

Expected folder structure:
    LiteRSNet_Dataset/
        images/
            train/
            val/
            test/
        masks/
            train/
            val/
            test/

Usage:
    python dataset_statistics.py --dataset-dir "D:/data/LiteRSNet_Dataset"
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ["train", "val", "test"]


def get_files(folder, extensions):
    """Return files matching the given extensions."""
    if not folder.is_dir():
        return []

    return sorted(
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in extensions
    )


def get_mask_name(filename):
    """Convert tile_0001_mask.png to tile_0001."""
    stem = Path(filename).stem

    if stem.endswith("_mask"):
        stem = stem[:-5]

    return stem


def main():

    parser = argparse.ArgumentParser(
        description="Validate the Lite-RSNet railway segmentation dataset."
    )

    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to the LiteRSNet_Dataset directory."
    )

    parser.add_argument(
        "--expected-size",
        type=int,
        default=1024,
        help="Expected mask width and height. Default: 1024."
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {dataset_dir}"
        )

    if args.expected_size <= 0:
        raise SystemExit(
            "--expected-size must be greater than zero."
        )

    print("\n==============================================")
    print("LITE-RSNET DATASET QUALITY CHECK")
    print("==============================================")

    print(f"Dataset: {dataset_dir}")

    total_images = 0
    total_masks = 0

    total_rail_pixels = 0
    total_background_pixels = 0

    problems_found = False

    for split in SPLITS:

        print("\n==============================================")
        print(f"{split.upper()} DATASET")
        print("==============================================")

        image_dir = dataset_dir / "images" / split
        mask_dir = dataset_dir / "masks" / split

        if not image_dir.is_dir():
            print(f"ERROR: Image directory not found: {image_dir}")
            problems_found = True
            continue

        if not mask_dir.is_dir():
            print(f"ERROR: Mask directory not found: {mask_dir}")
            problems_found = True
            continue

        image_files = get_files(
            image_dir,
            {".tif", ".tiff"}
        )

        mask_files = get_files(
            mask_dir,
            {".png"}
        )

        print(f"\nImages: {len(image_files)}")
        print(f"Masks : {len(mask_files)}")

        total_images += len(image_files)
        total_masks += len(mask_files)

        image_names = {
            file.stem for file in image_files
        }

        mask_names = {
            get_mask_name(file.name)
            for file in mask_files
        }

        missing_masks = sorted(
            image_names - mask_names
        )

        missing_images = sorted(
            mask_names - image_names
        )

        print(f"Missing masks : {len(missing_masks)}")
        print(f"Missing images: {len(missing_images)}")

        if missing_masks or missing_images:
            problems_found = True

        if missing_masks:
            print("\nImages without masks:")

            for name in missing_masks[:20]:
                print("  ", name)

        if missing_images:
            print("\nMasks without images:")

            for name in missing_images[:20]:
                print("  ", name)

        rail_pixels_total = 0
        background_pixels_total = 0

        bad_dimensions = []
        bad_values = []
        unreadable_masks = []

        valid_masks = 0

        for mask_file in mask_files:

            mask_path = mask_dir / mask_file.name

            try:
                with Image.open(mask_path) as image:
                    mask = np.asarray(image)

            except Exception as error:
                unreadable_masks.append(
                    (mask_file.name, str(error))
                )
                problems_found = True
                continue

            if mask.ndim != 2:
                bad_values.append(
                    (mask_file.name, f"Expected single-band mask; found shape {mask.shape}")
                )
                problems_found = True
                continue

            height, width = mask.shape

            if (
                width != args.expected_size
                or height != args.expected_size
            ):
                bad_dimensions.append(
                    (mask_file.name, width, height)
                )
                problems_found = True

            unique_values = np.unique(mask)

            if not set(unique_values.tolist()).issubset({0, 1}):
                bad_values.append(
                    (mask_file.name, unique_values.tolist())
                )
                problems_found = True

            rail_pixels = int(
                np.count_nonzero(mask == 1)
            )

            background_pixels = int(
                np.count_nonzero(mask == 0)
            )

            rail_pixels_total += rail_pixels
            background_pixels_total += background_pixels

            valid_masks += 1

        print(f"\nReadable masks: {valid_masks}")
        print(f"Incorrect dimensions: {len(bad_dimensions)}")
        print(f"Invalid masks: {len(bad_values)}")
        print(f"Unreadable masks: {len(unreadable_masks)}")

        if bad_dimensions:
            print("\nIncorrect dimensions:")

            for item in bad_dimensions[:20]:
                print("  ", item)

        if bad_values:
            print("\nInvalid mask values or formats:")

            for item in bad_values[:20]:
                print("  ", item)

        if unreadable_masks:
            print("\nUnreadable masks:")

            for item in unreadable_masks[:20]:
                print("  ", item)

        split_pixels = (
            rail_pixels_total + background_pixels_total
        )

        rail_percentage = (
            100 * rail_pixels_total / split_pixels
            if split_pixels else 0
        )

        print(f"\nRail pixels: {rail_pixels_total:,}")
        print(f"Background pixels: {background_pixels_total:,}")
        print(f"Rail coverage: {rail_percentage:.4f}%")

        total_rail_pixels += rail_pixels_total
        total_background_pixels += background_pixels_total

    global_pixels = (
        total_rail_pixels + total_background_pixels
    )

    global_coverage = (
        100 * total_rail_pixels / global_pixels
        if global_pixels else 0
    )

    print("\n==============================================")
    print("FINAL DATASET SUMMARY")
    print("==============================================")

    print(f"\nTotal images: {total_images}")
    print(f"Total masks: {total_masks}")

    print(f"Total rail pixels: {total_rail_pixels:,}")
    print(f"Total background pixels: {total_background_pixels:,}")

    print(f"Global rail coverage: {global_coverage:.4f}%")

    print("\n==============================================")
    print("DATASET CHECK COMPLETED")
    print("==============================================")

    if problems_found:
        print("\nWARNING: Dataset problems were detected.")
        print("Review the errors and warnings above.")
    else:
        print("\nSUCCESS: No dataset problems were detected.")


if __name__ == "__main__":
    main()
