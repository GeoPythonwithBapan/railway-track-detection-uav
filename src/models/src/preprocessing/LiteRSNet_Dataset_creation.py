import os
import shutil
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TILES_DIR = r"D:\Bapan\auto_digi\track\tiles"

MASK_DIR = r"D:\Bapan\auto_digi\track\mask_generation"

TILE_INDEX_CSV = r"D:\Bapan\auto_digi\track\tiles\tile_index.csv"

OUTPUT_DIR = r"D:\Bapan\auto_digi\track\LiteRSNet_Dataset"


# ============================================================
# SPLIT RATIO
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CHECK RATIOS
# ============================================================

if abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) > 1e-6:

    raise ValueError(
        "Train/Validation/Test ratios must sum to 1.0"
    )


# ============================================================
# READ TILE INDEX
# ============================================================

print("\n==============================================")
print("READING TILE INDEX")
print("==============================================")

if not os.path.exists(TILE_INDEX_CSV):

    raise FileNotFoundError(
        f"Tile index not found:\n{TILE_INDEX_CSV}"
    )


tile_index = pd.read_csv(
    TILE_INDEX_CSV
)

print("\nTile index columns:")
print(tile_index.columns.tolist())

print(
    f"\nTotal tiles in tile index: {len(tile_index)}"
)


# ============================================================
# FIND TILE NAME COLUMN
# ============================================================

possible_name_columns = [
    "tile_name",
    "filename",
    "tile",
    "name"
]

tile_name_column = None

for col in possible_name_columns:

    if col in tile_index.columns:

        tile_name_column = col
        break


if tile_name_column is None:

    raise ValueError(
        "\nCould not find tile-name column.\n"
        "Available columns:\n"
        + str(tile_index.columns.tolist())
    )


print(
    f"Using tile-name column: {tile_name_column}"
)


# ============================================================
# NORMALIZE TILE NAMES
# ============================================================

tile_index["tile_name_clean"] = (

    tile_index[tile_name_column]
    .astype(str)
    .str.strip()
    .str.replace(".tif", "", regex=False)
)


# ============================================================
# FIND ANNOTATED TILES
# ============================================================

print("\n==============================================")
print("FINDING ANNOTATED TILES")
print("==============================================")


mask_files = []

for filename in os.listdir(MASK_DIR):

    if filename.lower().endswith("_mask.png"):

        tile_name = filename[:-9]

        mask_files.append(tile_name)


mask_files = sorted(
    set(mask_files)
)


print(
    f"\nAnnotated mask tiles found: {len(mask_files)}"
)


if len(mask_files) == 0:

    raise RuntimeError(
        "No mask files found."
    )


# ============================================================
# KEEP ONLY ANNOTATED TILES
# ============================================================

annotated_index = tile_index[
    tile_index["tile_name_clean"].isin(mask_files)
].copy()


annotated_index = annotated_index.drop_duplicates(
    subset=["tile_name_clean"]
)


print(
    f"Annotated tiles matched with tile_index: "
    f"{len(annotated_index)}"
)


# ============================================================
# CHECK FOR MISSING TILE INDEX ENTRIES
# ============================================================

index_names = set(
    annotated_index["tile_name_clean"]
)

mask_names = set(
    mask_files
)

missing_index = sorted(
    mask_names - index_names
)


if missing_index:

    print(
        "\nWARNING: These mask tiles were not found "
        "in tile_index.csv:"
    )

    for name in missing_index:

        print(
            "   ",
            name
        )


# ============================================================
# CHECK REQUIRED POSITION COLUMNS
# ============================================================

print("\n==============================================")
print("CHECKING SPATIAL INFORMATION")
print("==============================================")


possible_x_columns = [
    "x_pixel",
    "x",
    "col",
    "column"
]

possible_y_columns = [
    "y_pixel",
    "y",
    "row",
    "rows"
]


x_column = None
y_column = None


for col in possible_x_columns:

    if col in annotated_index.columns:

        x_column = col
        break


for col in possible_y_columns:

    if col in annotated_index.columns:

        y_column = col
        break


if x_column is None or y_column is None:

    raise ValueError(
        "\nCould not find pixel-position columns "
        "in tile_index.csv.\n\n"
        "Available columns:\n"
        + str(annotated_index.columns.tolist())
    )


print(
    f"X position column: {x_column}"
)

print(
    f"Y position column: {y_column}"
)


# ============================================================
# SORT SPATIALLY
# ============================================================

annotated_index = annotated_index.sort_values(
    by=[y_column, x_column]
).reset_index(drop=True)


# ============================================================
# DISPLAY ANNOTATED TILES
# ============================================================

print("\n==============================================")
print("ANNOTATED TILE LIST")
print("==============================================")

for i, row in annotated_index.iterrows():

    print(
        f"{i + 1:3d}. "
        f"{row['tile_name_clean']:12s} "
        f"X={row[x_column]:6.0f} "
        f"Y={row[y_column]:6.0f}"
    )


# ============================================================
# SPATIAL SPLIT
# ============================================================
#
# We divide the sorted spatial list into three regions.
#
# This is NOT a random split.
#
# Approximately:
#
# 70% -> Train
# 15% -> Validation
# 15% -> Test
#
# ============================================================

total_tiles = len(annotated_index)

train_count = int(
    total_tiles * TRAIN_RATIO
)

val_count = int(
    total_tiles * VAL_RATIO
)

# Make sure all remaining tiles go to test
test_count = (
    total_tiles
    - train_count
    - val_count
)


print("\n==============================================")
print("CREATING SPATIAL DATASET SPLIT")
print("==============================================")


print(
    f"\nTotal annotated tiles: {total_tiles}"
)

print(
    f"Train: {train_count}"
)

print(
    f"Validation: {val_count}"
)

print(
    f"Test: {test_count}"
)


# ============================================================
# SPLIT
# ============================================================

train_df = annotated_index.iloc[
    :train_count
].copy()

val_df = annotated_index.iloc[
    train_count:
    train_count + val_count
].copy()

test_df = annotated_index.iloc[
    train_count + val_count:
].copy()


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

print("\n==============================================")
print("CREATING DATASET FOLDERS")
print("==============================================")


for split in [
    "train",
    "val",
    "test"
]:

    image_dir = os.path.join(
        OUTPUT_DIR,
        "images",
        split
    )

    mask_dir = os.path.join(
        OUTPUT_DIR,
        "masks",
        split
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    os.makedirs(
        mask_dir,
        exist_ok=True
    )


# ============================================================
# COPY FUNCTION
# ============================================================

def copy_dataset(
    dataframe,
    split
):

    print(
        f"\n------------- {split.upper()} -------------"
    )

    copied = 0
    missing_images = 0
    missing_masks = 0

    for _, row in dataframe.iterrows():

        tile_name = row[
            "tile_name_clean"
        ]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_src = os.path.join(
            TILES_DIR,
            tile_name + ".tif"
        )

        image_dst = os.path.join(
            OUTPUT_DIR,
            "images",
            split,
            tile_name + ".tif"
        )

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask_src = os.path.join(
            MASK_DIR,
            tile_name + "_mask.png"
        )

        mask_dst = os.path.join(
            OUTPUT_DIR,
            "masks",
            split,
            tile_name + "_mask.png"
        )

        # ----------------------------------------------------
        # CHECK IMAGE
        # ----------------------------------------------------

        if not os.path.exists(image_src):

            print(
                f"ERROR: Image not found: "
                f"{image_src}"
            )

            missing_images += 1

            continue

        # ----------------------------------------------------
        # CHECK MASK
        # ----------------------------------------------------

        if not os.path.exists(mask_src):

            print(
                f"ERROR: Mask not found: "
                f"{mask_src}"
            )

            missing_masks += 1

            continue

        # ----------------------------------------------------
        # COPY
        # ----------------------------------------------------

        shutil.copy2(
            image_src,
            image_dst
        )

        shutil.copy2(
            mask_src,
            mask_dst
        )

        print(
            f"[{split.upper()}] {tile_name}"
        )

        copied += 1

    print(
        f"\nCopied: {copied}"
    )

    print(
        f"Missing images: {missing_images}"
    )

    print(
        f"Missing masks: {missing_masks}"
    )

    return copied


# ============================================================
# COPY DATA
# ============================================================

train_copied = copy_dataset(
    train_df,
    "train"
)

val_copied = copy_dataset(
    val_df,
    "val"
)

test_copied = copy_dataset(
    test_df,
    "test"
)


# ============================================================
# SAVE SPLIT CSV
# ============================================================

split_records = []

for _, row in train_df.iterrows():

    split_records.append({
        "tile_name": row["tile_name_clean"],
        "split": "train",
        "x_pixel": row[x_column],
        "y_pixel": row[y_column]
    })


for _, row in val_df.iterrows():

    split_records.append({
        "tile_name": row["tile_name_clean"],
        "split": "val",
        "x_pixel": row[x_column],
        "y_pixel": row[y_column]
    })


for _, row in test_df.iterrows():

    split_records.append({
        "tile_name": row["tile_name_clean"],
        "split": "test",
        "x_pixel": row[x_column],
        "y_pixel": row[y_column]
    })


split_df = pd.DataFrame(
    split_records
)

split_csv = os.path.join(
    OUTPUT_DIR,
    "dataset_split.csv"
)

split_df.to_csv(
    split_csv,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==============================================")
print("DATASET CREATION COMPLETED")
print("==============================================")


print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nTrain images : {train_copied}"
)

print(
    f"Validation images : {val_copied}"
)

print(
    f"Test images : {test_copied}"
)

print(
    f"\nSplit information saved to:\n{split_csv}"
)


print("\nDataset structure:")

print(
    "images/train"
)

print(
    "images/val"
)

print(
    "images/test"
)

print(
    "masks/train"
)

print(
    "masks/val"
)

print(
    "masks/test"
)

print("\nDone.")
