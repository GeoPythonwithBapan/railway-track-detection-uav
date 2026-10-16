import pandas as pd
import os

TILE_INDEX = r"D:\Bapan\auto_digi\track\tiles\tile_index.csv"

ANNOTATED_TILES = [
    "tile_0038",
    "tile_0039",
    "tile_0053",
    "tile_0054",
    "tile_0068",
    "tile_0069",
    "tile_0083",
    "tile_0084"
]

df = pd.read_csv(TILE_INDEX)

selected = df[
    df["filename"].str.replace(".tif", "", regex=False).isin(
        ANNOTATED_TILES
    )
].copy()

print("\n==============================================")
print("ANNOTATED TILE LOCATIONS")
print("==============================================")

print(
    selected[
        [
            "tile_id",
            "filename",
            "x_pixel",
            "y_pixel",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ]
    ].to_string(index=False)
)

print("\nTotal annotated tiles found:", len(selected))
