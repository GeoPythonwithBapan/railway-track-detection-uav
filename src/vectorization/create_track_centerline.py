"""
Railway Track Centerline Generation
===================================

Generate railway track centerlines from paired rail-line vectors.

Workflow:
    Repaired Rail Lines
          |
          v
    Validate Rail Geometries
          |
          v
    Find Parallel Rail Pairs
          |
          v
    Sample Corresponding Points
          |
          v
    Calculate Midpoints
          |
          v
    Generate Track Centerlines
          |
          v
    Export GeoPackage Layers

Input:
    Repaired railway rail lines in a GeoPackage.

Output:
    1. Railway Track
    2. Railway Centerline

Requirements:
    geopandas
    shapely
    pandas
    numpy
"""

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_INPUT = (
    r"D:\Bapan\auto_digi\track\final_output"
    r"\repair\repaired_rail_lines_v2.gpkg"
)

DEFAULT_INPUT_LAYER = "repaired_rails"

DEFAULT_OUTPUT = (
    r"D:\Bapan\auto_digi\track\final_output"
    r"\track_centerline.gpkg"
)

SAMPLE_POINTS = 500

MAX_RAIL_DISTANCE = 3.0       # Metres
MAX_ANGLE_DIFF = 15.0         # Degrees
MIN_RAIL_LENGTH = 5.0         # Metres

OUTPUT_TRACK_LAYER = "Railway Track"
OUTPUT_CENTERLINE_LAYER = "Railway Centerline"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# GEOMETRY UTILITIES
# ============================================================

def normalize_line_geometry(geometry):
    """
    Convert a LineString or mergeable MultiLineString
    into a single LineString.

    Returns None if the geometry cannot be used.
    """

    if geometry is None or geometry.is_empty:
        return None

    if geometry.geom_type == "LineString":
        line = geometry

    elif geometry.geom_type == "MultiLineString":
        try:
            merged = linemerge(geometry)

            if merged.geom_type != "LineString":
                return None

            line = merged

        except Exception:
            return None

    else:
        return None

    if line.is_empty or len(line.coords) < 2:
        return None

    if not line.is_valid:
        try:
            line = line.simplify(0)
        except Exception:
            return None

    if line.is_empty or line.length <= 0:
        return None

    return line


def calculate_line_angle(line):
    """
    Calculate the approximate overall direction of a line.

    The result is in degrees.
    """

    start = line.coords[0]
    end = line.coords[-1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    return math.degrees(math.atan2(dy, dx))


def angle_difference(angle1, angle2):
    """
    Calculate the smallest orientation difference.

    Railway lines pointing in opposite directions can still
    be parallel, so orientation is treated modulo 180 degrees.
    """

    difference = abs(angle1 - angle2) % 180.0

    return min(difference, 180.0 - difference)


def orient_lines_consistently(line1, line2):
    """
    Reverse the second line when necessary so that its
    endpoints correspond more closely to those of the first.
    """

    coords1 = list(line1.coords)
    coords2 = list(line2.coords)

    direct_distance = (
        math.dist(coords1[0][:2], coords2[0][:2])
        + math.dist(coords1[-1][:2], coords2[-1][:2])
    )

    reverse_distance = (
        math.dist(coords1[0][:2], coords2[-1][:2])
        + math.dist(coords1[-1][:2], coords2[0][:2])
    )

    if reverse_distance < direct_distance:
        line2 = LineString(coords2[::-1])

    return line1, line2


def sample_line(line, number_of_points):
    """
    Sample evenly spaced points along a line.
    """

    distances = np.linspace(
        0,
        line.length,
        number_of_points
    )

    return [
        line.interpolate(float(distance))
        for distance in distances
    ]


# ============================================================
# RAIL PAIR EVALUATION
# ============================================================

def evaluate_rail_pair(
    line1,
    line2,
    max_distance,
    max_angle_difference,
    sample_count
):
    """
    Evaluate whether two lines form a suitable rail pair.

    Checks:
        1. Overall direction difference.
        2. Average corresponding-point distance.
        3. Maximum sampled corresponding-point distance.

    Returns:
        Dictionary of measurements if accepted, otherwise None.
    """

    angle1 = calculate_line_angle(line1)
    angle2 = calculate_line_angle(line2)

    angle_diff = angle_difference(angle1, angle2)

    if angle_diff > max_angle_difference:
        return None

    line1, line2 = orient_lines_consistently(line1, line2)

    points1 = sample_line(line1, sample_count)
    points2 = sample_line(line2, sample_count)

    distances = np.array([
        point1.distance(point2)
        for point1, point2 in zip(points1, points2)
    ])

    mean_distance = float(np.mean(distances))
    max_sampled_distance = float(np.max(distances))

    if mean_distance > max_distance:
        return None

    # Reject pairs whose sampled separation is excessively large.
    # This helps prevent pairing unrelated railway lines.
    if max_sampled_distance > max_distance * 2:
        return None

    return {
        "line1": line1,
        "line2": line2,
        "angle_diff": angle_diff,
        "mean_distance": mean_distance,
        "max_sampled_distance": max_sampled_distance,
    }


# ============================================================
# CENTERLINE GENERATION
# ============================================================

def generate_centerline(line1, line2, sample_count):
    """
    Generate a centerline from two corresponding rail lines.

    Both lines are sampled at equal normalized distances.
    Their corresponding coordinates are averaged.

    Returns:
        Centerline geometry and average rail separation.
    """

    line1, line2 = orient_lines_consistently(line1, line2)

    points1 = sample_line(line1, sample_count)
    points2 = sample_line(line2, sample_count)

    center_coordinates = []

    distances = []

    for point1, point2 in zip(points1, points2):

        midpoint_x = (point1.x + point2.x) / 2.0
        midpoint_y = (point1.y + point2.y) / 2.0

        center_coordinates.append(
            (midpoint_x, midpoint_y)
        )

        distances.append(
            point1.distance(point2)
        )

    if len(center_coordinates) < 2:
        return None, None

    centerline = LineString(center_coordinates)

    if centerline.is_empty or centerline.length <= 0:
        return None, None

    average_separation = float(np.mean(distances))

    return centerline, average_separation


# ============================================================
# MAIN PROCESSING
# ============================================================

def create_track_centerlines(
    input_path,
    input_layer,
    output_path,
    sample_count=500,
    max_rail_distance=3.0,
    max_angle_diff=15.0,
    min_rail_length=5.0,
):
    """
    Read rail lines, identify rail pairs and export centerlines.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input GeoPackage not found: {input_path}"
        )

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    if max_rail_distance <= 0:
        raise ValueError(
            "max_rail_distance must be greater than zero."
        )

    if not 0 <= max_angle_diff <= 90:
        raise ValueError(
            "max_angle_diff must be between 0 and 90 degrees."
        )

    logger.info("Reading repaired rail lines...")
    logger.info("Input: %s", input_path)
    logger.info("Layer: %s", input_layer)

    rails = gpd.read_file(
        input_path,
        layer=input_layer
    )

    if rails.empty:
        raise ValueError(
            "The input layer contains no features."
        )

    if rails.crs is None:
        raise ValueError(
            "Input CRS is missing. Assign the correct CRS before processing."
        )

    # Distances and lengths must be measured in metres.
    if rails.crs.is_geographic:
        raise ValueError(
            "Input CRS is geographic. Reproject the rail lines "
            "to an appropriate projected CRS with metre units first."
        )

    logger.info("Input features: %d", len(rails))
    logger.info("Input CRS: %s", rails.crs)

    # --------------------------------------------------------
    # Normalize and filter input geometries
    # --------------------------------------------------------

    valid_rails = []

    for index, row in rails.iterrows():

        line = normalize_line_geometry(row.geometry)

        if line is None:
            continue

        if line.length < min_rail_length:
            continue

        valid_rails.append({
            "source_index": index,
            "geometry": line,
            "length": float(line.length),
        })

    if len(valid_rails) < 2:
        raise ValueError(
            "Fewer than two valid rail lines remain after filtering."
        )

    logger.info(
        "Valid rail lines after filtering: %d",
        len(valid_rails)
    )

    # --------------------------------------------------------
    # Generate potential rail-pair candidates
    # --------------------------------------------------------

    logger.info("Searching for suitable rail pairs...")

    candidates = []

    for i in range(len(valid_rails)):

        line1 = valid_rails[i]["geometry"]

        for j in range(i + 1, len(valid_rails)):

            line2 = valid_rails[j]["geometry"]

            # A quick spatial filter before detailed sampling.
            if line1.distance(line2) > max_rail_distance:
                continue

            evaluation = evaluate_rail_pair(
                line1=line1,
                line2=line2,
                max_distance=max_rail_distance,
                max_angle_difference=max_angle_diff,
                sample_count=sample_count,
            )

            if evaluation is None:
                continue

            candidates.append({
                "i": i,
                "j": j,
                "evaluation": evaluation,
                "score": (
                    evaluation["mean_distance"]
                    + evaluation["angle_diff"] * 0.01
                ),
            })

    if not candidates:
        logger.warning(
            "No suitable rail pairs found. Consider reviewing "
            "the input geometry and pairing parameters."
        )

        empty_tracks = gpd.GeoDataFrame(
            {
                "Track_ID": pd.Series(dtype="int64"),
                "Rail_1": pd.Series(dtype="int64"),
                "Rail_2": pd.Series(dtype="int64"),
                "Rail_Sep_m": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(
                    [],
                    crs=rails.crs
                ),
            },
            geometry="geometry",
            crs=rails.crs,
        )

        empty_centerlines = gpd.GeoDataFrame(
            {
                "Track_ID": pd.Series(dtype="int64"),
                "Center_L_m": pd.Series(dtype="float64"),
                "Rail_Sep_m": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(
                    [],
                    crs=rails.crs
                ),
            },
            geometry="geometry",
            crs=rails.crs,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if output_path.exists():
            output_path.unlink()

        empty_tracks.to_file(
            output_path,
            layer=OUTPUT_TRACK_LAYER,
            driver="GPKG",
        )

        empty_centerlines.to_file(
            output_path,
            layer=OUTPUT_CENTERLINE_LAYER,
            driver="GPKG",
            mode="a",
        )

        logger.info("Empty output layers were created.")
        return

    # --------------------------------------------------------
    # Select pairs greedily
    #
    # A rail feature is used only once. This reduces the chance
    # of generating several centerlines from the same rail.
    # --------------------------------------------------------

    candidates.sort(key=lambda item: item["score"])

    used_rails = set()
    selected_pairs = []

    for candidate in candidates:

        i = candidate["i"]
        j = candidate["j"]

        if i in used_rails or j in used_rails:
            continue

        selected_pairs.append(candidate)

        used_rails.add(i)
        used_rails.add(j)

    logger.info(
        "Selected rail pairs: %d",
        len(selected_pairs)
    )

    # --------------------------------------------------------
    # Generate output features
    # --------------------------------------------------------

    track_records = []
    centerline_records = []

    for track_id, pair in enumerate(
        selected_pairs,
        start=1
    ):

        i = pair["i"]
        j = pair["j"]

        evaluation = pair["evaluation"]

        line1 = evaluation["line1"]
        line2 = evaluation["line2"]

        centerline, average_separation = generate_centerline(
            line1=line1,
            line2=line2,
            sample_count=sample_count,
        )

        if centerline is None:
            logger.warning(
                "Could not generate centerline for pair %d.",
                track_id
            )
            continue

        rail1_source = valid_rails[i]["source_index"]
        rail2_source = valid_rails[j]["source_index"]

        # Keep a unique ID for each selected pair.
        track_records.append({
            "Track_ID": track_id,
            "Rail_1": str(rail1_source),
            "Rail_2": str(rail2_source),
            "Rail_Sep_m": round(
                average_separation,
                3
            ),
            "Angle_Diff": round(
                evaluation["angle_diff"],
                3
            ),
            "geometry": MultiLineString([
                line1,
                line2
            ]),
        })

        centerline_records.append({
            "Track_ID": track_id,
            "Center_L_m": round(
                centerline.length,
                3
            ),
            "Rail_Sep_m": round(
                average_separation,
                3
            ),
            "geometry": centerline,
        })

        logger.info(
            "Track %d | Separation: %.3f m | "
            "Centerline length: %.3f m",
            track_id,
            average_separation,
            centerline.length,
        )

    # --------------------------------------------------------
    # Create GeoDataFrames
    # --------------------------------------------------------

    if track_records:

        track_gdf = gpd.GeoDataFrame(
            track_records,
            geometry="geometry",
            crs=rails.crs,
        )

    else:

        track_gdf = gpd.GeoDataFrame(
            {
                "Track_ID": pd.Series(dtype="int64"),
                "Rail_1": pd.Series(dtype="str"),
                "Rail_2": pd.Series(dtype="str"),
                "Rail_Sep_m": pd.Series(dtype="float64"),
                "Angle_Diff": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(
                    [],
                    crs=rails.crs
                ),
            },
            geometry="geometry",
            crs=rails.crs,
        )

    if centerline_records:

        centerline_gdf = gpd.GeoDataFrame(
            centerline_records,
            geometry="geometry",
            crs=rails.crs,
        )

    else:

        centerline_gdf = gpd.GeoDataFrame(
            {
                "Track_ID": pd.Series(dtype="int64"),
                "Center_L_m": pd.Series(dtype="float64"),
                "Rail_Sep_m": pd.Series(dtype="float64"),
                "geometry": gpd.GeoSeries(
                    [],
                    crs=rails.crs
                ),
            },
            geometry="geometry",
            crs=rails.crs,
        )

    # --------------------------------------------------------
    # Export GeoPackage
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():
        output_path.unlink()

    logger.info("Exporting results...")

    track_gdf.to_file(
        output_path,
        layer=OUTPUT_TRACK_LAYER,
        driver="GPKG",
    )

    centerline_gdf.to_file(
        output_path,
        layer=OUTPUT_CENTERLINE_LAYER,
        driver="GPKG",
        mode="a",
    )

    logger.info("=" * 60)
    logger.info("CENTERLINE GENERATION COMPLETED")
    logger.info("=" * 60)
    logger.info(
        "Railway Track features: %d",
        len(track_gdf)
    )
    logger.info(
        "Railway Centerline features: %d",
        len(centerline_gdf)
    )
    logger.info("Output: %s", output_path)
    logger.info("CRS: %s", rails.crs)
    logger.info("=" * 60)


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate railway track centerlines "
            "from repaired rail-line vectors."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input rail-line GeoPackage path.",
    )

    parser.add_argument(
        "--input-layer",
        default=DEFAULT_INPUT_LAYER,
        help="Input GeoPackage layer name.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output GeoPackage path.",
    )

    parser.add_argument(
        "--sample-points",
        type=int,
        default=SAMPLE_POINTS,
        help="Number of corresponding points sampled per rail pair.",
    )

    parser.add_argument(
        "--max-rail-distance",
        type=float,
        default=MAX_RAIL_DISTANCE,
        help="Maximum mean separation between paired rails, in CRS units.",
    )

    parser.add_argument(
        "--max-angle-diff",
        type=float,
        default=MAX_ANGLE_DIFF,
        help="Maximum angle difference in degrees.",
    )

    parser.add_argument(
        "--min-rail-length",
        type=float,
        default=MIN_RAIL_LENGTH,
        help="Minimum accepted rail-line length, in CRS units.",
    )

    args = parser.parse_args()

    create_track_centerlines(
        input_path=args.input,
        input_layer=args.input_layer,
        output_path=args.output,
        sample_count=args.sample_points,
        max_rail_distance=args.max_rail_distance,
        max_angle_diff=args.max_angle_diff,
        min_rail_length=args.min_rail_length,
    )


if __name__ == "__main__":
    main()
