import os
import math
import numpy as np
import geopandas as gpd

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_GPKG = r"D:\Bapan\auto_digi\track\final_output\rail_lines.gpkg"
INPUT_LAYER = "rail_segments"

OUTPUT_DIR = r"D:\Bapan\auto_digi\track\final_output\repair"

OUTPUT_GPKG = os.path.join(
    OUTPUT_DIR,
    "repaired_rail_lines_v2.gpkg"
)

OUTPUT_LAYER = "repaired_rails"


# ============================================================
# PARAMETERS
# ============================================================

# Remove tiny skeleton/vector fragments
MIN_SEGMENT_LENGTH = 5.0       # meters

# IMPORTANT:
# The previous script used 1.2 m. That is too restrictive for
# the intermittent gaps visible in the current result.
MAX_GAP = 8.0                  # meters

# Direction of the two fragments must remain almost parallel.
MAX_ANGLE_DIFF = 15.0          # degrees

# Direction from each endpoint toward the other endpoint must
# agree with the local rail direction.
MAX_ENDPOINT_ANGLE = 20.0      # degrees

# Do not allow a single endpoint to create multiple branches.
MAX_CONNECTIONS_PER_ENDPOINT = 1

# Geometry simplification
SMOOTH_TOLERANCE = 0.05        # meters


# ============================================================
# ANGLE / VECTOR FUNCTIONS
# ============================================================

def normalize_angle(angle):
    """Normalize an undirected line angle to [0, 180)."""
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def angle_difference(a1, a2):
    """Smallest difference between two undirected angles."""
    diff = abs(a1 - a2) % 180.0
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def vector_angle_deg(v):
    """Angle of a 2D vector in degrees."""
    return math.degrees(math.atan2(v[1], v[0]))


def angle_between_vectors(v1, v2):
    """Smallest 0-180 degree angle between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0

    dot = np.dot(v1, v2) / (n1 * n2)
    dot = np.clip(dot, -1.0, 1.0)

    return math.degrees(math.acos(dot))


def line_direction(line):
    """Approximate overall direction of a line."""
    coords = list(line.coords)

    if len(coords) < 2:
        return None

    p1 = np.array(coords[0], dtype=float)
    p2 = np.array(coords[-1], dtype=float)

    return normalize_angle(
        vector_angle_deg(p2 - p1)
    )


def local_endpoint_info(line):
    """
    Return endpoint coordinates and local tangent vectors.

    The tangent vector always points FROM the endpoint INTO
    the line. This is important for deciding whether another
    fragment is a true continuation of the same rail.
    """

    coords = list(line.coords)

    start = np.array(coords[0], dtype=float)
    end = np.array(coords[-1], dtype=float)

    # Use a short local section, but never more than 2 m
    # if the geometry has enough points.
    total_length = line.length
    sample_distance = min(2.0, max(0.25, total_length * 0.10))

    start_ref = np.array(
        line.interpolate(sample_distance).coords[0],
        dtype=float
    )

    end_ref = np.array(
        line.interpolate(total_length - sample_distance).coords[0],
        dtype=float
    )

    start_tangent = start_ref - start
    end_tangent = end_ref - end

    # Normalize
    if np.linalg.norm(start_tangent) > 1e-9:
        start_tangent /= np.linalg.norm(start_tangent)

    if np.linalg.norm(end_tangent) > 1e-9:
        end_tangent /= np.linalg.norm(end_tangent)

    return {
        "start": start,
        "end": end,
        "start_tangent": start_tangent,
        "end_tangent": end_tangent,
    }


# ============================================================
# ENDPOINT CONNECTION TEST
# ============================================================

def test_endpoint_connection(
    p1,
    tangent1,
    p2,
    tangent2,
):
    """
    Test whether p1 -> p2 looks like a continuation of the
    same rail.

    At both endpoints, the tangent points INTO its own segment.
    Therefore the bridge direction must point AWAY from segment 1
    and INTO segment 2.

    Returns:
        (valid, distance, bridge_angle, angle1, angle2)
    """

    bridge = p2 - p1
    distance = np.linalg.norm(bridge)

    if distance < 1e-6:
        return False, distance, 0.0, 180.0, 180.0

    bridge_unit = bridge / distance

    # Segment 1 tangent points into segment 1, so the bridge
    # should point opposite to tangent1.
    expected_from_1 = -tangent1

    # Segment 2 tangent points into segment 2, so the bridge
    # direction should point in the same direction as tangent2.
    expected_into_2 = tangent2

    angle1 = angle_between_vectors(
        bridge_unit,
        expected_from_1
    )

    angle2 = angle_between_vectors(
        bridge_unit,
        expected_into_2
    )

    bridge_angle = normalize_angle(
        vector_angle_deg(bridge)
    )

    valid = (
        angle1 <= MAX_ENDPOINT_ANGLE
        and angle2 <= MAX_ENDPOINT_ANGLE
    )

    return (
        valid,
        distance,
        bridge_angle,
        angle1,
        angle2
    )


# ============================================================
# LOAD
# ============================================================

print("=" * 75)
print("RAIL-AWARE CONTINUITY REPAIR V2")
print("=" * 75)

print("\nInput:")
print(INPUT_GPKG)

gdf = gpd.read_file(
    INPUT_GPKG,
    layer=INPUT_LAYER
)

print("\nOriginal segments:", len(gdf))
print("CRS:", gdf.crs)

if gdf.crs is None:
    raise RuntimeError("Input layer has no CRS.")


# ============================================================
# CLEAN INPUT
# ============================================================

gdf = gdf[
    gdf.geometry.notna()
].copy()

gdf = gdf[
    ~gdf.geometry.is_empty
].copy()

gdf["length_m"] = gdf.geometry.length

gdf = gdf[
    gdf["length_m"] >= MIN_SEGMENT_LENGTH
].copy()

gdf = gdf.reset_index(drop=True)

print(
    "Segments after removing short fragments:",
    len(gdf)
)

if len(gdf) == 0:
    raise RuntimeError(
        "No valid rail segments remain."
    )


# ============================================================
# STORE SEGMENTS
# ============================================================

segments = list(gdf.geometry)

directions = [
    line_direction(line)
    for line in segments
]

endpoint_data = [
    local_endpoint_info(line)
    for line in segments
]


# ============================================================
# BUILD ENDPOINT CANDIDATES
# ============================================================

print("\nSearching for rail-continuation gaps...")
print("Maximum gap:", MAX_GAP, "m")
print("Maximum endpoint angle:", MAX_ENDPOINT_ANGLE, "degrees")

candidates = []

endpoint_names = ["start", "end"]

for i in range(len(segments)):

    for j in range(i + 1, len(segments)):

        # First make a cheap overall-direction test.
        if (
            directions[i] is None
            or directions[j] is None
        ):
            continue

        overall_angle = angle_difference(
            directions[i],
            directions[j]
        )

        if overall_angle > MAX_ANGLE_DIFF:
            continue

        for name_i in endpoint_names:
            for name_j in endpoint_names:

                p1 = endpoint_data[i][name_i]
                p2 = endpoint_data[j][name_j]

                tangent1 = endpoint_data[i][
                    name_i + "_tangent"
                ]

                tangent2 = endpoint_data[j][
                    name_j + "_tangent"
                ]

                valid, distance, bridge_angle, a1, a2 = (
                    test_endpoint_connection(
                        p1,
                        tangent1,
                        p2,
                        tangent2
                    )
                )

                if not valid:
                    continue

                if distance > MAX_GAP:
                    continue

                candidates.append({
                    "distance": distance,
                    "i": i,
                    "j": j,
                    "endpoint_i": name_i,
                    "endpoint_j": name_j,
                    "p1": p1,
                    "p2": p2,
                    "angle1": a1,
                    "angle2": a2,
                    "overall_angle": overall_angle,
                    "bridge_angle": bridge_angle,
                })


print(
    "Potential rail-continuation connections:",
    len(candidates)
)


# ============================================================
# SORT BEST CONNECTIONS FIRST
# ============================================================

# Prefer:
# 1. shortest gap
# 2. best local directional agreement
candidates.sort(
    key=lambda c: (
        c["distance"],
        c["angle1"] + c["angle2"]
    )
)


# ============================================================
# UNION-FIND
# ============================================================

parent = list(range(len(segments)))


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]

    return x


def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a
        return True

    return False


# ============================================================
# ACCEPT CONNECTIONS
# ============================================================

accepted_connections = []

# Each endpoint is allowed only one connection.
# This prevents branches / rail-to-multiple-rail connections.
used_endpoints = set()

for candidate in candidates:

    i = candidate["i"]
    j = candidate["j"]

    endpoint_i = candidate["endpoint_i"]
    endpoint_j = candidate["endpoint_j"]

    key_i = (i, endpoint_i)
    key_j = (j, endpoint_j)

    if key_i in used_endpoints:
        continue

    if key_j in used_endpoints:
        continue

    root_i = find(i)
    root_j = find(j)

    if root_i == root_j:
        continue

    if union(i, j):

        accepted_connections.append(
            candidate
        )

        used_endpoints.add(key_i)
        used_endpoints.add(key_j)


print(
    "Accepted rail-continuation connections:",
    len(accepted_connections)
)


# ============================================================
# CONNECTION STATISTICS
# ============================================================

if accepted_connections:

    gaps = [
        c["distance"]
        for c in accepted_connections
    ]

    print(
        "Shortest accepted gap:",
        round(min(gaps), 3),
        "m"
    )

    print(
        "Longest accepted gap:",
        round(max(gaps), 3),
        "m"
    )

    print(
        "Average accepted gap:",
        round(float(np.mean(gaps)), 3),
        "m"
    )


# ============================================================
# BUILD CONNECTED GROUPS
# ============================================================

groups = {}

for i in range(len(segments)):

    root = find(i)

    if root not in groups:
        groups[root] = []

    groups[root].append(i)


print(
    "Connected groups:",
    len(groups)
)


# ============================================================
# CREATE BRIDGES
# ============================================================

bridge_lines = []

for connection in accepted_connections:

    p1 = connection["p1"]
    p2 = connection["p2"]

    bridge = LineString([
        tuple(p1),
        tuple(p2)
    ])

    bridge_lines.append(bridge)


# ============================================================
# MERGE EACH GROUP
# ============================================================

repaired_lines = []

for root, indices in groups.items():

    group_lines = [
        segments[i]
        for i in indices
    ]

    group_bridges = []

    for connection in accepted_connections:

        i = connection["i"]
        j = connection["j"]

        if (
            find(i) == root
            and find(j) == root
        ):

            group_bridges.append(
                LineString([
                    tuple(connection["p1"]),
                    tuple(connection["p2"])
                ])
            )

    all_lines = (
        group_lines +
        group_bridges
    )

    merged = linemerge(
        MultiLineString(all_lines)
    )

    if merged.geom_type == "LineString":

        repaired_lines.append(
            merged
        )

    elif merged.geom_type == "MultiLineString":

        for line in merged.geoms:

            if line.length >= MIN_SEGMENT_LENGTH:
                repaired_lines.append(line)


print(
    "\nLines after rail-aware merging:",
    len(repaired_lines)
)


# ============================================================
# SMOOTH / SIMPLIFY
# ============================================================

print("\nApplying geometry smoothing...")

smoothed_lines = []

for line in repaired_lines:

    if line.length < MIN_SEGMENT_LENGTH:
        continue

    smooth = line.simplify(
        SMOOTH_TOLERANCE,
        preserve_topology=False
    )

    if smooth.geom_type == "LineString":

        if smooth.length >= MIN_SEGMENT_LENGTH:

            smoothed_lines.append(
                smooth
            )


print(
    "Final repaired lines:",
    len(smoothed_lines)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_gdf = gpd.GeoDataFrame(
    {
        "rail_id": range(
            1,
            len(smoothed_lines) + 1
        ),
        "length_m": [
            line.length
            for line in smoothed_lines
        ],
        "geometry": smoothed_lines
    },
    crs=gdf.crs
)

output_gdf.to_file(
    OUTPUT_GPKG,
    layer=OUTPUT_LAYER,
    driver="GPKG"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("RAIL-AWARE REPAIR COMPLETED")
print("=" * 75)

print(
    "\nOutput:",
    OUTPUT_GPKG
)

print(
    "Layer:",
    OUTPUT_LAYER
)

print(
    "Number of final lines:",
    len(output_gdf)
)

print(
    "Total rail length:",
    round(
        output_gdf["length_m"].sum(),
        2
    ),
    "m"
)

print("\nParameters:")
print(
    "MIN_SEGMENT_LENGTH:",
    MIN_SEGMENT_LENGTH,
    "m"
)

print(
    "MAX_GAP:",
    MAX_GAP,
    "m"
)

print(
    "MAX_ANGLE_DIFF:",
    MAX_ANGLE_DIFF,
    "degrees"
)

print(
    "MAX_ENDPOINT_ANGLE:",
    MAX_ENDPOINT_ANGLE,
    "degrees"
)

print(
    "SMOOTH_TOLERANCE:",
    SMOOTH_TOLERANCE,
    "m"
)

print("\nDone.")
