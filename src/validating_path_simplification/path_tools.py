import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def visualize_paths(path: np.ndarray, simplified_path: np.ndarray):
    """
    Use matplotlib to visualize both the original and the simplified
    paths overlayed onto one another.

    Args:
        path: Array with shape (n,3).
        simplified_path: Array with shape (m,3) (not necessarily the
            same number of points as the original path).
    """
    # Create a subplot that can visualize 3d data
    figure = plt.figure()
    subplot = figure.add_subplot(projection="3d")

    # Guard against empty ndarrays.
    if len(path) > 0:
        path_t = path.T
        subplot.plot(path_t[0], path_t[1], path_t[2], "o-", color="black")

    if len(simplified_path) > 0:
        simplified_path_t = simplified_path.T
        subplot.plot(
            simplified_path_t[0],
            simplified_path_t[1],
            simplified_path_t[2],
            "o-",
            color="red",
        )

    subplot.set_xlabel("X-axis")
    subplot.set_ylabel("Y-axis")
    subplot.set_zlabel("Z-axis")
    plt.title("Original Path (Black) and Simplified Path (Red)")

    plt.show()


def simplify_path(points: np.ndarray, epsilon: float) -> list[int]:
    """
    Take a list of 3d points and return the indexes which should
    be included in the simplified path.
    """
    # Catch some degenerate cases.
    if len(points) == 0:
        return []
    elif len(points) == 1:
        return [0]

    return simplify_path_rdp(points, 0, len(points), epsilon)


def simplify_path_rdp(
    points: np.ndarray, start: int, end: int, epsilon: float
) -> list[int]:
    """
    Simplify the path using the Ramer-Douglas-Peucker algorithm
    described here: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

    It's usually better to call `simplify_path()` instead of calling
    this function directly since it has a few additional safety checks
    for points arrays of length 0 or 1. This function assumes that
    the points array has at least 2 elements.

    Args:
        points: Point array of shape (n,3).
        start: Starting index.
        end: Ending index + 1. Fine to use len() for this input if running
            on the entire point array.
        epsilon: The distance from the line segment below which a point
            will be removed in the simplification.
    """
    assert len(points) > 2
    assert epsilon > 0

    max_distance = 0
    max_distance_index = None

    # These points define the line segment we're checking against.
    q1 = points[start]
    q2 = points[end - 1]

    # Check distance of all points in the range from the line segment.
    for i in range(start + 1, end - 1):
        d = point_segment_distance(points[i], q1, q2)
        if d > max_distance:
            max_distance = d
            max_distance_index = i

    if max_distance > epsilon and max_distance_index is not None:
        left = simplify_path_rdp(points, start, max_distance_index + 1, epsilon)
        right = simplify_path_rdp(points, max_distance_index, end, epsilon)
        # TODO: Verify that this join doesn't add excessively more runtime

        return left + right[1:]
    else:
        return [start, end - 1]


def point_line_distance(p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Point to line distance based on https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_vector_formulation

    All points expected to have shape (3,).

    Args:
        p: A 3d point.
        q1: One of two points that defines a line.
        q2: The other of two points that defines a line.

    Returns:
        The shortest distance between `p` and the line passing
        through `q1` and `q2`.

    Raises:
        ValueError: q1 and q2 must be different points to define a unique line.
    """
    if (q1 == q2).all():
        raise ValueError("q1 and q2 must be different points to define a unique line.")

    line_direction = q2 - q1
    line_to_point = q1 - p

    return np.abs(
        np.linalg.norm(np.cross(line_direction, line_to_point))
        / np.linalg.norm(line_direction)
    )


def point_segment_distance(p: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """
    As discussed in the README, point-line distance and point-segment
    distance are slightly different, and which one we choose alters
    the way paths are simplified. In point-segment distance, the
    part of the segment closest to `p` could be either endpoint of
    the segment, or it could be another point on the segment. We have
    to account for each of these cases separately.

    All points expected to have shape (3,).

    Args:
        p: A 3d point.
        q1: One end of a line segment.
        q2: The other end of a line segment.

    Returns:
        The shortest distance between `p` and the line segment formed
        by `q1` and `q2`.
    """
    # In the degenerate case where q1 == q2, we're basically dealing
    # with a segment of length 0. So, we can calculate the distance
    # between p and q1 and be done.
    if (q1 == q2).all():
        return float(np.linalg.norm(q1 - p))

    # Get the vector from q1 to q2
    v_segment = q2 - q1
    v_q1_to_p = p - q1
    v_q2_to_p = p - q2

    # If the dot product of v_segment and the vector from q2 to p
    # is positive, the angle between v_segment and v_q2_to_p is
    # less than 90 degrees. That means v_q2_to_p is pointing in a
    # similar direction to v_segment, and p lies closest to q2.
    if np.sum(v_segment * v_q2_to_p) > 0:
        return float(np.linalg.norm(q2 - p))
    # If the dot product of v_segment and the vector from q1 to p
    # is negative, the angle between v_segment and v_q1_to_p is
    # greater than 90 degrees. That means p lies outside of the
    # segment and closest to q1.
    elif np.sum(v_segment * v_q1_to_p) < 0:
        return float(np.linalg.norm(q1 - p))
    # Otherwise, p is closest to the line between the points. So,
    # we need to rely on the point line distance, instead.
    else:
        return point_line_distance(p, q1, q2)


def build_3d_rotation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Build a rotation matrix R which rotates a 3d vector by tx, ty, and tz.

    Expected usage is `R @ x`, where `x` is the vector to rotate.

    Args:
        tx: Angle to rotate about x axis (in radians).
        ty: Angle to rotate about y axis (in radians).
        tz: Angle to rotate about z axis (in radians).

    Returns:
        A 3x3 rotation matrix which is the combination of each
        rotation around the x, y, and z axes.
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(tx), -math.sin(tx)],
            [0, math.sin(tx), math.cos(tx)],
        ]
    )
    R_y = np.array(
        [
            [math.cos(ty), 0, math.sin(ty)],
            [0, 1, 0],
            [-math.sin(ty), 0, math.cos(ty)],
        ]
    )
    R_z = np.array(
        [
            [math.cos(tz), -math.sin(tz), 0],
            [math.sin(tz), math.cos(tz), 0],
            [0, 0, 1],
        ]
    )

    return R_z @ R_y @ R_x


def generate_random_path(
    n: int, scale: float = 1.0, seed: int | None = None
) -> np.ndarray:
    """
    Generates totally random 3D points.

    This is the easiest way to generate a path, but it isn't very
    representative of real-world data.
    """
    rng = np.random.default_rng(seed)
    return scale * rng.random((n, 3))


def generate_constrained_random_path(
    n: int, max_angle: float = math.pi / 4, seed: int | None = None
) -> np.ndarray:
    """
    Build a path that looks a little bit more like the path a person
    would take by making the next point in the list move in roughly the
    same direction as the previous two points.
    """
    rng = np.random.default_rng(seed)

    if n == 0:
        return np.array([])
    if n == 1:
        return rng.random((1, 3))

    starting_point = rng.random(3)

    # Start with two random points one unit length away from each other.
    # Choose a random direction for the second point.
    tx, ty, tz = rng.random(3) * 2 * math.pi
    R = build_3d_rotation_matrix(tx, ty, tz)
    points = [starting_point, starting_point + (R @ np.array([1, 0, 0]))]

    for i in range(n - 2):
        tx, ty, tz = rng.random(3) * math.pi / 4
        R = build_3d_rotation_matrix(tx, ty, tz)

        prev_point = points[-1]
        prev_prev_point = points[-2]
        v = prev_point - prev_prev_point

        # So that the lengths of vectors have some variation,
        # scale v by a random factor between 0.5 and 1.5
        scaled_v = (rng.random(1) * 0.4 + 0.8) * v

        new_point = prev_point + R @ scaled_v
        points.append(new_point)

    return np.array(points)
