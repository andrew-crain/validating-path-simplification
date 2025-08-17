import math

import numpy as np

from validating_path_simplification import path_tools


class TestPointLineDistance:
    def test_simple_2d_distance(self):
        """
        This test is easy to calculate by hand since all points
        are on the same 2d plane.
        """
        # Points that define a line segment.
        q1 = np.array([0, 0, 0])
        q2 = np.array([6, 0, 0])
        # Point from which to calculate distance to the line segment.
        p = np.array([3, 3, 0])

        expected_distance = 3.0

        assert math.isclose(
            path_tools.point_line_distance(p, q1, q2), expected_distance
        )

    def test_simple_3d_distance(self):
        """
        Extends the 2d case in a way that's still easy to calculate
        by hand.
        """
        # Points that define a line segment.
        q1 = np.array([0, 0, 0])
        q2 = np.array([6, 0, 0])
        # Point from which to calculate distance to the line segment.
        p = np.array([3, 3, 3])

        expected_distance = math.sqrt(3**2 + 3**2)

        assert math.isclose(
            path_tools.point_line_distance(p, q1, q2), expected_distance
        )


class TestPointSegmentDistance:
    def test_point_nearest_leftmost_end_point(self):
        q1 = np.array([0, 0, 0])
        q2 = np.array([6, 0, 0])
        p = np.array([-6, 1, 0])

        expected_distance = math.sqrt(6**2 + 1**2)

        assert math.isclose(
            path_tools.point_segment_distance(p, q1, q2), expected_distance
        )

    def test_point_nearest_rightmost_end_point(self):
        q1 = np.array([0, 0, 0])
        q2 = np.array([6, 0, 0])
        p = np.array([12, 1, 0])

        expected_distance = math.sqrt(6**2 + 1**2)

        assert math.isclose(
            path_tools.point_segment_distance(p, q1, q2), expected_distance
        )

    def test_point_nearest_line(self):
        q1 = np.array([0, 0, 0])
        q2 = np.array([6, 0, 0])
        p = np.array([3, 1, 0])

        expected_distance = 1.0

        assert math.isclose(
            path_tools.point_segment_distance(p, q1, q2), expected_distance
        )


class TestPathSimplification:
    def test_3_points_no_removal(self):
        path = np.array([[0, 0, 0], [5, 5, 5], [0, 0, 1]], dtype=np.float64)

        indexes = path_tools.simplify_path(path, epsilon=0.5)

        assert indexes == [0, 1, 2]

    def test_3_points_middle_removed(self):
        path = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float64)

        indexes = path_tools.simplify_path(path, epsilon=0.5)

        assert indexes == [0, 2]

    def test_5_points_3_middle_removed(self):
        path = np.array(
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=np.float64
        )

        indexes = path_tools.simplify_path(path, epsilon=0.5)

        assert indexes == [0, 4]

    def test_5_points_1_and_3_removed(self):
        path = np.array(
            [[0, 0, 0], [2, 2, 1.8], [4, 4, 4], [4, 2, 1.8], [4, 0, 0]],
            dtype=np.float64,
        )

        indexes = path_tools.simplify_path(path, epsilon=0.5)

        assert indexes == [0, 2, 4]
