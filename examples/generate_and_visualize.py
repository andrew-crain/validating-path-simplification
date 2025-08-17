import numpy as np

from validating_path_simplification import path_tools


def hand_picked_path_example():
    path = np.array(
        [[0, 0, 0], [1, 1, 0], [2, 8, 0], [3, 10, 0], [5, 4, 0], [6, 6, 0], [8, 8, 0]],
        dtype=np.float64,
    )

    simplified_path_indexes = path_tools.simplify_path(path, epsilon=2.0)

    simplified_path = path[simplified_path_indexes]

    path_tools.visualize_paths(path, simplified_path)


def random_path_example():
    path = path_tools.generate_random_path(n=10, seed=5050)

    simplified_path_indexes = path_tools.simplify_path(path, epsilon=0.5)

    simplified_path = path[simplified_path_indexes]

    path_tools.visualize_paths(path, simplified_path)


def constrained_random_path_example():
    # All of these seeds used n=30 and the default max_angle
    # Interesting seeds: 1234, 9001
    # Lots of removal: 1337, 2345
    # Feel free to play around with other values.
    path = path_tools.generate_constrained_random_path(30, seed=9001)

    simplified_path_indexes = path_tools.simplify_path(path, epsilon=0.5)

    simplified_path = path[simplified_path_indexes]

    path_tools.visualize_paths(path, simplified_path)


def main():
    hand_picked_path_example()

    random_path_example()

    constrained_random_path_example()


if __name__ == "__main__":
    main()
