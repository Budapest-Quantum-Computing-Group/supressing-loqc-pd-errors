import numpy as np
import piquasso as pq
from piquasso._math.indices import get_operator_index
from piquasso.decompositions.clements import get_weights_from_interferometer


def get_initial_weights():
    calculator = pq.NumpyCalculator()
    modes = (0, 2, 4, 5)

    U = np.array(
        [
            [-1 / 3, -np.sqrt(2) / 3, np.sqrt(2) / 3, 2 / 3],
            [np.sqrt(2) / 3, -1 / 3, -2 / 3, np.sqrt(2) / 3],
            [
                -np.sqrt(3 + np.sqrt(6)) / 3,
                np.sqrt(3 - np.sqrt(6)) / 3,
                -np.sqrt((3 + np.sqrt(6)) / 2) / 3,
                np.sqrt(1 / 6 - 1 / (3 * np.sqrt(6))),
            ],
            [
                -np.sqrt(3 - np.sqrt(6)) / 3,
                -np.sqrt(3 + np.sqrt(6)) / 3,
                -np.sqrt(1 / 6 - 1 / (3 * np.sqrt(6))),
                -np.sqrt((3 + np.sqrt(6)) / 2) / 3,
            ],
        ]
    )

    V = calculator.embed_in_identity(U, get_operator_index(modes), 6)

    return get_weights_from_interferometer(V, calculator)
