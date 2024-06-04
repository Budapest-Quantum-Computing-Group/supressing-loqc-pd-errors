import numpy as np
import piquasso as pq


def get_expected_density_matrix(
    state_vector: np.ndarray, cutoff: int, calculator: pq.TensorflowCalculator
) -> np.ndarray:
    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    config = pq.Config(normalize=False, cutoff=cutoff)
    expected_program = pq.Program(
        instructions=[
            pq.StateVector(state_00, coefficient=state_vector[0]),
            pq.StateVector(state_01, coefficient=state_vector[1]),
            pq.StateVector(state_10, coefficient=state_vector[2]),
            pq.StateVector(state_11, coefficient=-state_vector[3]),
        ]
    )

    simulator = pq.PureFockSimulator(d=4, config=config, calculator=calculator)
    expected_state = simulator.execute(expected_program).state

    return expected_state.density_matrix
