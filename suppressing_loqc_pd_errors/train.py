from pathlib import Path
from typing import Callable

import numpy as np
import piquasso as pq
import tensorflow as tf
from piquasso.decompositions.clements import get_interferometer_from_weights
from tensorflow.python.keras.optimizers import Optimizer
from tqdm import tqdm


def calculate_loss(
    weights: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    state_vector: tf.Tensor,
    cutoff: int,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    S_star: tf.Tensor,
) -> tf.Tensor:
    d = 6
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np

    ancilla_modes = (4, 5)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    ancilla_state = [1, 1]

    interferometer = get_interferometer_from_weights(
        weights, d, calculator, config.complex_dtype
    )

    program = pq.Program(
        instructions=[
            pq.StateVector(state_00 + ancilla_state, coefficient=state_vector[0]),
            pq.StateVector(state_01 + ancilla_state, coefficient=state_vector[1]),
            pq.StateVector(state_10 + ancilla_state, coefficient=state_vector[2]),
            pq.StateVector(state_11 + ancilla_state, coefficient=state_vector[3]),
            pq.Interferometer(interferometer),
            pq.ImperfectPostSelectPhotons(
                postselect_modes=ancilla_modes,
                photon_counts=ancilla_state,
                detector_efficiency_matrix=P,
            ),
        ]
    )

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    state = simulator.execute(program).state

    density_matrix = state.density_matrix
    success_prob = np.real(np.trace(density_matrix))
    normalized_density_matrix = density_matrix / success_prob

    fidelity = np.real(np.trace(normalized_density_matrix @ expected_density_matrix))
    loss = (
        1
        - np.sqrt(fidelity)
        + alpha * np.log(1 + np.exp(-beta * (success_prob - S_star))) / beta
    )

    return loss, success_prob, fidelity


def train_step(
    weights: tf.Tensor,
    P: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    expected_density_matrix: tf.Tensor,
    state_vector: tf.Tensor,
    cutoff: int,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    S_star: tf.Tensor,
):
    with tf.GradientTape() as tape:
        loss, success_prob, fidelity = calculate_loss(
            weights=weights,
            P=P,
            calculator=calculator,
            expected_density_matrix=expected_density_matrix,
            state_vector=state_vector,
            cutoff=cutoff,
            alpha=alpha,
            beta=beta,
            S_star=S_star,
        )

    grad = tape.gradient(loss, weights)

    return loss, success_prob, fidelity, grad


def train(
    iterations: int,
    opt: Optimizer,
    _train_step: Callable,
    initial_weights: np.ndarray,
    P: tf.Tensor,
    state_vector: tf.Tensor,
    expected_density_matrix: tf.Tensor,
    calculator: pq.TensorflowCalculator,
    cutoff: int,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    S_star: tf.Tensor,
    output_dir: Path,
):
    weights = tf.Variable(initial_weights, dtype=tf.float64)
    checkpoint = tf.train.Checkpoint(weights=weights)

    output_dir.mkdir(exist_ok=True)

    lossfile = output_dir / f"losses.csv"
    with open(str(lossfile), "w") as f:
        f.write("iteration,loss,success_prob,fidelity,prob\n")

    for i in tqdm(range(iterations)):
        loss, success_prob, fidelity, grad = _train_step(
            weights=weights,
            P=P,
            calculator=calculator,
            expected_density_matrix=expected_density_matrix,
            state_vector=state_vector,
            cutoff=cutoff,
            alpha=alpha,
            beta=beta,
            S_star=S_star,
        )

        opt.apply_gradients(zip([grad], [weights]))

        with open(str(lossfile), "a+") as f:
            f.write(f"{i},{loss},{success_prob},{fidelity},{S_star}\n")

    checkpoint.save(str(output_dir / "weights"))
