from dataclasses import dataclass
from pathlib import Path

import piquasso as pq
import tensorflow as tf
import tyro

from suppressing_loqc_pd_errors.expected_density_matrix import (
    get_expected_density_matrix,
)
from suppressing_loqc_pd_errors.initial_weights import get_initial_weights
from suppressing_loqc_pd_errors.pd_efficiency_matrix import get_P
from suppressing_loqc_pd_errors.train import train, train_step

tf.get_logger().setLevel("ERROR")


@dataclass
class Args:
    output_dir: str
    "Path to the parent directory where to generate the losses.csv files"

    learning_rate: int = 0.00025
    "Learning rate for the Adam optimizer"

    starting_s_star: float = 0.05
    "Starting S^*"
    ending_s_star: float = 0.15
    "Ending S^* (exclusive)"

    step_size: float = 0.005
    "Step size for the values of S^*"

    iterations: int = 1000
    "Iterations for each train"

    alpha: float = 10
    "The alpha in the loss function"

    beta: float = 10_000
    "The beta in the loss function"

    enhanced: bool = False
    "If enabled the script uses tf.function"

    use_jit: bool = False
    "If enabled the script uses XLA"


def main():
    args = tyro.cli(Args)

    decorator = tf.function(jit_compile=args.use_jit) if args.enhanced else None
    calculator = pq.TensorflowCalculator(decorate_with=decorator)

    np = calculator.np

    cutoff = 5

    P = get_P(cutoff)
    P = tf.convert_to_tensor(P)

    initial_weights = get_initial_weights()

    state_vector = np.sqrt([1 / 4, 1 / 4, 1 / 4, 1 / 4])

    expected_density_matrix = get_expected_density_matrix(
        state_vector, cutoff, calculator
    )
    expected_density_matrix = tf.convert_to_tensor(expected_density_matrix)

    _train_step = decorator(train_step) if decorator is not None else train_step

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    S_star = args.starting_s_star
    while S_star < args.ending_s_star:
        opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        kwargs = {
            "iterations": args.iterations,
            "opt": opt,
            "_train_step": _train_step,
            "initial_weights": initial_weights.copy(),
            "P": P,
            "expected_density_matrix": expected_density_matrix,
            "state_vector": state_vector,
            "calculator": calculator,
            "cutoff": cutoff,
            "alpha": tf.convert_to_tensor(args.alpha, dtype=tf.float64),
            "beta": tf.convert_to_tensor(args.beta, dtype=tf.float64),
            "S_star": tf.convert_to_tensor(S_star, dtype=tf.float64),
            "output_dir": output_dir / f"{S_star:.4f}",
        }

        train(
            iterations=args.iterations,
            opt=opt,
            _train_step=_train_step,
            initial_weights=initial_weights,
            P=P,
            expected_density_matrix=expected_density_matrix,
            state_vector=state_vector,
            calculator=calculator,
            cutoff=cutoff,
            alpha=tf.convert_to_tensor(args.alpha, dtype=tf.float64),
            beta=tf.convert_to_tensor(args.beta, dtype=tf.float64),
            S_star=tf.convert_to_tensor(S_star, dtype=tf.float64),
            output_dir=output_dir / f"{S_star:.4f}"
        )
        S_star += args.step_size

if __name__ == "__main__":
    main()
