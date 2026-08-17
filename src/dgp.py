import numpy as np


def generate_chain_data(
    n_samples: int = 1000,
    seed: int = 42,
):
    """
    Generate data from:

        A -> B -> C
    """

    rng = np.random.default_rng(seed)

    eps_A = rng.normal(0, 1, n_samples)
    eps_B = rng.normal(0, 1, n_samples)
    eps_C = rng.normal(0, 1, n_samples)

    A = eps_A
    B = 1.5 * A + eps_B
    C = 1.2 * B + eps_C

    return np.column_stack([A, B, C])


def generate_fork_data(
    n_samples: int = 1000,
    seed: int = 42,
):
    """
    Generate data from:

        A <- B -> C
    """

    rng = np.random.default_rng(seed)

    eps_A = rng.normal(0, 1, n_samples)
    eps_B = rng.normal(0, 1, n_samples)
    eps_C = rng.normal(0, 1, n_samples)

    B = eps_B
    A = 1.5 * B + eps_A
    C = 1.2 * B + eps_C

    return np.column_stack([A, B, C])


def generate_collider_data(
    n_samples: int = 1000,
    seed: int = 42,
):
    """
    Generate data from:

        A -> B <- C
    """

    rng = np.random.default_rng(seed)

    eps_A = rng.normal(0, 1, n_samples)
    eps_B = rng.normal(0, 1, n_samples)
    eps_C = rng.normal(0, 1, n_samples)

    A = eps_A
    C = eps_C
    B = 1.5 * A + 1.2 * C + eps_B

    return np.column_stack([A, B, C])