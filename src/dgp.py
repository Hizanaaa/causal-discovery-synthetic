import numpy as np
import networkx as nx


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

def generate_random_dag(
    n_nodes: int = 10,
    edge_probability: float = 0.2,
    seed: int = 42,
):
    """
    Generate a random DAG with controlled sparsity.

    A random topological ordering is created first. Edges are then
    sampled only from earlier nodes to later nodes, guaranteeing
    that the resulting graph is acyclic.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.

    edge_probability : float
        Probability of creating an edge between two nodes
        respecting the topological ordering.

    seed : int
        Random seed.

    Returns
    -------
    graph : nx.DiGraph
        Ground-truth DAG.

    weights : dict
        Mapping from (source, target) to structural coefficient.
    """

    rng = np.random.default_rng(seed)

    graph = nx.DiGraph()

    nodes = [f"X{i}" for i in range(1, n_nodes + 1)]
    graph.add_nodes_from(nodes)

    # Random topological ordering.
    ordering = list(nodes)
    rng.shuffle(ordering)

    weights = {}

    # Only create edges from earlier -> later in the ordering.
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if rng.random() < edge_probability:
                source = ordering[i]
                target = ordering[j]

                graph.add_edge(source, target)

                # Avoid near-zero coefficients.
                magnitude = rng.uniform(0.5, 2.0)
                sign = rng.choice([-1.0, 1.0])

                weights[(source, target)] = sign * magnitude

    return graph, weights


def simulate_linear_gaussian_sem(
    graph,
    weights,
    n_samples: int = 1000,
    noise_std: float = 1.0,
    seed: int = 42,
):
    """
    Generate data from a linear Gaussian structural equation model.

    For each node:

        X_i = sum(w_ij * X_j) + epsilon_i

    where epsilon_i ~ N(0, noise_std^2).

    Parameters
    ----------
    graph : nx.DiGraph
        Ground-truth DAG.

    weights : dict
        Mapping from (parent, child) to structural coefficient.

    n_samples : int
        Number of observations.

    noise_std : float
        Standard deviation of Gaussian noise.

    seed : int
        Random seed.

    Returns
    -------
    data : np.ndarray
        Shape (n_samples, n_nodes).

    node_names : list
        Column names corresponding to the data matrix.
    """

    rng = np.random.default_rng(seed)

    node_names = list(nx.topological_sort(graph))
    node_to_index = {
        node: index
        for index, node in enumerate(node_names)
    }

    data = np.zeros((n_samples, len(node_names)))

    for node in node_names:

        node_index = node_to_index[node]

        noise = rng.normal(
            0,
            noise_std,
            n_samples,
        )

        values = noise.copy()

        for parent in graph.predecessors(node):
            parent_index = node_to_index[parent]
            coefficient = weights[(parent, node)]

            values += coefficient * data[:, parent_index]

        data[:, node_index] = values

    return data, node_names