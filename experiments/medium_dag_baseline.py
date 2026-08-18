import time

from causallearn.search.ConstraintBased.PC import pc

from src.dgp import (
    generate_random_dag,
    simulate_linear_gaussian_sem,
)

from src.metrics import (
    dag_to_cpdag,
    causal_learn_skeleton,
    skeleton_precision_recall_f1,
    cpdag_structural_hamming_distance,
    causal_learn_edges,
)


def main():
    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    n_nodes = 10
    n_samples = 1000
    edge_probability = 0.2
    alpha = 0.05
    seed = 42

    # ---------------------------------------------------------
    # Generate ground-truth DAG
    # ---------------------------------------------------------
    true_dag, weights = generate_random_dag(
        n_nodes=n_nodes,
        edge_probability=edge_probability,
        seed=seed,
    )

    node_names = list(true_dag.nodes())
    true_edges = list(true_dag.edges())

    # ---------------------------------------------------------
    # Generate observational data
    # ---------------------------------------------------------
    data, data_names = simulate_linear_gaussian_sem(
        true_dag,
        weights,
        n_samples=n_samples,
        seed=seed,
    )

    # ---------------------------------------------------------
    # Run PC
    # ---------------------------------------------------------
    start = time.perf_counter()

    learned = pc(
        data,
        alpha=alpha,
        indep_test="fisherz",
        stable=True,
        node_names=data_names,
        show_progress=True,
    )

    runtime = time.perf_counter() - start

    # ---------------------------------------------------------
    # Convert true DAG -> true CPDAG
    # ---------------------------------------------------------
    true_cpdag = dag_to_cpdag(
        true_edges,
        data_names,
    )

    # ---------------------------------------------------------
    # Extract learned skeleton
    # ---------------------------------------------------------
    learned_skeleton = causal_learn_skeleton(learned.G)

    true_skeleton = {
        frozenset(edge)
        for edge in true_edges
    }

    precision, recall, f1 = skeleton_precision_recall_f1(
        true_skeleton,
        learned_skeleton,
    )

    # ---------------------------------------------------------
    # CPDAG evaluation
    # ---------------------------------------------------------
    shd = cpdag_structural_hamming_distance(
        true_cpdag.G,
        learned.G,
    )

    true_cpdag_edges = set(
        causal_learn_edges(true_cpdag.G)
    )

    learned_cpdag_edges = set(
        causal_learn_edges(learned.G)
    )

    exact_match = (
        true_cpdag_edges == learned_cpdag_edges
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("MEDIUM DAG BASELINE")
    print("=" * 60)

    print(f"Nodes:              {n_nodes}")
    print(f"Samples:            {n_samples}")
    print(f"Edge probability:   {edge_probability}")
    print(f"Alpha:              {alpha}")
    print(f"Seed:               {seed}")

    print("\nGround-truth DAG edges:")
    for edge in true_edges:
        print(f"  {edge[0]} -> {edge[1]}")

    print("\nLearned CPDAG:")
    print(learned.G)

    print("\nMetrics:")
    print(f"Skeleton Precision: {precision:.3f}")
    print(f"Skeleton Recall:    {recall:.3f}")
    print(f"Skeleton F1:        {f1:.3f}")
    print(f"CPDAG SHD:          {shd}")
    print(f"Exact CPDAG Match:  {exact_match}")
    print(f"Runtime:            {runtime:.6f} seconds")


if __name__ == "__main__":
    main()
