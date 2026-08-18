import time
import csv

from pathlib import Path

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


def run_experiment(
    true_dag,
    weights,
    true_cpdag,
    n_samples,
    alpha=0.05,
    seed=42,
):
    data, node_names = simulate_linear_gaussian_sem(
        true_dag,
        weights,
        n_samples=n_samples,
        seed=seed,
    )

    start = time.perf_counter()

    learned = pc(
        data,
        alpha=alpha,
        indep_test="fisherz",
        stable=True,
        node_names=node_names,
        show_progress=False,
    )

    runtime = time.perf_counter() - start

    true_skeleton = {
        frozenset(edge)
        for edge in true_dag.edges()
    }

    learned_skeleton = causal_learn_skeleton(
        learned.G
    )

    precision, recall, f1 = (
        skeleton_precision_recall_f1(
            true_skeleton,
            learned_skeleton,
        )
    )

    shd = cpdag_structural_hamming_distance(
        true_cpdag.G,
        learned.G,
    )

    true_edges = set(
        causal_learn_edges(true_cpdag.G)
    )

    learned_edges = set(
        causal_learn_edges(learned.G)
    )

    exact_match = (
        true_edges == learned_edges
    )

    return {
        "n_samples": n_samples,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "shd": shd,
        "exact_match": exact_match,
        "runtime": runtime,
    }


def main():
    n_nodes = 10
    edge_probability = 0.2
    dag_seed = 42
    alpha = 0.05

    sample_sizes = [
        100,
        250,
        500,
        1000,
        2000,
        5000,
    ]

    true_dag, weights = generate_random_dag(
        n_nodes=n_nodes,
        edge_probability=edge_probability,
        seed=dag_seed,
    )

    node_names = list(true_dag.nodes())

    true_cpdag = dag_to_cpdag(
        list(true_dag.edges()),
        node_names,
    )

    print("=" * 70)
    print("SAMPLE-SIZE SWEEP")
    print("=" * 70)

    print(f"Nodes:              {n_nodes}")
    print(f"Edge probability:   {edge_probability}")
    print(f"DAG seed:           {dag_seed}")
    print(f"Alpha:              {alpha}")
    print()

    results = []

    for n_samples in sample_sizes:

        print(f"Running n = {n_samples}...")

        result = run_experiment(
            true_dag,
            weights,
            true_cpdag,
            n_samples=n_samples,
            alpha=alpha,
            seed=dag_seed,
        )

        results.append(result)

    print()
    print(
        f"{'Samples':>8} "
        f"{'Prec.':>8} "
        f"{'Recall':>8} "
        f"{'F1':>8} "
        f"{'SHD':>8} "
        f"{'Exact':>8} "
        f"{'Runtime':>10}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['n_samples']:>8} "
            f"{result['precision']:>8.3f} "
            f"{result['recall']:>8.3f} "
            f"{result['f1']:>8.3f} "
            f"{result['shd']:>8} "
            f"{str(result['exact_match']):>8} "
            f"{result['runtime']:>10.4f}"
        )

    # Save results to CSV
    output_path = Path("results/tables/sample_size_sweep.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_samples",
                "precision",
                "recall",
                "f1",
                "shd",
                "exact_match",
                "runtime",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()