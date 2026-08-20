import time
import csv
from pathlib import Path

import numpy as np
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
    seed,
    alpha=0.05,
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

    precision, recall, f1 = skeleton_precision_recall_f1(
        true_skeleton,
        learned_skeleton,
    )

    shd = cpdag_structural_hamming_distance(
        true_cpdag.G,
        learned.G,
    )

    true_edges = set(causal_learn_edges(true_cpdag.G))
    learned_edges = set(causal_learn_edges(learned.G))

    exact_match = true_edges == learned_edges

    return {
        "n_samples": n_samples,
        "seed": seed,
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

    data_seeds = [
        42,
        43,
        44,
        45,
        46,
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

    results = []

    print("=" * 70)
    print("MULTI-SEED SAMPLE-SIZE SWEEP")
    print("=" * 70)

    print(f"Nodes:              {n_nodes}")
    print(f"Edge probability:   {edge_probability}")
    print(f"DAG seed:           {dag_seed}")
    print(f"Data seeds:         {data_seeds}")
    print(f"Alpha:              {alpha}")
    print()

    for n_samples in sample_sizes:

        print(f"Running n = {n_samples}...")

        for seed in data_seeds:

            result = run_experiment(
                true_dag,
                weights,
                true_cpdag,
                n_samples=n_samples,
                seed=seed,
                alpha=alpha,
            )

            results.append(result)

    print()
    print(
        f"{'Samples':>8} "
        f"{'F1 mean':>10} "
        f"{'F1 std':>9} "
        f"{'SHD mean':>10} "
        f"{'SHD std':>9} "
        f"{'Exact rate':>12} "
        f"{'Runtime':>10}"
    )

    print("-" * 75)

    for n_samples in sample_sizes:

        subset = [
            r for r in results
            if r["n_samples"] == n_samples
        ]

        f1_values = np.array(
            [r["f1"] for r in subset]
        )

        shd_values = np.array(
            [r["shd"] for r in subset]
        )

        exact_values = np.array(
            [r["exact_match"] for r in subset],
            dtype=float,
        )

        runtime_values = np.array(
            [r["runtime"] for r in subset]
        )

        print(
            f"{n_samples:>8} "
            f"{f1_values.mean():>10.3f} "
            f"{f1_values.std(ddof=1):>9.3f} "
            f"{shd_values.mean():>10.3f} "
            f"{shd_values.std(ddof=1):>9.3f} "
            f"{exact_values.mean():>12.3f} "
            f"{runtime_values.mean():>10.4f}"
        )

    output_path = Path(
        "results/tables/multiseed_sample_size_raw.csv"
    )
    
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_samples",
                "seed",
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

    print(
        f"\nRaw results saved to: {output_path}"
    )

if __name__ == "__main__":
    main()
