import time

from causallearn.search.ConstraintBased.PC import pc

from src.dgp import (
    generate_chain_data,
    generate_fork_data,
    generate_collider_data,
)

from src.metrics import (
    causal_learn_edges,
    causal_learn_skeleton,
    skeleton_precision_recall_f1,
    orientation_accuracy,
)


def run_pc(data, name, true_edges):
    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")

    start_time = time.perf_counter()

    cg = pc(
        data,
        alpha=0.05,
        indep_test="fisherz",
        stable=True,
    )

    runtime = time.perf_counter() - start_time

    edges = causal_learn_edges(cg.G)
    predicted_skeleton = causal_learn_skeleton(cg.G)

    precision, recall, f1 = skeleton_precision_recall_f1(
        true_edges,
        predicted_skeleton,
    )

    orientation_acc = orientation_accuracy(
        true_edges,
        edges,
    )

    print("\nRecovered edges:")
    for edge in edges:
        print(edge)

    print("\nMetrics:")
    print(f"Skeleton Precision: {precision:.3f}")
    print(f"Skeleton Recall:    {recall:.3f}")
    print(f"Skeleton F1:        {f1:.3f}")
    print(f"Orientation Accuracy: {orientation_acc:.3f}")
    print(f"Runtime:            {runtime:.6f} seconds")


def main():

    run_pc(
        generate_chain_data(n_samples=1000, seed=42),
        "CHAIN: A -> B -> C",
        [("X1", "X2"), ("X2", "X3")],
    )

    run_pc(
        generate_fork_data(n_samples=1000, seed=42),
        "FORK: A <- B -> C",
        [("X1", "X2"), ("X2", "X3")],
    )

    run_pc(
        generate_collider_data(n_samples=1000, seed=42),
        "COLLIDER: A -> B <- C",
        [("X1", "X2"), ("X3", "X2")],
    )


if __name__ == "__main__":
    main()