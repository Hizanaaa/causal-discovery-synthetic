import csv
from pathlib import Path

import numpy as np


def main():
    input_path = Path(
        "results/tables/multiseed_sample_size_raw.csv"
    )

    output_path = Path(
        "results/tables/multiseed_sample_size_summary.csv"
    )

    rows = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "n_samples": int(row["n_samples"]),
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                    "shd": float(row["shd"]),
                    "exact_match": row["exact_match"] == "True",
                    "runtime": float(row["runtime"]),
                }
            )

    sample_sizes = sorted(
        {row["n_samples"] for row in rows}
    )

    summary = []

    for n_samples in sample_sizes:
        subset = [
            row
            for row in rows
            if row["n_samples"] == n_samples
        ]

        f1 = np.array([row["f1"] for row in subset])
        shd = np.array([row["shd"] for row in subset])
        exact = np.array(
            [row["exact_match"] for row in subset],
            dtype=float,
        )
        runtime = np.array(
            [row["runtime"] for row in subset]
        )

        summary.append(
            {
                "n_samples": n_samples,
                "f1_mean": f1.mean(),
                "f1_std": f1.std(ddof=1),
                "shd_mean": shd.mean(),
                "shd_std": shd.std(ddof=1),
                "exact_recovery_rate": exact.mean(),
                "runtime_mean": runtime.mean(),
                "runtime_std": runtime.std(ddof=1),
            }
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
                "f1_mean",
                "f1_std",
                "shd_mean",
                "shd_std",
                "exact_recovery_rate",
                "runtime_mean",
                "runtime_std",
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    print("=" * 80)
    print("MULTI-SEED SAMPLE-SIZE SUMMARY")
    print("=" * 80)

    print(
        f"{'Samples':>8} "
        f"{'F1 mean':>10} "
        f"{'F1 std':>9} "
        f"{'SHD mean':>10} "
        f"{'SHD std':>9} "
        f"{'Exact rate':>12} "
        f"{'Runtime':>10}"
    )

    print("-" * 80)

    for row in summary:
        print(
            f"{row['n_samples']:>8} "
            f"{row['f1_mean']:>10.3f} "
            f"{row['f1_std']:>9.3f} "
            f"{row['shd_mean']:>10.3f} "
            f"{row['shd_std']:>9.3f} "
            f"{row['exact_recovery_rate']:>12.3f} "
            f"{row['runtime_mean']:>10.4f}"
        )

    print(
        f"\nSummary saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
