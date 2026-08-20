import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    input_path = Path(
        "results/tables/multiseed_sample_size_summary.csv"
    )

    output_path = Path(
        "results/figures/multiseed_sample_size.png"
    )

    samples = []
    f1_mean = []
    f1_std = []
    shd_mean = []
    shd_std = []
    exact_rate = []

    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            samples.append(int(row["n_samples"]))
            f1_mean.append(float(row["f1_mean"]))
            f1_std.append(float(row["f1_std"]))
            shd_mean.append(float(row["shd_mean"]))
            shd_std.append(float(row["shd_std"]))
            exact_rate.append(
                float(row["exact_recovery_rate"])
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 10),
    )

    # F1
    axes[0].errorbar(
        samples,
        f1_mean,
        yerr=f1_std,
        marker="o",
        capsize=4,
    )

    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Skeleton F1")
    axes[0].set_title(
        "Skeleton F1 vs Sample Size"
    )
    axes[0].grid(True, alpha=0.3)

    # SHD
    axes[1].errorbar(
        samples,
        shd_mean,
        yerr=shd_std,
        marker="o",
        capsize=4,
    )

    axes[1].set_xscale("log")
    axes[1].set_ylabel("CPDAG SHD")
    axes[1].set_title(
        "CPDAG Structural Hamming Distance vs Sample Size"
    )
    axes[1].grid(True, alpha=0.3)

    # Exact recovery
    axes[2].plot(
        samples,
        exact_rate,
        marker="o",
    )

    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Number of Samples")
    axes[2].set_ylabel("Exact Recovery Rate")
    axes[2].set_title(
        "Exact CPDAG Recovery Rate vs Sample Size"
    )
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Multi-Seed Causal Discovery Performance",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Figure saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
