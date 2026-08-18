import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    input_path = Path("results/tables/sample_size_sweep.csv")
    output_path = Path("results/figures/sample_size_sweep.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Skeleton F1
    axes[0].plot(
        df["n_samples"],
        df["f1"],
        marker="o",
    )
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Skeleton F1")
    axes[0].set_title("Skeleton F1 vs Sample Size")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    # CPDAG SHD
    axes[1].plot(
        df["n_samples"],
        df["shd"],
        marker="o",
    )
    axes[1].set_xscale("log")
    axes[1].set_ylabel("CPDAG SHD")
    axes[1].set_title("CPDAG Structural Hamming Distance vs Sample Size")
    axes[1].grid(True, alpha=0.3)

    # Runtime
    axes[2].plot(
        df["n_samples"],
        df["runtime"],
        marker="o",
    )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Number of Samples")
    axes[2].set_ylabel("Runtime (seconds)")
    axes[2].set_title("PC Runtime vs Sample Size")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Causal Discovery Performance vs Sample Size",
        fontsize=14,
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
