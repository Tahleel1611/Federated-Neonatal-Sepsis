from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    base = Path("results")
    sweep_path = base / "dp_sweep_summary.csv"
    no_dp_path = base / "fl_no_dp_rounds.csv"
    out_path = base / "privacy_frontier_spec_vs_epsilon.png"

    sweep = pd.read_csv(sweep_path).sort_values("epsilon_cumulative")
    no_dp = pd.read_csv(no_dp_path)

    no_dp_best = no_dp.loc[no_dp["tuned_specificity"].idxmax()]
    no_dp_spec = float(no_dp_best["tuned_specificity"])
    no_dp_round = int(no_dp_best["round"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.axhspan(0.40, 1.0, color="#dff3e3", alpha=0.6, label="Clinical Safety Zone (Spec > 0.40)")
    ax.axhline(0.40, color="green", linestyle="--", linewidth=1.5)

    x = sweep["epsilon_cumulative"].to_list()
    y = sweep["tuned_specificity"].to_list()
    ax.plot(x, y, marker="o", color="#d62728", linewidth=2, label="DP Sweep")

    for _, row in sweep.iterrows():
        ax.annotate(
            f"NM={row['noise_multiplier']}",
            (row["epsilon_cumulative"], row["tuned_specificity"]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=9,
        )

    ax.scatter([0.0], [no_dp_spec], marker="*", s=220, color="#1f77b4", label=f"No-DP Best (R{no_dp_round})")

    ax.set_title("Privacy-Utility Frontier: Specificity vs Cumulative Epsilon")
    ax.set_xlabel("Cumulative Epsilon")
    ax.set_ylabel("Tuned Specificity")
    ax.set_ylim(0.0, max(0.60, max(y) + 0.05))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()