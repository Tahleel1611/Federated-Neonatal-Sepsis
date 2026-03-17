from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    base = Path("results")
    sweep_path = base / "dp_sweep_summary.csv"
    no_dp_path = base / "fl_no_dp_rounds.csv"
    out_path = base / "privacy_frontier_dual_axis.png"

    sweep = pd.read_csv(sweep_path).sort_values("epsilon_cumulative")
    no_dp = pd.read_csv(no_dp_path)

    no_dp_best = no_dp.loc[no_dp["tuned_specificity"].idxmax()]
    no_dp_sens = float(no_dp_best["tuned_sensitivity"])
    no_dp_spec = float(no_dp_best["tuned_specificity"])

    x = sweep["epsilon_cumulative"].to_list()
    sens = sweep["tuned_sensitivity"].to_list()
    spec = sweep["tuned_specificity"].to_list()
    labels = sweep["noise_multiplier"].to_list()

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    l1 = ax1.plot(x, sens, marker="o", linewidth=2.2, color="#1f77b4", label="Sensitivity")
    l2 = ax2.plot(x, spec, marker="s", linewidth=2.2, color="#d62728", label="Specificity")

    ax1.axhline(0.80, color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.axhline(0.40, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)

    for xi, yi, nm in zip(x, sens, labels):
        ax1.annotate(f"NM={nm}", (xi, yi), textcoords="offset points", xytext=(6, 8), fontsize=9)

    ax1.scatter([0.0], [no_dp_sens], marker="*", s=200, color="#1f77b4", alpha=0.85)
    ax2.scatter([0.0], [no_dp_spec], marker="*", s=200, color="#d62728", alpha=0.85)

    ax1.set_title("DP Scissor Effect: Sensitivity Robustness vs Specificity Cliff")
    ax1.set_xlabel("Cumulative Epsilon")
    ax1.set_ylabel("Sensitivity", color="#1f77b4")
    ax2.set_ylabel("Specificity", color="#d62728")

    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_ylim(0.75, 0.85)
    ax2.set_ylim(0.0, 0.60)
    ax1.grid(alpha=0.25)

    lines = l1 + l2
    labels_legend = [line.get_label() for line in lines]
    ax1.legend(lines, labels_legend, loc="center right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()