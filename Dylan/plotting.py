"""
Plot comparison results from multiple experiment runs.

Usage:
    python plotting.py                              # default: results/comparison
    python plotting.py --log_dir results/comparison # explicit
    python plotting.py --window 50                  # smoothing window
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def discover_runs(log_dir):
    """Find all subdirectories containing a metrics.csv and label them."""
    runs = []
    for name in sorted(os.listdir(log_dir)):
        csv_path = os.path.join(log_dir, name, "metrics.csv")
        if os.path.isfile(csv_path):
            label = _make_label(name)
            runs.append({"name": name, "label": label, "csv": csv_path})
    return runs


def _make_label(run_name):
    """
    Derive a human-readable label from the run directory name.
    e.g. 'ppo_vs_random_1774473167' → 'ppo vs random'
    We also check for the experiment banner saved alongside metrics.
    """
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0].replace("_", " ")
    return run_name.replace("_", " ")


def load_run(csv_path, window=20):
    """Load a metrics CSV and add smoothed columns."""
    df = pd.read_csv(csv_path)
    for col in ["predator_reward", "prey_reward"]:
        if col in df.columns:
            df[f"{col}_smooth"] = df[col].rolling(window, min_periods=1).mean()
    pg_cols = [c for c in df.columns if c.endswith("_pg_loss")]
    if pg_cols:
        df["pg_loss_smooth"] = df[pg_cols[0]].rolling(window, min_periods=1).mean()
    v_cols = [c for c in df.columns if c.endswith("_v_loss")]
    if v_cols:
        df["v_loss_smooth"] = df[v_cols[0]].rolling(window, min_periods=1).mean()
    entropy_cols = [c for c in df.columns if c.endswith("_entropy")]
    if entropy_cols:
        df["entropy_smooth"] = df[entropy_cols[0]].rolling(window, min_periods=1).mean()
    return df


def enrich_labels(runs, log_dir):
    """
    Try to read the experiment banner from stdout logs or infer sharing/algo
    from the run directory structure to produce richer labels.
    """
    for run in runs:
        label_parts = []
        base = run["label"]
        df = pd.read_csv(run["csv"], nrows=1)

        cols = df.columns.tolist()
        has_pred0 = any("predator_0_" in c for c in cols)
        has_pred = any(c.startswith("predator_pg") for c in cols)

        if "mappo" in base:
            label_parts.append("MAPPO")
        elif "ppo" in base:
            label_parts.append("IPPO")
        else:
            label_parts.append(base.split(" vs ")[0].upper())

        if has_pred0 and not has_pred:
            # Column named predator_0_pg_loss → could be shared (label=predator_0)
            # or independent. Check if predator_1 columns exist.
            has_pred1 = any("predator_1_" in c for c in cols)
            if has_pred1:
                label_parts.append("independent")
            else:
                label_parts.append("shared")
        elif has_pred:
            label_parts.append("shared")
        else:
            label_parts.append("shared")

        run["rich_label"] = " / ".join(label_parts)
    return runs


def plot_comparison(runs, window, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison — 2 Predators vs 3 Random Prey", fontsize=14, y=0.98)

    colors = plt.cm.tab10.colors

    for i, run in enumerate(runs):
        df = load_run(run["csv"], window=window)
        color = colors[i % len(colors)]
        label = run.get("rich_label", run["label"])

        # 1) Predator reward
        ax = axes[0, 0]
        ax.plot(df["episode"], df["predator_reward_smooth"], label=label,
                color=color, linewidth=1.5)
        ax.fill_between(df["episode"], df["predator_reward"], alpha=0.08, color=color)
        ax.set_title("Predator Team Reward (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")

        # 2) Prey reward
        ax = axes[0, 1]
        ax.plot(df["episode"], df["prey_reward_smooth"], label=label,
                color=color, linewidth=1.5)
        ax.fill_between(df["episode"], df["prey_reward"], alpha=0.08, color=color)
        ax.set_title("Prey Team Reward (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")

        # 3) Policy loss
        ax = axes[1, 0]
        if "pg_loss_smooth" in df.columns:
            ax.plot(df["episode"], df["pg_loss_smooth"], label=label,
                    color=color, linewidth=1.5)
        ax.set_title("Predator Policy Loss (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("PG Loss")

        # 4) Entropy
        ax = axes[1, 1]
        if "entropy_smooth" in df.columns:
            ax.plot(df["episode"], df["entropy_smooth"], label=label,
                    color=color, linewidth=1.5)
        ax.set_title("Predator Policy Entropy (smoothed)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Entropy")

    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Plot experiment comparison results")
    p.add_argument("--log_dir", type=str, default="results/comparison")
    p.add_argument("--window", type=int, default=20, help="rolling average window")
    p.add_argument("--save", type=str, default=None,
                   help="save figure to file instead of displaying (e.g. comparison.png)")
    args = p.parse_args()

    runs = discover_runs(args.log_dir)
    if not runs:
        print(f"No runs found in {args.log_dir}")
        return

    runs = enrich_labels(runs, args.log_dir)

    print(f"Found {len(runs)} runs:")
    for r in runs:
        df = pd.read_csv(r["csv"])
        print(f"  {r['rich_label']:<30s}  ({len(df)} episodes)  [{r['name']}]")
    print()

    plot_comparison(runs, window=args.window, save_path=args.save)


if __name__ == "__main__":
    main()
