"""Analysis for harmonic oscillator: input noise x dt sweep."""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Reuse shared utilities
from analyze_physics import fetch_runs, fetch_trajectory, pick_best_lr, CACHE_DIR
import wandb


def analyze_harmonic_dt(sweep_id, force_refresh=False):
    api = wandb.Api(timeout=120)
    project = "/".join(sweep_id.split("/")[:2])

    parsed = fetch_runs(sweep_id, force_refresh=force_refresh)
    if not parsed:
        print("No usable runs.")
        return

    # Group by (dt, input_noise), then by LR
    groups = defaultdict(lambda: defaultdict(list))
    for p in parsed:
        key = (p["dt"], p["input_noise"])
        groups[key][p["lr"]].append(p)

    # Powers-of-2 horizons
    horizons = [2**i for i in range(9)]  # 1..256
    h_keys = [f"eval/score_t+{h}" for h in horizons]

    # --- Part 1: Score table ---
    results = []
    best_by_group = {}

    for gk in sorted(groups.keys()):
        lr, avg_loss, best_runs = pick_best_lr(groups[gk])
        best_by_group[gk] = best_runs

        row = {"dt": gk[0], "input_noise": gk[1],
               "best_lr": lr, "train_loss": avg_loss}
        for hk in h_keys:
            vals = [r["scores"][hk] for r in best_runs if hk in r["scores"]]
            row[hk] = np.mean(vals) if vals else None
        row["n_seeds"] = len(best_runs)
        results.append(row)

    df = pd.DataFrame(results)
    cols = ["dt", "input_noise", "best_lr", "train_loss"] + h_keys + ["n_seeds"]
    df = df[[c for c in cols if c in df.columns]]

    pd.options.display.float_format = "{:,.6f}".format
    pd.options.display.max_columns = 20
    pd.options.display.width = 200
    print("\n=== Harmonic: Eval Scores at t+2^i (best LR per config) ===")
    print(df.to_string(index=False))

    csv_path = os.path.join("analysis", "harmonic_dt_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # --- Part 2: Eval score vs horizon (one subplot per dt) ---
    dt_vals = sorted({gk[0] for gk in groups})
    noise_levels = sorted({gk[1] for gk in groups if gk[1] is not None})

    cmap = plt.cm.viridis
    noise_colors = {n: cmap(i / max(1, len(noise_levels) - 1))
                    for i, n in enumerate(noise_levels)}

    fig, axes = plt.subplots(1, len(dt_vals), figsize=(6 * len(dt_vals), 4),
                             squeeze=False, sharey=True)

    for ci, dt in enumerate(dt_vals):
        ax = axes[0, ci]
        for noise in noise_levels:
            gk = (dt, noise)
            best_runs = best_by_group.get(gk, [])
            if not best_runs:
                continue

            all_h = sorted({int(k.split("+")[1])
                            for r in best_runs for k in r["scores"]})
            mean_scores = []
            for h in all_h:
                key = f"eval/score_t+{h}"
                vals = [r["scores"][key] for r in best_runs if key in r["scores"]]
                mean_scores.append(np.mean(vals) if vals else np.nan)

            ax.plot(all_h, mean_scores, color=noise_colors[noise],
                    label=f"input_noise={noise}")

        ax.set_yscale("log")
        ax.set_title(f"dt={dt}")
        ax.set_xlabel("Horizon h")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")

    axes[0, 0].set_ylabel("MSE")
    fig.suptitle("Harmonic: Eval score vs horizon (best LR per config)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fname = os.path.join("analysis", "harmonic_dt_eval_scores.pdf")
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)

    # --- Part 3: Trajectory plots (one per dt, lines colored by input_noise) ---
    for dt in dt_vals:
        all_seeds = sorted({r["seed"] for n in noise_levels
                            for r in best_by_group.get((dt, n), [])})

        fig, axes_t = plt.subplots(1, len(all_seeds), figsize=(5 * len(all_seeds), 4),
                                   squeeze=False, sharey=True)
        gt_plotted = [False] * len(all_seeds)
        any_plotted = False

        for noise in noise_levels:
            gk = (dt, noise)
            runs_for_plot = best_by_group.get(gk, [])

            for run_data in runs_for_plot:
                seed = run_data["seed"]
                if seed not in all_seeds:
                    continue
                seed_idx = all_seeds.index(seed)
                ax = axes_t[0, seed_idx]

                print(f"  Fetching trajectory: dt={dt}, noise={noise}, seed={seed}...")
                traj = fetch_trajectory(api, project, run_data["run_id"],
                                        run_data["table_path"])
                if traj is None:
                    print(f"    Skipped (no data)")
                    continue

                if not gt_plotted[seed_idx]:
                    ax.plot(traj["steps"], traj["truth"], color="black",
                            linewidth=1.5, label="GT", zorder=1)
                    gt_plotted[seed_idx] = True

                ax.plot(traj["steps"], traj["model"], color=noise_colors[noise],
                        alpha=0.8, linewidth=1, label=f"noise={noise}", zorder=5)
                ax.set_title(f"seed={seed}")
                ax.set_xlabel("Horizon")
                ax.grid(True, alpha=0.3)
                any_plotted = True

        if not any_plotted:
            print(f"  No trajectories fetched for dt={dt}, skipping plot.")
            plt.close(fig)
            continue

        axes_t[0, 0].set_ylabel("y")
        handles, labels = axes_t[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize="small")
        fig.suptitle(f"Harmonic dt={dt} — Rollout trajectories", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fname = os.path.join("analysis", f"harmonic_dt{dt}_rollout.pdf")
        fig.savefig(fname, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_harmonic_dt.py <sweep_id> [--refresh]")
    else:
        refresh = "--refresh" in sys.argv
        analyze_harmonic_dt(sys.argv[1], force_refresh=refresh)
