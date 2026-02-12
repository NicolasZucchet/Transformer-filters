"""Analysis for physics / input jitter sweep."""
import sys
import os
import json
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from collections import defaultdict

CACHE_DIR = os.path.join("analysis", ".cache")


def get_config_val(config, key):
    """Get config value, handling both nested and flat (dotted) formats."""
    if key in config:
        return config[key]
    parts = key.split(".")
    val = config
    for p in parts:
        if isinstance(val, dict):
            val = val.get(p)
        else:
            return None
    return val


def _cache_path(sweep_id, suffix="runs"):
    """Return cache file path for a sweep."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_id = sweep_id.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_id}_{suffix}.json")


def fetch_runs(sweep_id, force_refresh=False):
    """Fetch all finished runs, with JSON caching."""
    cache = _cache_path(sweep_id, "runs")
    if not force_refresh and os.path.exists(cache):
        print(f"Loading cached runs from {cache}")
        with open(cache) as f:
            parsed = json.load(f)
        print(f"  {len(parsed)} runs loaded from cache.")
        return parsed

    api = wandb.Api(timeout=120)
    sweep = api.sweep(sweep_id)

    parsed = []
    n_finished, n_running = 0, 0

    for run in sweep.runs:
        if run.state == "running":
            n_running += 1
            continue
        if run.state != "finished":
            continue
        n_finished += 1

        config = run.config
        system = get_config_val(config, "data.physics_system")
        obs_noise = get_config_val(config, "data.obs_noise_std")
        input_noise = get_config_val(config, "data.input_noise_std")
        lr = get_config_val(config, "train.learning_rate")
        seed = get_config_val(config, "seed")

        final_loss = run.summary.get("train/loss")
        scores = {k: v for k, v in run.summary.items() if k.startswith("eval/score_t+")}

        table_meta = run.summary.get("eval/rollout_y0_ex0_table")
        table_path = None
        if table_meta is not None and hasattr(table_meta, "get"):
            table_path = table_meta.get("path")

        if final_loss is None or not scores:
            continue

        dt = get_config_val(config, "data.dt")

        parsed.append({
            "system": system, "obs_noise": obs_noise, "input_noise": input_noise,
            "dt": dt, "lr": lr, "seed": seed, "final_loss": final_loss,
            "scores": scores, "run_id": run.id, "table_path": table_path,
        })

    print(f"Sweep: {n_finished} finished, {n_running} running, {len(parsed)} usable.")
    with open(cache, "w") as f:
        json.dump(parsed, f)
    print(f"Cached to {cache}")
    return parsed


def fetch_trajectory(api, project, run_id, table_path):
    """Download a rollout trajectory table, with caching."""
    if not table_path:
        return None

    # Check cache
    cache = _cache_path(f"{run_id}_traj", "table")
    if os.path.exists(cache):
        with open(cache) as f:
            cached = json.load(f)
        return {k: np.array(v) for k, v in cached.items()}

    try:
        run = api.run(f"{project}/{run_id}")
        with tempfile.TemporaryDirectory() as tmpdir:
            run.file(table_path).download(replace=True, root=tmpdir)
            with open(os.path.join(tmpdir, table_path)) as fh:
                table = json.load(fh)

        data = {"truth": {}, "model": {}}
        for row in table["data"]:
            step, key, val = row
            if key in data:
                data[key][step] = val

        steps = sorted(data["truth"].keys())
        result = {
            "steps": [float(s) for s in steps],
            "truth": [data["truth"][s] for s in steps],
            "model": [data["model"][s] for s in steps],
        }
        # Cache it
        with open(cache, "w") as f:
            json.dump(result, f)
        return {k: np.array(v) for k, v in result.items()}
    except Exception as e:
        print(f"    Warning: failed to fetch trajectory for {run_id}: {e}")
        return None


def pick_best_lr(runs_by_lr):
    """Pick LR with lowest avg final training loss."""
    best_lr, best_loss, best_runs = None, float("inf"), None
    for lr, lr_runs in runs_by_lr.items():
        avg = np.mean([r["final_loss"] for r in lr_runs])
        if avg < best_loss:
            best_lr, best_loss, best_runs = lr, avg, lr_runs
    return best_lr, best_loss, best_runs


def analyze_physics_sweep(sweep_id, force_refresh=False):
    api = wandb.Api(timeout=120)
    project = "/".join(sweep_id.split("/")[:2])

    parsed = fetch_runs(sweep_id, force_refresh=force_refresh)
    if not parsed:
        print("No usable runs.")
        return

    # Group by (system, obs_noise, input_noise), then by LR
    groups = defaultdict(lambda: defaultdict(list))
    for p in parsed:
        key = (p["system"], p["obs_noise"], p["input_noise"])
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

        row = {"system": gk[0], "obs_noise": gk[1], "input_noise": gk[2],
               "best_lr": lr, "train_loss": avg_loss}
        for hk in h_keys:
            vals = [r["scores"][hk] for r in best_runs if hk in r["scores"]]
            row[hk] = np.mean(vals) if vals else None
        row["n_seeds"] = len(best_runs)
        results.append(row)

    df = pd.DataFrame(results)
    cols = ["system", "obs_noise", "input_noise", "best_lr", "train_loss"] + h_keys + ["n_seeds"]
    df = df[[c for c in cols if c in df.columns]]

    pd.options.display.float_format = "{:,.6f}".format
    pd.options.display.max_columns = 20
    pd.options.display.width = 200
    print("\n=== Eval Scores at t+2^i (best LR per config) ===")
    print(df.to_string(index=False))

    csv_path = os.path.join("analysis", "physics_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")

    # --- Part 2: Eval score vs horizon plot (2x2 grid) ---
    plot_eval_scores(best_by_group, groups)

    # --- Part 3: Trajectory plots (obs_noise=0, per system) ---
    systems = sorted({gk[0] for gk in groups if gk[1] == 0})
    noise_levels = sorted({gk[2] for gk in groups if gk[1] == 0 and gk[2] is not None})

    cmap = plt.cm.viridis
    noise_colors = {n: cmap(i / max(1, len(noise_levels) - 1))
                    for i, n in enumerate(noise_levels)}

    for system in systems:
        # Collect all seeds across noise levels
        all_seeds = sorted({r["seed"] for n in noise_levels
                            for r in best_by_group.get((system, 0, n), [])})

        fig, axes = plt.subplots(1, len(all_seeds), figsize=(5 * len(all_seeds), 4),
                                 squeeze=False, sharey=True)
        gt_plotted = [False] * len(all_seeds)
        any_plotted = False

        for noise in noise_levels:
            gk = (system, 0, noise)
            runs_for_plot = best_by_group.get(gk, [])

            for run_data in runs_for_plot:
                seed = run_data["seed"]
                if seed not in all_seeds:
                    continue
                seed_idx = all_seeds.index(seed)
                ax = axes[0, seed_idx]

                print(f"  Fetching trajectory: {system}, noise={noise}, seed={seed}...")
                traj = fetch_trajectory(api, project, run_data["run_id"],
                                        run_data["table_path"])
                if traj is None:
                    print(f"    Skipped (no data)")
                    continue

                # Plot GT in black (once per seed, behind model lines)
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
            print(f"  No trajectories fetched for {system}, skipping plot.")
            plt.close(fig)
            continue

        axes[0, 0].set_ylabel("y")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize="small")
        fig.suptitle(f"{system} — Rollout trajectories (obs_noise=0)", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        fname = os.path.join("analysis", f"physics_rollout_{system}.pdf")
        fig.savefig(fname, bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close(fig)


def plot_eval_scores(best_by_group, groups):
    """Plot eval/score_t+h vs h. 2x2 grid: rows=system, cols=obs_noise."""
    systems = sorted({gk[0] for gk in groups})
    obs_noises = sorted({gk[1] for gk in groups})
    noise_levels = sorted({gk[2] for gk in groups if gk[2] is not None})

    cmap = plt.cm.viridis
    noise_colors = {n: cmap(i / max(1, len(noise_levels) - 1))
                    for i, n in enumerate(noise_levels)}

    fig, axes = plt.subplots(len(systems), len(obs_noises),
                             figsize=(6 * len(obs_noises), 4 * len(systems)),
                             squeeze=False, sharex=True)

    for ri, system in enumerate(systems):
        for ci, obs in enumerate(obs_noises):
            ax = axes[ri, ci]
            for noise in noise_levels:
                gk = (system, obs, noise)
                best_runs = best_by_group.get(gk, [])
                if not best_runs:
                    continue

                # Collect all horizons from these runs
                all_h = sorted({int(k.split("+")[1])
                                for r in best_runs for k in r["scores"]})
                # Average across seeds
                mean_scores = []
                for h in all_h:
                    key = f"eval/score_t+{h}"
                    vals = [r["scores"][key] for r in best_runs if key in r["scores"]]
                    mean_scores.append(np.mean(vals) if vals else np.nan)

                ax.plot(all_h, mean_scores, color=noise_colors[noise],
                        label=f"input_noise={noise}")

            ax.set_yscale("log")
            ax.set_title(f"{system}, obs_noise={obs}")
            ax.set_xlabel("Horizon h")
            ax.set_ylabel("MSE")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize="small")

    fig.suptitle("Eval score vs horizon (best LR per config)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = os.path.join("analysis", "physics_eval_scores.pdf")
    fig.savefig(fname, bbox_inches="tight")
    print(f"Saved {fname}")
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_physics.py <sweep_id> [--refresh]")
    else:
        refresh = "--refresh" in sys.argv
        analyze_physics_sweep(sys.argv[1], force_refresh=refresh)
