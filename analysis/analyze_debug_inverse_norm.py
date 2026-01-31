import wandb
import pandas as pd
import sys
import os

def analyze_debug_sweep(sweep_id):
    print(f"Fetching runs for sweep: {sweep_id}")
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_id)
    except Exception as e:
        print(f"Error accessing sweep: {e}")
        return

    data = []
    
    # Define horizons to track
    horizons = [2, 8, 64]
    metric_keys = [f"eval/score_t+{h}" for h in horizons]
    keys_to_fetch = ["step"] + metric_keys
    
    for run in sweep.runs:
        if run.state != "finished":
            continue
            
        config = run.config
        patch_size = config.get("patch_size")
        remove_inv = config.get("remove_inverse_norm")
        seed = config.get("seed")
        
        # Fetch history
        # use pandas=True for efficiency if available, but run.history returns DataFrame by default
        hist = run.history(keys=keys_to_fetch)
        
        # Filter rows that have evaluation metrics
        # (dropna on one of the metrics)
        if not metric_keys[0] in hist.columns:
            continue
            
        hist = hist.dropna(subset=[metric_keys[0]])
        
        for _, row in hist.iterrows():
            record = {
                "patch_size": patch_size,
                "remove_inv": remove_inv,
                "seed": seed,
                "step": row["step"]
            }
            for h, k in zip(horizons, metric_keys):
                if k in row:
                    record[f"t+{h}"] = row[k]
            
            data.append(record)
            
    if not data:
        print("No data found.")
        return

    df = pd.DataFrame(data)
    
    # Group by config and step, average over seeds
    group_cols = ["patch_size", "remove_inv", "step"]
    df_avg = df.groupby(group_cols)[["t+2", "t+8", "t+64"]].mean().reset_index()
    
    # Sort
    df_avg = df_avg.sort_values(by=group_cols)
    
    # Print
    pd.options.display.float_format = '{:,.4f}'.format
    print("\nDebug Analysis Results (Averaged over seeds):")
    print(df_avg.to_string(index=False))
    
    # Save
    out_path = "analysis/debug_inverse_norm_results.csv"
    df_avg.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analysis/analyze_debug_inverse_norm.py <sweep_id>")
    else:
        analyze_debug_sweep(sys.argv[1])
