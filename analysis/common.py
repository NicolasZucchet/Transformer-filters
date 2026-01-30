import wandb
import pandas as pd
import sys
import os
from collections import defaultdict

def fetch_and_analyze_sweep(sweep_id, grouping_keys, csv_filename=None):
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_id)
    except Exception as e:
        print(f"Error accessing sweep: {e}")
        return

    runs = sweep.runs
    
    # Nested dictionary: tuple(group_values) -> lr -> list of runs/metrics
    grouped_runs = defaultdict(lambda: defaultdict(list))
    finished_count = 0
    running_count = 0
    
    # Identify all t+ metrics
    metric_keys = set()
    
    for run in runs:
        if run.state == "finished":
            finished_count += 1
            
            # Extract grouping values
            group_values = []
            for key in grouping_keys:
                val = run.config.get(key)
                group_values.append(val)
            group_values = tuple(group_values)
            
            lr = run.config.get("lr")
            kf_mse = run.summary.get("baseline_kf_mse")
            
            # Find all relevant keys in summary
            metrics = {k: v for k, v in run.summary.items() if k.startswith("eval/score_t+")}
            metric_keys.update(metrics.keys())
            
            if kf_mse is not None and metrics:
                row = {'kf_mse': kf_mse, **metrics}
                grouped_runs[group_values][lr].append(row)
        elif run.state == "running":
            running_count += 1
            
    print(f"Sweep Status: {finished_count} finished, {running_count} running.")
    
    if not grouped_runs:
        print("No finished runs with metrics found yet.")
        return

    # Sort keys by horizon t+X
    sorted_metric_keys = sorted(list(metric_keys), key=lambda x: int(x.split("+")[1]))
    
    # Prepare data for DataFrame/Printing
    results = []
    
    for group_values in sorted(grouped_runs.keys()):
        best_lr = None
        best_score = float('inf')
        best_rows = None
        
        # Find best LR based on t+2 score
        for lr, rows in grouped_runs[group_values].items():
            valid_rows = [r for r in rows if 'eval/score_t+2' in r]
            if not valid_rows:
                continue
            
            avg_t2 = sum(r['eval/score_t+2'] for r in valid_rows) / len(valid_rows)
            
            if avg_t2 < best_score:
                best_score = avg_t2
                best_lr = lr
                best_rows = valid_rows
        
        if best_rows:
            avg_kf = sum(r['kf_mse'] for r in best_rows) / len(best_rows)
            
            # Create result row
            row_data = {}
            for k, v in zip(grouping_keys, group_values):
                row_data[k] = v
            
            row_data['Best LR'] = best_lr
            row_data['KF MSE'] = avg_kf
            
            for k in sorted_metric_keys:
                vals = [r[k] for r in best_rows if k in r]
                if vals:
                    avg = sum(vals) / len(vals)
                    row_data[k] = avg
                else:
                    row_data[k] = None
            
            row_data['count'] = len(best_rows)
            results.append(row_data)

    df = pd.DataFrame(results)
    
    # Reorder columns for nice printing
    cols = grouping_keys + ['Best LR', 'KF MSE'] + sorted_metric_keys + ['count']
    df = df[cols]
    
    # Print table
    # Format floats
    pd.options.display.float_format = '{:,.4f}'.format
    print(df.to_string(index=False))
    
    # Save to CSV
    if csv_filename:
        # Check if we need to prepend analysis/
        if not csv_filename.startswith("analysis/"):
             csv_path = os.path.join("analysis", csv_filename)
        else:
             csv_path = csv_filename
             
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
