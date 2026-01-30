import wandb
from collections import defaultdict
import sys

def analyze_sweep(sweep_id):
    api = wandb.Api()
    try:
        sweep = api.sweep(sweep_id)
    except Exception as e:
        print(f"Error accessing sweep: {e}")
        return

    runs = sweep.runs
    
    # Nested dictionary: (lambda, diagonal) -> lr -> list of runs/metrics
    grouped_runs = defaultdict(lambda: defaultdict(list))
    finished_count = 0
    running_count = 0
    
    # Identify all t+ metrics
    metric_keys = set()
    
    for run in runs:
        if run.state == "finished":
            finished_count += 1
            l_val = run.config.get("lambda_val")
            diag = run.config.get("diagonal_A")
            lr = run.config.get("lr")
            kf_mse = run.summary.get("baseline_kf_mse")
            
            # Find all relevant keys in summary
            metrics = {k: v for k, v in run.summary.items() if k.startswith("eval/score_t+")}
            metric_keys.update(metrics.keys())
            
            if kf_mse is not None and metrics:
                row = {'kf_mse': kf_mse, **metrics}
                grouped_runs[(l_val, diag)][lr].append(row)
        elif run.state == "running":
            running_count += 1
            
    print(f"Sweep Status: {finished_count} finished, {running_count} running.")
    
    if not grouped_runs:
        print("No finished runs with metrics found yet.")
        return

    # Sort keys by horizon t+X
    sorted_keys = sorted(list(metric_keys), key=lambda x: int(x.split("+")[1]))
    
    # Header
    header = f"{'lambda':<8} | {'Diag':<5} | {'Best LR':<10} | {'KF MSE':<10}"
    for k in sorted_keys:
        t = k.split("+")[1]
        header += f" | {f't+{t}':<8}"
    header += f" | {'count':<5}"
    
    print(header)
    print("-" * len(header))
    
    for (l_val, diag) in sorted(grouped_runs.keys()):
        best_lr = None
        best_score = float('inf')
        best_rows = None
        
        # Find best LR based on t+2 score
        for lr, rows in grouped_runs[(l_val, diag)].items():
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
            
            line = f"{l_val:<8.2f} | {diag:<5} | {best_lr:<10.4f} | {avg_kf:<10.3e}"
            
            for k in sorted_keys:
                vals = [r[k] for r in best_rows if k in r]
                if vals:
                    avg = sum(vals) / len(vals)
                    line += f" | {avg:<8.4f}"
                else:
                    line += f" | {'N/A':<8}"
                    
            line += f" | {len(best_rows):<5}"
            print(line)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp2.py <sweep_id>")
    else:
        analyze_sweep(sys.argv[1])
