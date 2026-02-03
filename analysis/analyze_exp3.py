import sys
import pandas as pd
import matplotlib.pyplot as plt
import os
from common import fetch_and_analyze_sweep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp3.py <sweep_id>")
    else:
        # Exp 3: Group by lambda_val, structure, patch_size
        csv_filename = "exp3_results.csv"
        fetch_and_analyze_sweep(
            sys.argv[1], 
            grouping_keys=["lambda_val", "structure", "patch_size"], 
            csv_filename=csv_filename
        )
        
        # Load the results
        csv_path = os.path.join("analysis", csv_filename)
        if not os.path.exists(csv_path):
             csv_path = csv_filename
             
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Filter for lambda=0.98
            subset = df[df["lambda_val"] == 0.98]
            
            if subset.empty:
                print("No data found for lambda_val=0.98")
            else:
                plt.figure(figsize=(10, 6))
                
                # Identify horizon columns
                horizon_cols = [c for c in df.columns if c.startswith("eval/score_t+")]
                # Sort them by horizon t (integer)
                horizon_cols.sort(key=lambda x: int(x.split("+")[1]))
                
                horizons = [int(x.split("+")[1]) for x in horizon_cols]
                
                for _, row in subset.iterrows():
                    patch_size = row["patch_size"]
                    scores = row[horizon_cols].values
                    
                    plt.plot(horizons, scores, label=f"Patch Size {patch_size}")
                
                plt.xlabel("Horizon (t)")
                plt.ylabel("Loss Ratio (Model / KF)")
                plt.title("Loss vs Horizon for lambda=0.98")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                output_plot = os.path.join("analysis", "exp3_lambda0.98_loss_vs_horizon.png")
                plt.savefig(output_plot)
                print(f"Plot saved to {output_plot}")
        else:
            print(f"Could not find {csv_path} to generate plots.")
