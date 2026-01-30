import sys
from common import fetch_and_analyze_sweep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp4.py <sweep_id>")
    else:
        # Exp 4: Group by dim_y, patch_size
        fetch_and_analyze_sweep(
            sys.argv[1], 
            grouping_keys=["dim_y", "patch_size"], 
            csv_filename="exp4_results.csv"
        )
