import sys
from common import fetch_and_analyze_sweep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp1.py <sweep_id>")
    else:
        # Exp 1: Group by lambda_val
        fetch_and_analyze_sweep(
            sys.argv[1], 
            grouping_keys=["lambda_val"], 
            csv_filename="exp1_results.csv"
        )
