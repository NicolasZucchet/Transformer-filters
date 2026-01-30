import sys
from common import fetch_and_analyze_sweep

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_exp3.py <sweep_id>")
    else:
        # Exp 3: Group by lambda_val, diagonal_A, patch_size
        fetch_and_analyze_sweep(
            sys.argv[1], 
            grouping_keys=["lambda_val", "diagonal_A", "patch_size"], 
            csv_filename="exp3_results.csv"
        )
