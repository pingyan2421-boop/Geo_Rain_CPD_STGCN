import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data_loader.date_loader import load_and_clean_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbors_file", required=True)
    parser.add_argument("--target_indices_file", required=True)
    parser.add_argument("--file_path", default="dataset/inter228_5241.csv")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load dataset dates
    raw_seq, _, _, time_cols, _ = load_and_clean_data(args.file_path)
    dates = pd.to_datetime([d.replace('.1', '') if type(d)==str else d for d in time_cols], errors='coerce')
    
    # Load target indices
    target_indices = np.load(args.target_indices_file)
    
    # Load neighbors
    neighbors_df = pd.read_csv(args.neighbors_file)
    
    records = []
    has_leak = False
    
    for _, row in neighbors_df.iterrows():
        sample_index = int(row["sample_index"])
        target_abs_idx = target_indices[sample_index]
        target_date = dates[target_abs_idx]
        
        for k in range(30):
            col = f"neighbor_{k}"
            if col not in row:
                break
            
            neighbor_abs_idx = int(row[col])
            neighbor_date = dates[neighbor_abs_idx]
            
            is_future = neighbor_abs_idx >= target_abs_idx
            if is_future:
                has_leak = True
                
            weight_col = f"weight_{k}"
            weight = row[weight_col] if weight_col in row else np.nan
            
            records.append({
                "target_sample_index": sample_index,
                "target_abs_index": target_abs_idx,
                "target_origin_date": target_date,
                "neighbor_abs_index": neighbor_abs_idx,
                "neighbor_origin_date": neighbor_date,
                "is_future_neighbor": is_future,
                "weight": weight
            })
            
    out_df = pd.DataFrame(records)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path / "knn_time_leak_audit.csv", index=False)
    
    print("=== KNN Time Leak Audit ===")
    print(f"Total neighbor links audited: {len(out_df)}")
    print(f"Future leaks detected: {out_df['is_future_neighbor'].sum()}")
    
    if has_leak:
        print("CRITICAL WARNING: Time leak detected! The KNN teacher used future samples to correct predictions.")
    else:
        print("SUCCESS: No time leak detected. The KNN teacher is strictly causal.")

if __name__ == "__main__":
    main()
