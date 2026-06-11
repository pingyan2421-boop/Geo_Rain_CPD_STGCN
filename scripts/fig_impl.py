import numpy as np
import argparse
import os
import pandas as pd

def process_files(path):
    # Load the prediction and groundtruth data
    preds = np.load(os.path.join(path, 'y_test.npy'))
    preds_f = np.load(os.path.join(path, 'pred_array.npy'))
    trues = np.load(os.path.join(path, 'x_test.npy'))
    evl = np.load(os.path.join(path, 'evl.npy'))

    preds_df = pd.DataFrame(preds.reshape(preds.shape[0], -1))
    preds_f_df = pd.DataFrame(preds_f.reshape(preds_f.shape[0], -1))
    trues_df = pd.DataFrame(trues.reshape(trues.shape[0], -1))
    trues_df2 = trues_df.values.reshape(3, -1)
    evl_df = pd.DataFrame({'evl': evl.flatten()})
    preds_df2 = preds_df * args.std + args.mean
    preds_f_df2 = preds_f_df * args.std + args.mean
    trues_df2 = pd.DataFrame(trues_df2 * args.std + args.mean)
    metrics_df = abs(preds_df2 - trues_df2)
    # Save to CSV files
    preds_df2.to_csv(os.path.join(path, 'y_test.csv'), index=False)
    trues_df2.to_csv(os.path.join(path, 'x_test.csv'), index=False)
    metrics_df.to_csv(os.path.join(path, 'metrics.csv'), index=False)
    preds_f_df2.to_csv(os.path.join(path, 'pred_array.csv'), index=False)
    evl_df.to_csv(os.path.join(path, 'evl.csv'), index=False)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process the path and filename.')
    # Add the arguments
    parser.add_argument('Path', metavar='path', type=str, help='the path to process')
    parser.add_argument('--mean', type=float, default=1e-3)
    parser.add_argument('--std', type=float, default=1e-3)
    # Execute the parse_args() method
    args = parser.parse_args()
    path = args.Path

    process_files(path)
