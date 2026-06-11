import numpy as np
import argparse
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(preds, trues):
    num_rows = preds.shape[0]
    metrics_list = []

    for i in range(num_rows):
        # Ensure the same data type for both trues and preds
        trues_i = trues[i].astype(preds[i].dtype)

        mse = mean_squared_error(trues_i, preds[i])
        mae = mean_absolute_error(trues_i, preds[i])
        mape = np.mean(np.abs((trues_i - preds[i]) / trues_i)) * 100  # Avoid division by zero in MAPE
        evar = explained_variance_score(trues_i, preds[i])
        ssim_value = ssim(trues_i, preds[i])

        metrics_list.append((mse, mae, mape, evar, ssim_value))

    return metrics_list

def process_files(path):
    # Load the prediction and groundtruth data
    preds = np.load(os.path.join(path, 'y_test.npy'))
    preds_f = np.load(os.path.join(path, 'pred_array.npy'))
    trues = np.load(os.path.join(path, 'x_test.npy'))
    evl = np.load(os.path.join(path, 'evl.npy'))
    preds = preds[:, 2, :]
    preds_f = preds_f[[2, 5, 8], 2, :]
    trues = trues[:, 2, :]

    preds_df = pd.DataFrame(preds.reshape(preds.shape[0], -1))
    preds_f_df = pd.DataFrame(preds_f.reshape(preds_f.shape[0], -1))
    trues_df = pd.DataFrame(trues.reshape(trues.shape[0], -1))
    trues_df2 = trues_df.values.reshape(3, -1)
    evl_df = pd.DataFrame({'evl': evl.flatten()})
    preds_df2 = preds_df * args.std + args.mean
    preds_f_df2 = preds_f_df * args.std + args.mean
    trues_df2 = pd.DataFrame(trues_df2 * args.std + args.mean)
    metrics_df = abs(preds_df2 - trues_df2)

    # Calculate performance metrics
    f1, f2, f3 = calculate_metrics(preds_df2.values, trues_df2.values)

    # Save metrics to CSV files
    metrics_df.to_csv(os.path.join(path, 'metrics.csv'), index=False)

    # Print performance metrics
    print(f'time_step1): {f1}')
    print(f'time_step2: {f2}')
    print(f'time_step3: {f3}')

    # Save metrics values to a separate CSV file
    metrics_values_df = pd.DataFrame({'time_step1': [f1], 'time_step2': [f2], 'time_step3': [f3]})
    metrics_values_df.to_csv(os.path.join(path, 'performance_metrics.csv'), index=False)

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
