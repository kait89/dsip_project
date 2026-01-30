import argparse
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from cons import (
    RESULTS_DIR,
    DEFAULT_PREDICTION_CSV,
    DEFAULT_TARGET_COLUMN,
    DEFFAULT_PREDICTION_NAME,
)

def calculate_classification_report(df, 
        target_name=DEFAULT_TARGET_COLUMN,
        pred_name=DEFFAULT_PREDICTION_NAME):
    report = classification_report(df[target_name], df[pred_name])
    print(report)
    
    print('--- Confusion Matrix ---')
    print(confusion_matrix(df[target_name], df[pred_name]))


def results(csv_path, per_dataset):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'No prediction file found at {csv_path}')
        
    predictions_df = pd.read_csv(csv_path)
    
    if per_dataset:
        datasets = predictions_df['dataset_name'].unique().tolist()
        for dataset in datasets:
            print(f'\nProcessing Dataset: {dataset}')
            df = predictions_df[predictions_df['dataset_name'] == dataset]
            calculate_classification_report(df)
    else:
        calculate_classification_report(predictions_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Results')
    parser.add_argument('-p', '--csv-path', default=DEFAULT_PREDICTION_CSV)
    parser.add_argument('-pd', '--per-dataset', action='store_true')
    
    args = parser.parse_args()

    resolved_path = os.path.abspath(args.csv_path)
    results_dir_abs = os.path.abspath(RESULTS_DIR)
    if ".." in args.csv_path or not resolved_path.startswith(results_dir_abs):
        raise ValueError(f"Security Error: csv_path must be located inside the {RESULTS_DIR} folder.")

    results(args.csv_path, args.per_dataset)