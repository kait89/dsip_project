import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from cons import (
    DATA_DIR,
    DEFAULT_CSV_RAW_PATH,
    DEFAULT_TARGET_COLUMN,
)

def remove_duplicates(df: pd.DataFrame):
    # TODO: remove duplicates
    return df.drop_duplicates()

def remove_outliers(df: pd.DataFrame):
    # TODO: remove outliers
    return df

def imputate(df: pd.DataFrame):
    # TODO: fill_missing_values
    return df

def normalize(df: pd.DataFrame):
    # TODO: normalize
    return df


def spit_train_test(df: pd.DataFrame, target_column, include_val=False, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if include_val:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, val_df, test_df
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        return train_df, None, test_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw data into training sets')
    parser.add_argument('-crp', '--csv-raw-path', default=DEFAULT_CSV_RAW_PATH)
    parser.add_argument('-tc', '--target-column', default=DEFAULT_TARGET_COLUMN)
    parser.add_argument('-v', '--val', action='store_true')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    df = pd.read_csv(args.csv_raw_path)
    fname = os.path.splitext(os.path.basename(args.csv_raw_path))[0]
    
    result_df = df.copy()
    preprocessing_steps = [remove_duplicates, remove_outliers, imputate, normalize]
    
    for step_func in preprocessing_steps:
        result_df = step_func(result_df)
        step_filename = f'{fname}_{step_func.__name__}.csv'
        result_df.to_csv(os.path.join(DATA_DIR, step_filename), index=False)

    train_df, val_df, test_df = spit_train_test(result_df, args.target_column, args.val)

    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    if val_df is not None:
        val_path = os.path.join(DATA_DIR, 'val.csv')
        val_df.to_csv(val_path, index=False)

        print(f'Successfully created {train_path}, {val_path} and {test_path}')
    else:
        print(f'Successfully created {train_path} and {test_path}')
