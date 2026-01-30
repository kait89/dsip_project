import os
import pandas as pd
import torch
from cons import (
    DATA_DIR,
    MODELS_DIR,
    FLAN_T5_BASE,
    DEFAULT_EPOCH_NUM,
    DEFAULT_MAX_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
)


def get_data():
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    df_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    df_val = None
    val_path = os.path.join(DATA_DIR, 'val.csv')
    if os.path.exists(val_path):
        df_val = pd.read_csv(val_path)
    return df_train, df_val, df_test


def train(model_name: str, 
          epochs: int, batch_size: int,
          max_steps: int, learning_rate: float,
          df_train: pd.DataFrame, df_va: pd.DataFrame, df_test: pd.DataFrame):
    # TODO: pass arguments
    # TODO: train
    # TODO: evaluate
    model = None
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('-m', '--model-name', default=FLAN_T5_BASE)
    parser.add_argument('-e', '--epochs', type='int', default=DEFAULT_EPOCH_NUM)
    parser.add_argument('-bs', '--batch-size', type='int', default=DEFAULT_BATCH_SIZE)
    parser.add_argument('-ms', '--max-steps', type='int', default=DEFAULT_MAX_STEPS)
    parser.add_argument('-lr', '--learning-rate', type='float', default=DEFAULT_LEARNING_RATE)

    args = parser.parse_args()

    df_train, df_val, df_test = get_data()
    model = train(*args, df_train, df_val, df_test)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, args.model_name)
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
