from invoke import task
import os
from datetime import datetime
from cons import (
    BEST_MODEL,
    S3_CSV_PATH,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    ARCHIVED_EXPERIMENTS_DIR,
    DATE_TIME_PATTERN,
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
)


@task
def download_best_model(c, aws_path=BEST_MODEL):
    c.run(f'aws s3 cp {aws_path} models/')

@task
def prepare_data_for_pipeline(c, s3_csv_path=S3_CSV_PATH):
    c.run(f'aws s3 cp {s3_csv_path} data/raw_data.csv')

@task
def pipeline(c, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LEARNING_RATE):
    prepare_data_for_pipeline()
    c.run(f'python preproces.py')
    c.run(f'python train.py --epochs {epochs} --batch-size {batch_size} --learning-rate {learning_rate}')


@task
def archive(c, name, base_folder=ARCHIVED_EXPERIMENTS_DIR):
    name = f'{datetime.now().strftime(DATE_TIME_PATTERN)}_{name}'
    print(f'archived experiment: {name}')

    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    exp_path = f'{base_folder}/{name}'
    c.run(f'mkdir {exp_path}')

    for folder in [DATA_DIR, RESULTS_DIR, MODELS_DIR]:
        if os.path.exists(folder):
            c.run(f'mv {folder} {exp_path}')
            c.run(f'mkdir {folder}')