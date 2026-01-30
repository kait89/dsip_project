import argparse
import os
import pandas as panda
import torch
from cons import (
    RESULTS_DIR,
    MODELS_DIR,
    FLAN_T5_BASE,
    DEFFAULT_PREDICTION_NAME,
)

def predict(df, model, text_label):
    # df['prediction'] = model.predict(df[text_label])
    return df

def predict_csv(csv_path, model_name, output_csv_name, text_label, build_prompt):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()

    df = pd.read_csv(csv_path)
    results_df = predict(df, model, text_label)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_csv_path = os.path.join(RESULTS_DIR, output_csv_name)
    
    results_df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to: {output_csv_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predictor")
    
    parser.add_argument('-p', '--csv-path', required=True)
    parser.add_argument('-m', '--model-name', default=FLAN_T5_BASE)
    parser.add_argument('-o', '--output-csv-name', required=True)
    parser.add_argument('-t', '--text-label', default=DEFFAULT_PREDICTION_NAME)
    parser.add_argument('-b', '--dont-build-prompt', action='store_true')
    
    args = parser.parse_args()

    resolved_path = os.path.abspath(args.output_csv_name)
    results_dir_abs = os.path.abspath(RESULTS_DIR)
    if ".." in args.output_csv_name or not resolved_path.startswith(results_dir_abs):
        raise ValueError(f"Security Error: csv_path must be located inside the {RESULTS_DIR} folder.")

    predict_csv(
        csv_path=args.csv_path, 
        model_name=args.model_name, 
        output_csv_name=args.output_csv_name, 
        text_label=args.text_label,
        build_prompt=not args.dont_build_prompt
    )