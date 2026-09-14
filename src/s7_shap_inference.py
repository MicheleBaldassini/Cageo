# -*- coding: utf-8 -*-
"""Generate SHAP explanations for external inference."""

import os
import pickle
import numpy as np
import pandas as pd
import shap

from config import (
    regressors,
    ALL_TARGETS,
    DATASET_DIR,
    FIGURES_DIR,
    RANDOM_SEED,
    NAME_TO_SYMBOL,
    RESULTS_DIR,
    RETURN_PERIOD_COLUMN,
    TARGET_TO_DATASET,
)
from assets import build_explanation, save_waterfall_plot


def run_real_cases_inference(slope="drained"):
    """Compute and plot SHAP waterfall explanations for external dataset."""
    print("\n=== INFERENCE SHAP ===")

    input_csv = os.path.join(DATASET_DIR, "real_cases.csv")
    if not os.path.isfile(input_csv):
        return print(f"[SKIPPED] File not found: {input_csv}")

    # Load the external validation dataset.
    df_real = pd.read_csv(input_csv)
    df_real["_original_row_index"] = np.arange(len(df_real), dtype=int)
    
    if len(df_real) == 0:
        return print(f"[SKIPPED] No external cases available for {slope}.")

    for target in ALL_TARGETS:
        dataset_suffix = TARGET_TO_DATASET[target]
        
        # Combine the training and testing datasets to serve as the SHAP background distribution.
        df_train = pd.read_csv(os.path.join(DATASET_DIR, "train", f"drained_{dataset_suffix}.csv"))
        df_test = pd.read_csv(os.path.join(DATASET_DIR, "test", f"drained_{dataset_suffix}.csv"))
        df_background = pd.concat([df_train, df_test], ignore_index=True)

        for reg_model in regressors:
            model_name = reg_model.__class__.__name__
            inference_out_dir = os.path.join(FIGURES_DIR, "shap_inference", "drained", target, model_name)
            os.makedirs(inference_out_dir, exist_ok=True)

            # Load the inference model.
            with open(os.path.join(RESULTS_DIR, "drained", target, model_name, f"{model_name}_inference.pkl"), "rb") as f:
                bundle = pickle.load(f)

            features, scaler = bundle["features"], bundle.get("scaler")
            feature_names = [NAME_TO_SYMBOL.get(n, n) for n in features]

            # Extract the raw input features and apply the scaler.
            X_bg_raw = df_background[features]#.to_numpy(float)
            X_real_raw = df_real[features]#.to_numpy(float)
            
            X_bg = scaler.transform(X_bg_raw) if scaler else X_bg_raw
            X_real = scaler.transform(X_real_raw) if scaler else X_real_raw

            # Compute the SHAP values using the background distribution.
            explanation_real = build_explanation(bundle["model"], model_name, X_bg, X_real, X_real_raw, feature_names, RANDOM_SEED)

            # Generate a waterfall plot for each instance in the external dataset.
            for pos, (_, row) in enumerate(df_real.iterrows()):
                tr_val = row[RETURN_PERIOD_COLUMN]

                # Format the return period value to generate a filename string.
                if tr_val != tr_val:
                    tr_lbl = '-'
                else:
                    tr_lbl = str(int(row[RETURN_PERIOD_COLUMN]))
                
                out_file = os.path.join(inference_out_dir, f"{model_name}_row_{int(row['_original_row_index'])}_Tr_{tr_lbl}_inference_waterfall.png")
                save_waterfall_plot(explanation_real[pos], out_file)

            print(f" {target} | {model_name} | samples={len(df_real)} | output={inference_out_dir}")


if __name__ == "__main__":
    run_real_cases_inference()