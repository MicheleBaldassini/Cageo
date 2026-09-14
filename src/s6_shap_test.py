# -*- coding: utf-8 -*-
"""Generate SHAP explanations using saved train and test partitions (Part 1)."""

import os
import pickle
import numpy as np
import pandas as pd
import shap

from config import (
    regressors,
    ALL_SLOPES,
    ALL_TARGETS,
    DATASET_DIR,
    FIGURES_DIR,
    NAME_TO_SYMBOL,
    RAINFALL_COLUMNS,
    RANDOM_SEED,
    RESULTS_DIR,
    RETURN_PERIOD_COLUMN,
    TARGET_TO_DATASET,
)
from assets import build_explanation, save_beeswarm_plot, save_waterfall_plot


# Define parameters and column subsets for local test case identification.
NUM_TEST_LOCAL_SLOPES = 10
TEST_REQUIRED_RETURN_PERIODS = (0.0, 30.0, 200.0, 500.0)
CASE_IDENTIFIER_COLUMNS = {
    "_source_split",  "_tr_numeric", "_static_case_key", "_explanation_position"
}

def get_static_case_columns(dataframe):
    """Isolate the feature columns by excluding targets, rainfall features."""
    excluded = set(RAINFALL_COLUMNS) | set(ALL_TARGETS) | set(CASE_IDENTIFIER_COLUMNS)
    return [col for col in dataframe.columns if col not in excluded and not str(col).lower().startswith("unnamed:")]


def select_complete_test_cases(dataframe_test, dataframe_train, number_of_slopes=NUM_TEST_LOCAL_SLOPES):
    """Extract a number of cases containing a sequence of required return periods."""
    test_df = dataframe_test.copy().reset_index(drop=True)
    train_df = dataframe_train.copy().reset_index(drop=True)

    test_df["_source_split"], train_df["_source_split"] = "test", "train"
    test_df["_tr_numeric"] = pd.to_numeric(test_df[RETURN_PERIOD_COLUMN], errors="coerce")
    train_df["_tr_numeric"] = pd.to_numeric(train_df[RETURN_PERIOD_COLUMN], errors="coerce")

    common_columns = [col for col in test_df.columns if col in train_df.columns]
    case_columns = get_static_case_columns(test_df[common_columns].copy())

    def make_key(row):
        """Generate a unique identifier based on the static case features."""
        return tuple(int(row[c]) if isinstance(row[c], (np.integer, int)) else 
                     round(float(row[c]), 10) if isinstance(row[c], (np.floating, float)) else 
                     str(row[c]).strip() for c in case_columns)

    test_df["_static_case_key"] = test_df.apply(make_key, axis=1)
    train_df["_static_case_key"] = train_df.apply(make_key, axis=1)

    # Establish an ordered list of unique case identifiers.
    ordered_case_keys = list(dict.fromkeys(test_df["_static_case_key"].tolist() + train_df["_static_case_key"].tolist()))
    
    selected_rows = []
    selected_case_number = 0

    for case_key in ordered_case_keys:
        test_case = test_df[test_df["_static_case_key"] == case_key]
        train_case = train_df[train_df["_static_case_key"] == case_key]
        rows_for_case = []

        # Check the presence of all return periods for the current case.
        for scenario_position, required_tr in enumerate(TEST_REQUIRED_RETURN_PERIODS, start=1):
            matches = test_case[np.isclose(test_case["_tr_numeric"].to_numpy(float), float(required_tr), rtol=0, atol=1e-9)]
            
            if len(matches) == 0:
                matches = train_case[np.isclose(train_case["_tr_numeric"].to_numpy(float), float(required_tr), rtol=0, atol=1e-9)]
            
            if len(matches) == 0:
                rows_for_case = []
                break
                
            selected_row = matches.iloc[0].copy()
            selected_row["_scenario_position"] = scenario_position
            rows_for_case.append(selected_row)

        if len(rows_for_case) != len(TEST_REQUIRED_RETURN_PERIODS):
            continue

        selected_case_number += 1
        for row in rows_for_case:
            row["_case_code"] = f"P{selected_case_number}"
            selected_rows.append(row)

        if selected_case_number >= number_of_slopes:
            break

    # Create the final dataset and assign sequential positions.
    selected_dataframe = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected_dataframe["_explanation_position"] = np.arange(len(selected_dataframe), dtype=int)

    print(f"Selected local slopes: {selected_case_number} | "
          f"test samples={(selected_dataframe['_source_split'] == 'test').sum()} | "
          f"train samples={(selected_dataframe['_source_split'] == 'train').sum()}")

    return selected_dataframe


def run_train_test_shap():
    """Generate SHAP beeswarm and waterfall plots."""
    print("\n=== TRAIN / TEST SHAP ===")

    for slope in ALL_SLOPES:
        for target in ALL_TARGETS:
            dataset_suffix = TARGET_TO_DATASET[target]
            df_train = pd.read_csv(os.path.join(DATASET_DIR, "train", f"{slope}_{dataset_suffix}.csv"))
            df_test = pd.read_csv(os.path.join(DATASET_DIR, "test", f"{slope}_{dataset_suffix}.csv"))
            
            df_test_local = select_complete_test_cases(df_test, df_train)

            target_output_root = os.path.join(FIGURES_DIR, "shap_test", slope, target)
            os.makedirs(target_output_root, exist_ok=True)

            print(f"\n{slope.upper()} | {target} | train={len(df_train)} | test={len(df_test)} | local={len(df_test_local)}")

            for reg_model in regressors:
                model_name = reg_model.__class__.__name__
                test_output_dir = os.path.join(target_output_root, model_name)
                os.makedirs(test_output_dir, exist_ok=True)

                # Load the hold-out test model.
                with open(os.path.join(RESULTS_DIR, slope, target, model_name, f"{model_name}_test.pkl"), "rb") as f:
                    bundle = pickle.load(f)

                features, scaler = bundle["features"], bundle.get("scaler")
                feature_names = [NAME_TO_SYMBOL.get(n, n) for n in features]

                # Extract the input matrices and scale the feature values.
                X_train_raw = df_train[features]
                X_test_raw = df_test[features]
                X_local_raw = df_test_local[features]

                X_train = scaler.transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)
                X_local = scaler.transform(X_local_raw)

                # Generate the global SHAP explanations for the complete test set.
                explanation_test = build_explanation(bundle["model"], model_name, X_train, X_test, X_test_raw, feature_names, RANDOM_SEED)
                
                save_beeswarm_plot(explanation_test, os.path.join(test_output_dir, f"{model_name}_beeswarm.png"), RANDOM_SEED)
                
                # Filter SHAP explanations to generate beeswarm plots for each return period.
                # for tr_val in sorted(df_test[RETURN_PERIOD_COLUMN].dropna().unique(), key=float):
                #     positions = np.flatnonzero(df_test[RETURN_PERIOD_COLUMN] == tr_val)
                #     if len(positions) == 0: 
                #         continue
                    
                #     tr_lbl = str(int(tr_val)) if float(tr_val).is_integer() else f"{float(tr_val):.6g}".replace(".", "p")
                #     tr_dir = os.path.join(test_output_dir, f"Tr_{tr_lbl}")
                #     os.makedirs(tr_dir, exist_ok=True)
                #     save_beeswarm_plot(explanation_test[positions], os.path.join(tr_dir, f"{model_name}_beeswarm.png"), RANDOM_SEED)

                # Generate the SHAP waterfall plots.
                explanation_local = build_explanation(bundle["model"], model_name, X_train, X_local, X_local_raw, feature_names, RANDOM_SEED)

                for case_code, group in df_test_local.groupby("_case_code", sort=False):
                    case_dir = os.path.join(test_output_dir, str(case_code))
                    os.makedirs(case_dir, exist_ok=True)

                    for _, row in group.sort_values("_scenario_position", kind="stable").iterrows():
                        tr_lbl = str(int(row[RETURN_PERIOD_COLUMN]))#  if float(row[RETURN_PERIOD_COLUMN]).is_integer() else f"{float(row[RETURN_PERIOD_COLUMN]):.6g}".replace(".", "p")
                        if tr_lbl == '0':
                            tr_lbl = '-'
                        out_file = os.path.join(case_dir, f"{model_name}_{case_code}_Tr_{tr_lbl}_test_waterfall.png")
                        save_waterfall_plot(explanation_local[int(row["_explanation_position"])], out_file)

                print(f"  {model_name} | global test={len(df_test)} | local test={len(df_test_local)} | output={test_output_dir}")


if __name__ == "__main__":
    run_train_test_shap()