# -*- coding: utf-8 -*-
"""Generate SHAP explanations using saved train and test partitions."""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

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
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
    TARGET_TO_DATASET,
    TARGET_FOS,
    TARGET_SLIP_DEPTH,
    TARGET_ZWD_FINAL,
    TARGET_ZWU_FINAL
)
from assets import (
    load_legacy_model,
    build_explainer,
    build_explanation,
    save_beeswarm_plot,
    save_waterfall_plot
)


# Define parameters and column subsets for local test case identification.
NUM_TEST_LOCAL_SLOPES = 10
CASE_IDENTIFIER_COLUMNS = {
    "_source_split","_original_train_index", "_tr_numeric", "_dur_numeric", "_static_case_key", "_explanation_position"
}

def select_cross_target_cases(all_train_dfs, all_test_dfs, number_of_slopes=NUM_TEST_LOCAL_SLOPES):
    """Filter test cases across targets based on specific rainfall conditions.
    Require a minimum number of distinct rainfall scenarios (TR + Duration) per slope.
    Rank valid slopes by test sample frequency and impute missing scenarios using the training subset."""
    
    pools = {}
    valid_keys_per_target = {}
    
    # Concatenate the train and test partitions into a single structure.
    for target in ALL_TARGETS:
        df_test = all_test_dfs[target].copy()
        df_train = all_train_dfs[target].copy()
        
        # Assign split identifiers to the respective dataframes.
        df_test["_source_split"] = "test"
        df_train["_source_split"] = "train"
        
        # Preserve the original training indices.
        df_train["_original_train_index"] = df_train.index
        
        combined_df = pd.concat([df_test, df_train], ignore_index=True)
        
        # Explicitly cast Return Period and Duration to numeric types, filling NaNs to avoid grouping issues.
        combined_df["_tr_numeric"] = pd.to_numeric(combined_df[RETURN_PERIOD_COLUMN], errors="coerce").fillna(-1)
        combined_df["_dur_numeric"] = pd.to_numeric(combined_df[RAIN_DURATION_COLUMN], errors="coerce").fillna(-1)
        
        # Define the subset of columns required for the static configuration key (geometry + geotechnical params).
        excluded = set(RAINFALL_COLUMNS) | set(ALL_TARGETS) | set(CASE_IDENTIFIER_COLUMNS)
        case_columns = [col for col in combined_df.columns if col not in excluded and not str(col).lower().startswith("unnamed:")]

        def make_key(row):
            # Format and aggregate the static cell values into a unique tuple key.
            return tuple(int(row[c]) if isinstance(row[c], (np.integer, int)) else 
                         round(float(row[c]), 10) if isinstance(row[c], (np.floating, float)) else 
                         str(row[c]).strip() for c in case_columns)
                         
        # Apply the key generation function across all rows.
        combined_df["_static_case_key"] = combined_df.apply(make_key, axis=1)
        pools[target] = combined_df

    # Filter the combined dataframe based on target-specific precipitation rules.
    for target, combined_df in pools.items():
        if target in ["zwd", "zwu"]:
            valid_df = combined_df[combined_df[ACCUMULATED_RAIN_COLUMN] != 0]
        elif target == "fos":
            valid_df = combined_df[combined_df[ACCUMULATED_RAIN_COLUMN] == 0]
        else:
            valid_df = combined_df
            
        # Extract the unique valid configuration keys for the current target.
        valid_keys_per_target[target] = set(valid_df["_static_case_key"].unique())

    # Calculate the intersection of valid keys across all evaluated targets.
    common_keys = set.intersection(*valid_keys_per_target.values())
    
    # We require 4 distinct rainfall scenarios (e.g., TR -, TR 30, 200, 500 regardless of duration).
    REQUIRED_SCENARIO_COUNT = 4
    
    # Select a reference target to quantify the available scenarios per configuration.
    ref_target = "zwd" if "zwd" in pools else ("zwu" if "zwu" in pools else list(pools.keys())[0])
    combined_ref_df = pools[ref_target].copy()
    
    # Apply the precipitation filter to the reference dataset.
    if ref_target in ["zwd", "zwu"]:
        combined_ref_df = combined_ref_df[combined_ref_df[ACCUMULATED_RAIN_COLUMN] != 0]
        
    # Create a unified tuple to count unique pairs of (Return Period, Duration).
    combined_ref_df["_scenario_pair"] = list(zip(combined_ref_df["_tr_numeric"], combined_ref_df["_dur_numeric"]))
    scenario_counts_per_key = combined_ref_df.groupby("_static_case_key")["_scenario_pair"].nunique().to_dict()
    
    # Retain exclusively the keys meeting the minimum scenario count requirement.
    strictly_valid_keys = {k for k in common_keys if scenario_counts_per_key.get(k, 0) >= REQUIRED_SCENARIO_COUNT}
    
    # Isolate the test split from the reference target dataset.
    test_df_ref = pools[ref_target][pools[ref_target]["_source_split"] == "test"]
    
    if ref_target in ["zwd", "zwu"]:
        test_df_ref = test_df_ref[test_df_ref[ACCUMULATED_RAIN_COLUMN] != 0]
        
    # Compute the occurrence frequency of each configuration key within the test split.
    key_test_counts = test_df_ref["_static_case_key"].value_counts().to_dict()
    
    # Order the valid keys descendingly according to their test split frequencies.
    ordered_keys = sorted(list(strictly_valid_keys))
    ordered_keys.sort(key=lambda k: key_test_counts.get(k, 0), reverse=True)
    
    # Subset the specified top number of configuration keys.
    selected_keys = ordered_keys[:number_of_slopes]
    
    print(f"Slopes in the test set: {len(selected_keys)} ===> {[key_test_counts.get(k, 0) for k in selected_keys]}")

    # Construct the final filtered local datasets for each target variable.
    local_test_sets = {}
    for target, df in pools.items():
        target_rows = []
        for case_number, key in enumerate(selected_keys, start=1):
            
            # Extract all rows matching the current static configuration key.
            case_df = df[df["_static_case_key"] == key].copy()
            
            # Apply the target-specific precipitation filters to the current slope slice.
            if target in ["zwd", "zwu"]:
                case_df = case_df[case_df[ACCUMULATED_RAIN_COLUMN] != 0]
            elif target == "fos":
                case_df = case_df[case_df[ACCUMULATED_RAIN_COLUMN] == 0]
                
            if case_df.empty:
                continue

            selected_samples = []
            
            # Identify all unique pairs of (Return Period, Duration) for this specific slope geometry.
            unique_scenarios = case_df[["_tr_numeric", "_dur_numeric"]].drop_duplicates().values
            
            for tr_val, dur_val in unique_scenarios:
                # Isolate the data corresponding to the exact TR and Duration pairing.
                scenario_df = case_df[(case_df["_tr_numeric"] == tr_val) & (case_df["_dur_numeric"] == dur_val)]
                
                # Query the test split prioritizing unseen data for local explanations.
                test_samples = scenario_df[scenario_df["_source_split"] == "test"]
                if not test_samples.empty:
                    selected_samples.append(test_samples.iloc[[0]]) 
                    continue
                
                # Query the train split if the specific TR/Duration scenario is absent in the test split.
                train_samples = scenario_df[scenario_df["_source_split"] == "train"]
                if not train_samples.empty:
                    selected_samples.append(train_samples.iloc[[0]])
            
            if selected_samples:
                # Concatenate the selected records into a sequential dataframe.
                final_case_df = pd.concat(selected_samples, ignore_index=True)
                
                # Assign sequential scenario identifiers.
                final_case_df["_case_code"] = f"P{case_number}"
                
                # Sort hierarchically by Return Period then by Duration to maintain strict temporal progression.
                final_case_df = final_case_df.sort_values(["_tr_numeric", "_dur_numeric"], kind="stable")
                final_case_df["_scenario_position"] = np.arange(1, len(final_case_df) + 1)
                
                target_rows.append(final_case_df)
                
        # Aggregate all processed slope cases for the current target.
        local_df = pd.concat(target_rows, ignore_index=True) if target_rows else pd.DataFrame()
        if not local_df.empty:
            local_df["_explanation_position"] = np.arange(len(local_df))
        local_test_sets[target] = local_df

    return local_test_sets


def run_train_test_shap():
    """Generate SHAP beeswarm and waterfall plots."""
    print("\n=== TRAIN / TEST SHAP ===")

    best_models = {
        TARGET_FOS: "GradientBoostingRegressor",
        TARGET_SLIP_DEPTH: "GradientBoostingRegressor",
        TARGET_ZWU_FINAL: "DecisionTreeRegressor",
        TARGET_ZWD_FINAL: "DecisionTreeRegressor"
    }

    for slope in ALL_SLOPES:
        
        # Pre-load all datasets for this slope to apply the cross-target filter
        train_dfs = {}
        test_dfs = {}
        for target in ALL_TARGETS:
            dataset_suffix = TARGET_TO_DATASET[target]
            train_dfs[target] = pd.read_csv(os.path.join(DATASET_DIR, "train", f"{slope}_{dataset_suffix}.csv"))
            test_dfs[target] = pd.read_csv(os.path.join(DATASET_DIR, "test", f"{slope}_{dataset_suffix}.csv"))
            
        # Generate synchronized local sets across the different targets utilizing the refined TR/Duration pairing
        local_test_sets = select_cross_target_cases(train_dfs, test_dfs, NUM_TEST_LOCAL_SLOPES)

        for target in ALL_TARGETS:
            df_train = train_dfs[target]
            df_test = test_dfs[target]
            df_test_local = local_test_sets.get(target, pd.DataFrame())

            target_output_root = os.path.join(FIGURES_DIR, "shap_test", slope, target)
            os.makedirs(target_output_root, exist_ok=True)

            print(f"\n{slope.upper()} | {target} | train={len(df_train)} | test={len(df_test)} | local={len(df_test_local)}")

            if df_test_local.empty:
                print(f"No local cases found for {target}. Local waterfall explanations will be skipped.")

            for reg_model in regressors:
                model_name = reg_model.__class__.__name__
                if best_models[target] != model_name:
                    continue

                test_output_dir = os.path.join(target_output_root, model_name)
                os.makedirs(test_output_dir, exist_ok=True)

                # Load the inference model.
                try:
                    with open(os.path.join(RESULTS_DIR, "drained", target, model_name, f"{model_name}_inference.pkl"), "rb") as f:
                        bundle = pickle.load(f)
                except ModuleNotFoundError as e:
                    if e.name == 'sklearn.ensemble._gb_losses':
                        bundle = load_legacy_model(os.path.join(RESULTS_DIR, "drained", target, model_name, f"{model_name}_inference.pkl"))
                    else:
                        raise

                regressor = bundle["model"]
                features, scaler = bundle["features"], bundle.get("scaler")
                feature_names = [NAME_TO_SYMBOL.get(n, n) for n in features]

                # Remove rows used for local explanations from the train background set to prevent data leakage.
                if not df_test_local.empty and "_original_train_index" in df_test_local.columns:
                    idx_to_drop = df_test_local.loc[df_test_local["_source_split"] == "train", "_original_train_index"].dropna().astype(int)
                    df_train_bg = df_train.drop(index=idx_to_drop)
                else:
                    df_train_bg = df_train.copy()

                # Extract the input matrices and scale the feature values.
                X_train_bg_raw = df_train_bg[features]
                X_test_raw = df_test[features]
                
                X_train_bg = scaler.transform(X_train_bg_raw)
                X_test = scaler.transform(X_test_raw)

                bundle["model"] = bundle.get("model")
                
                # Build the explainer using the training background data.
                explainer = build_explainer(bundle["model"], model_name, X_train_bg, feature_names, RANDOM_SEED)

                # Generate and save global beeswarm explanations.
                explanation_test = build_explanation(explainer, model_name, X_test, X_test_raw, feature_names)
                save_beeswarm_plot(explanation_test, os.path.join(test_output_dir, f"{model_name}_beeswarm.png"), RANDOM_SEED)

                # Generate and save sequential waterfall plots.
                if not df_test_local.empty:
                    X_local_raw = df_test_local[features]
                    X_local = scaler.transform(X_local_raw)
                    
                    explanation_local = build_explanation(explainer, model_name, X_local, X_local_raw, feature_names)

                    for case_code, group in df_test_local.groupby("_case_code", sort=False):
                        case_dir = os.path.join(test_output_dir, str(case_code))
                        os.makedirs(case_dir, exist_ok=True)

                        for _, row in group.sort_values("_scenario_position", kind="stable").iterrows():
                            # Extract and format the Return Period label.
                            tr_val = row[RETURN_PERIOD_COLUMN]
                            tr_lbl = '-' if pd.isna(tr_val) or float(tr_val) <= 0 else str(int(tr_val))
                            
                            # Extract and format the rainfall duration label.
                            dur_val = row[RAIN_DURATION_COLUMN]
                            dur_lbl = '0' if pd.isna(dur_val) or float(dur_val) < 0 else str(int(dur_val))
                            
                            # Append both the TR and Duration variables to strictly identify the plot filename.
                            out_file = os.path.join(case_dir, f"{model_name}_{case_code}_Tr_{tr_lbl}_dur_{dur_lbl}_test_waterfall.png")
                            save_waterfall_plot(explanation_local[int(row["_explanation_position"])], out_file)

                print(f"  {model_name} | bg samples={len(X_train_bg_raw)} | output={test_output_dir}")


if __name__ == "__main__":
    run_train_test_shap()