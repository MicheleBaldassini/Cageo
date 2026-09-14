# -*- coding: utf-8 -*-
"""Tune model hyperparameters and train both the evaluation and final inference models."""

import json
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold

from config import (
    parameters_reg,
    regressors,
    BASE_DIR,
    ALL_SLOPES,
    ALL_TARGETS,
    RANDOM_SEED,
    RESULTS_DIR,
    TARGET_TO_DATASET,
    RETURN_PERIOD_COLUMN,
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
    TARGET_FOS,
    SAMPLE_WEIGHTS,
)
from assets import (
    sanitize_params,
    get_slope_groups,
    get_train_processed_indices,
    make_scaler,
    get_sample_weights,
    AutoWeightedRegressor
)

# Define the number of outer splits for the nested cross-validation process.
OUTER_FOLDS = 10
# Define random seeds for the number of repetitions.
CV_SEEDS = [RANDOM_SEED, 54923, 69875]
# Define the number of inner splits for the hyperparameter optimization.
INNER_FOLDS = 5
N_CORES = os.cpu_count() or 1

FEATURES_PATH = os.path.join(RESULTS_DIR, "selected_features.json")
PARAMS_PATH = os.path.join(RESULTS_DIR, "model_params.json")


# =========================================================
# EVALUATION FUNCTIONS
# =========================================================
def evaluate_outer_fold(
    train_idx,
    val_idx,
    df_train_source,
    features,
    slope_groups_train,
    slope_name,
    target_name,
    regressor_model,
    parameter_grid,
    seed,
):
    """Perform hyperparameter tuning on inner folds and evaluate on the outer validation fold."""
    # Extract the training and validation subsets.
    df_outer_train = df_train_source.iloc[train_idx]
    df_outer_val = df_train_source.iloc[val_idx]
    groups_outer_train_full = slope_groups_train.iloc[train_idx]

    # Apply the data filtering process.
    kept_train_idx = get_train_processed_indices(
        df_outer_train, target_name=target_name, slope_name=slope_name, random_seed=seed
    )

    # Isolate the feature matrices and target vectors.
    X_outer_train = df_outer_train.loc[kept_train_idx, features].astype(float)
    y_outer_train = df_outer_train.loc[kept_train_idx, target_name].astype(float)
    groups_outer_train = groups_outer_train_full.loc[kept_train_idx]

    X_outer_val = df_outer_val[features].astype(float)
    y_outer_val = df_outer_val[target_name].astype(float)

    # Standardize the input features.
    scaler = make_scaler()
    X_outer_train_scaled = scaler.fit_transform(X_outer_train)
    X_outer_val_scaled = scaler.transform(X_outer_val)

    # Define the internal cross-validation splits.
    n_internal_groups = groups_outer_train.nunique()
    actual_internal_cv = min(INNER_FOLDS, n_internal_groups)
    inner_group_cv = list(
        GroupKFold(n_splits=actual_internal_cv).split(
            X_outer_train_scaled, y_outer_train, groups=groups_outer_train
        )
    )

    if SAMPLE_WEIGHTS:
        estimator = AutoWeightedRegressor(regressor_model, target_name)
    else:
        estimator = clone(regressor_model)

    # Configure the hyperparameter grid search.
    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameter_grid,
        scoring="r2",
        cv=inner_group_cv,
        n_jobs=1,
        refit=True,
        verbose=0,
    )

    # Execute the grid search optimization.
    search.fit(X_outer_train_scaled, y_outer_train)

    # Generate predictions on the validation subset.
    y_outer_pred = search.predict(X_outer_val_scaled)

    # Calculate the evaluation metrics.
    outer_r2 = r2_score(y_outer_val, y_outer_pred)
    outer_mae = mean_absolute_error(y_outer_val, y_outer_pred)
    best_params = search.best_params_

    return {
        "r2": outer_r2,
        "mae": outer_mae,
        "record": {
            "params": best_params,
            "inner_r2": float(search.best_score_),
        },
    }


def nested_cross_validation(df_train_source, features, slope_groups_train, slope_name, target_name, regressor_model, parameter_grid):
    """Execute a repeated nested cross-validation process across parallel workers."""
    tasks = []
    X_for_split = df_train_source[features]
    y_for_split = df_train_source[target_name]

    # Iterate through the random seeds.
    for seed in CV_SEEDS:
        rng = np.random.RandomState(seed)
        
        # Shuffle the group identifiers.
        unique_groups = slope_groups_train.unique()
        permuted_groups = rng.permutation(unique_groups)
        group_map = {orig: perm for orig, perm in zip(unique_groups, permuted_groups)}
        shuffled_groups = slope_groups_train.map(group_map)

        # Define the outer cross-validation partitions.
        outer_cv = GroupKFold(n_splits=OUTER_FOLDS)
        for train_idx, val_idx in outer_cv.split(X_for_split, y_for_split, groups=shuffled_groups):
            tasks.append((train_idx, val_idx, seed))

    # Distribute the evaluation tasks across parallel workers.
    results = Parallel(n_jobs=N_CORES)(
        delayed(evaluate_outer_fold)(
            train_idx,
            val_idx,
            df_train_source,
            features,
            slope_groups_train,
            slope_name,
            target_name,
            regressor_model,
            parameter_grid,
            seed,
        )
        for train_idx, val_idx, seed in tasks
    )

    # Aggregate the evaluation scores and parameter records.
    outer_r2_scores = np.asarray([res["r2"] for res in results])
    outer_mae_scores = np.asarray([res["mae"] for res in results])
    best_parameter_records = [res["record"] for res in results]

    return outer_r2_scores, outer_mae_scores, best_parameter_records


def select_hyperparameters(records):
    """Identify the most robust hyperparameter configuration across all outer folds."""
    # Extract the hyperparameter combinations.
    keys = [json.dumps(record["params"], sort_keys=True) for record in records]

    # Count the frequency of each combination.
    counts = Counter(keys)
    max_count = max(counts.values())
    
    # Identify the most frequent hyperparameter combinations.
    candidates = [key for key, count in counts.items() if count == max_count]

    # Sort tied combinations by their inner validation scores.
    if len(candidates) > 1:
        candidates.sort(
            key=lambda key: np.mean(
                [
                    record["inner_r2"]
                    for record, record_key in zip(records, keys)
                    if record_key == key
                ]
            ),
            reverse=True,
        )

    return json.loads(candidates[0])


# =========================================================
# SCRIPT EXECUTION
# =========================================================
if __name__ == "__main__":
    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        selected_features = json.load(file)

    trained_params_all = {}

    for slope in ALL_SLOPES:
        trained_params_all[slope] = {}

        for target in ALL_TARGETS:
            trained_params_all[slope][target] = {}

            train_path = os.path.join(
                BASE_DIR,
                "dataset",
                "train",
                f"{slope}_{TARGET_TO_DATASET[target]}.csv",
            )
            test_path = os.path.join(
                BASE_DIR,
                "dataset",
                "test",
                f"{slope}_{TARGET_TO_DATASET[target]}.csv",
            )

            # Combine the partitioned training and testing datasets.
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            df_all = pd.concat([df_train, df_test], ignore_index=True)

            # Assign categorical identifiers based on slope groups.
            slope_groups_train = get_slope_groups(df_train)

            print(f"\n=== MODEL SELECTION | {slope} | {target} ===")
            print(f"Total samples: {len(df_train)}, Distinct Slopes: {slope_groups_train.nunique()}")

            first_partition_splitter = GroupKFold(n_splits=OUTER_FOLDS)
            first_train_idx, first_val_idx = next(
                first_partition_splitter.split(df_train, groups=slope_groups_train)
            )
            first_train_processed_idx = get_train_processed_indices(
                df_train.iloc[first_train_idx], target_name=target, slope_name=slope, random_seed=RANDOM_SEED
            )
            first_train_groups = slope_groups_train.iloc[first_train_idx].loc[first_train_processed_idx].nunique()
            first_val_groups = slope_groups_train.iloc[first_val_idx].nunique()
            print(
                f"[First Partition] Train (Processed): {len(first_train_processed_idx)} samples ({first_train_groups} slopes), "
                f"Val: {len(first_val_idx)} samples ({first_val_groups} slopes)"
            )

            target_model_scores_r2 = {}
            target_model_scores_mae = {}

            for regressor_model in regressors:
                model_name = regressor_model.__class__.__name__

                features = selected_features[slope][target][model_name]

                # Execute the nested cross-validation process.
                (outer_r2_scores, outer_mae_scores, parameter_records) = nested_cross_validation(
                    df_train,
                    features,
                    slope_groups_train,
                    slope,
                    target,
                    regressor_model,
                    parameters_reg[model_name],
                )

                nested_mean_r2 = float(np.mean(outer_r2_scores))
                nested_std_r2 = float(np.std(outer_r2_scores))
                nested_mean_mae = float(np.mean(outer_mae_scores))
                nested_std_mae = float(np.std(outer_mae_scores))

                # Select the final optimal hyperparameters.
                best_params = select_hyperparameters(parameter_records)

                target_model_scores_r2[model_name] = outer_r2_scores
                target_model_scores_mae[model_name] = outer_mae_scores

                # ---------------------------------------------------------
                # TEST MODEL (Train on full train set, Hold-out Test completo)
                # ---------------------------------------------------------
                # Extract the complete training and testing sets.
                X_train_final = df_train.loc[:, features].astype(float)
                y_train_final = df_train.loc[:, target].astype(float)

                X_test = df_test[features].astype(float)
                y_test = df_test[target].astype(float)

                # Standardize the final dataset features.
                scaler_test = make_scaler()
                X_train_scaled = scaler_test.fit_transform(X_train_final)
                X_test_scaled = scaler_test.transform(X_test)

                # Apply the selected hyperparameters to the model.
                model_configured = clone(regressor_model)
                model_configured.set_params(**best_params)

                if SAMPLE_WEIGHTS:
                    test_model = AutoWeightedRegressor(model_configured, target)
                else:
                    test_model = clone(model_configured)

                # Fit the model on the full training partition.
                test_model.fit(X_train_scaled, y_train_final)

                if SAMPLE_WEIGHTS:
                    test_model = test_model.estimator_

                model_params = test_model.get_params()
                clean_params = sanitize_params(model_params)
                trained_params_all[slope][target][model_name] = clean_params

                # Evaluate the model on the hold-out test set.
                y_test_pred = test_model.predict(X_test_scaled)
                holdout_r2 = r2_score(y_test, y_test_pred)
                holdout_mae = mean_absolute_error(y_test, y_test_pred)

                model_dir = os.path.join(RESULTS_DIR, slope, target, model_name)
                os.makedirs(model_dir, exist_ok=True)

                test_data = {
                    "model": test_model,
                    "scaler": scaler_test,
                    "features": features,
                    "best_params": best_params,
                    "params": clean_params,
                    "y_holdout_true": y_test,
                    "y_holdout_pred": y_test_pred,
                }

                with open(
                    os.path.join(model_dir, f"{model_name}_test.pkl"), "wb"
                ) as file:
                    pickle.dump(test_data, file)

                # ---------------------------------------------------------
                # INFERENCE MODEL (Train su df_all)
                # ---------------------------------------------------------
                # kept_all_idx = get_train_processed_indices(
                #     df_all, target_name=target, slope_name=slope, random_seed=RANDOM_SEED
                # )

                # Extract the complete aggregated dataset.
                X_all_final = df_all.loc[:, features].astype(float)
                y_all_final = df_all.loc[:, target].astype(float)

                # Standardize the aggregated dataset features.
                scaler_inference = make_scaler()
                X_all_scaled = scaler_inference.fit_transform(X_all_final)

                model_inference_configured = clone(regressor_model)
                model_inference_configured.set_params(**best_params)

                if SAMPLE_WEIGHTS:
                    inference_model = AutoWeightedRegressor(model_inference_configured, target)
                else:
                    inference_model = clone(model_inference_configured)

                # Retrain the model configuration on the aggregated dataset.
                inference_model.fit(X_all_scaled, y_all_final)

                if SAMPLE_WEIGHTS:
                    inference_model = inference_model.estimator_

                inference_data = {
                    "model": inference_model,
                    "scaler": scaler_inference,
                    "features": features,
                    "best_params": best_params,
                    "params": clean_params,
                }

                with open(
                    os.path.join(model_dir, f"{model_name}_inference.pkl"), "wb"
                ) as file:
                    pickle.dump(inference_data, file)

                print(
                    f"{model_name}: R2={nested_mean_r2:.4f} +/- {nested_std_r2:.4f} | MAE={nested_mean_mae:.4f} +/- {nested_std_mae:.4f} | hold-out R2={holdout_r2:.4f} | hold-out MAE={holdout_mae:.4f} | hp={best_params}"
                )

            scores_dir = os.path.join(RESULTS_DIR, slope, target)
            os.makedirs(scores_dir, exist_ok=True)

            pd.DataFrame(target_model_scores_r2).to_csv(
                os.path.join(scores_dir, "scores_r2.csv"), index=False
            )
            pd.DataFrame(target_model_scores_mae).to_csv(
                os.path.join(scores_dir, "scores_mae.csv"), index=False
            )

            print(f"Saved outer-fold scores for {slope} | {target} in {scores_dir}")

    with open(PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(trained_params_all, file, indent=4, ensure_ascii=False)

    print(f"\nModel parameters saved to: {PARAMS_PATH}")