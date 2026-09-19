# -*- coding: utf-8 -*-
"""Perform feature selection, cross-validation, and train the final testing and inference models."""

import json
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from config import (
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
    TARGET_SLIP_DEPTH,
    SAMPLE_WEIGHTS
)
from assets import (
    sanitize_params,
    get_slope_groups,
    get_train_processed_indices,
    make_scaler,
    get_sample_weights,
    AutoWeightedRegressor
)
from slope_sample_report import print_slope_sample_breakdown


# Define the number of splits for the cross-validation process.
NUM_PARTITIONS = 10
# Define random seeds for the number of repetitions.
RANDOM_STATES = [RANDOM_SEED, 54923, 69875]
NUM_SCORES = NUM_PARTITIONS * len(RANDOM_STATES)

# Minimum number of times a feature must be selected to be considered relevant.
MIN_SELECTION_COUNT = 15
# Tolerance for the Sequential Feature Selector.
SFS_TOL = 1e-6
SFS_INTERNAL_CV = 5
# Utilize all available CPU cores for parallel processing.
N_CORES = os.cpu_count() or 1

# Define the output paths
FEATURES_PATH = os.path.join(RESULTS_DIR, "selected_features.json")
PARAMS_PATH = os.path.join(RESULTS_DIR, "model_params.json")


def plot_target_unique_counts(y_series, target_name, slope_name, split="Train", decimals=1):
    """Plot bar chart."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    if target_name == TARGET_FOS:
        rounded_values = y_series.round(decimals)
        val_counts = rounded_values.value_counts().sort_index()
        title = f"Unique Values Count - {target_name} ({slope_name}, {split} - Rounded {decimals} dec)"
    else:
        val_counts = y_series.value_counts().sort_index()
        title = f"Unique Values Count - {target_name} ({slope_name}, {split} - Exact Values)"

    sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, ax=ax, color="teal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Unique Values", fontsize=11)
    ax.set_ylabel("Count (Number of Samples)", fontsize=11)
    ax.tick_params(axis="x", rotation=90)

    if len(val_counts) > 25:
        ax.tick_params(axis="x", labelsize=7)

    plt.tight_layout()

    out_dir = os.path.join(RESULTS_DIR, slope_name, target_name)
    os.makedirs(out_dir, exist_ok=True)
    clean_split = split.lower().replace(" ", "_")
    fig_path = os.path.join(out_dir, f"unique_counts_{clean_split}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  



# =========================================================
# EVALUATION FUNCTIONS
# =========================================================
def evaluate_partition(train_idx, validation_idx, df_train_source, feature_names, slope_groups_train, slope_name, target_name, regressor_model, seed):
    """Perform feature selection and evaluate the model on a single validation partition."""
    df_fold_train = df_train_source.iloc[train_idx]
    df_fold_val = df_train_source.iloc[validation_idx]
    groups_train_fold_full = slope_groups_train.iloc[train_idx]

    kept_train_idx = get_train_processed_indices(
        df_fold_train, target_name=target_name, slope_name=slope_name, random_seed=seed
    )

    X_train = df_fold_train.loc[kept_train_idx, feature_names].astype(float)
    y_train = df_fold_train.loc[kept_train_idx, target_name].astype(float)
    groups_train_fold = groups_train_fold_full.loc[kept_train_idx]

    X_validation = df_fold_val[feature_names].astype(float)
    y_validation = df_fold_val[target_name].astype(float)

    # Normalize predictor variables
    scaler = make_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)

    n_internal_groups = groups_train_fold.nunique()
    actual_internal_cv = min(SFS_INTERNAL_CV, n_internal_groups)
    inner_group_cv = list(GroupKFold(n_splits=actual_internal_cv).split(X_train_scaled, y_train, groups=groups_train_fold))

    if SAMPLE_WEIGHTS:
        wrapped_model = AutoWeightedRegressor(regressor_model, target_name)
    else:
        wrapped_model = clone(regressor_model)

    # Employ a sequential forward feature selection strategy within a nested 
    # cross-validation loop to identify the most robust predictive variables.
    sfs = SequentialFeatureSelector(
        estimator=wrapped_model,
        n_features_to_select="auto",
        tol=SFS_TOL,
        direction="forward",
        scoring="r2",
        cv=inner_group_cv,
        n_jobs=1,
    )
    sfs.fit(X_train_scaled, y_train)

    selected_mask = sfs.get_support()
    selected_features = [
        feature for feature, selected in zip(feature_names, selected_mask) if selected
    ]

    model = clone(regressor_model)
    model.fit(X_train_scaled[:, selected_mask], y_train)

    prediction = model.predict(X_validation_scaled[:, selected_mask])

    return {
        "features": selected_features,
        "r2": r2_score(y_validation, prediction),
        "mae": mean_absolute_error(y_validation, prediction),
    }


def select_features(feature_counts):
    """Identify the most relevant features based on their selection frequency."""
    for threshold in (MIN_SELECTION_COUNT, 10):
        selected = [feature for feature, count in feature_counts.items() if count >= threshold]
        if selected:
            return selected

    max_count = max(feature_counts.values())
    return [feature for feature, count in feature_counts.items() if count == max_count]


# =========================================================
# PHASE 1: FEATURE SELECTION
# =========================================================
os.makedirs(RESULTS_DIR, exist_ok=True)
selected_features_all = {}

for slope in ALL_SLOPES:
    selected_features_all[slope] = {}

    for target in ALL_TARGETS:
        train_path = os.path.join(BASE_DIR, "dataset", "train", f"{slope}_{TARGET_TO_DATASET[target]}.csv")
        df_train = pd.read_csv(train_path)

        feature_names = [col for col in df_train.columns if col not in ALL_TARGETS]
        X_train_full = df_train[feature_names].astype(float)
        y_train_full = df_train[target].astype(float)

        slope_groups_train = get_slope_groups(df_train)

        selected_features_all[slope][target] = {}

        print(f"\n=== FEATURE SELECTION | {slope} | {target} ===")
        print(f"Total samples: {len(df_train)}, Distinct Slopes: {slope_groups_train.nunique()}")

        first_partition_splitter = GroupKFold(n_splits=NUM_PARTITIONS)
        first_train_idx, first_val_idx = next(
            first_partition_splitter.split(X_train_full, y_train_full, groups=slope_groups_train)
        )
        first_train_processed_idx = get_train_processed_indices(
            df_train.iloc[first_train_idx], target_name=target, slope_name=slope, random_seed=RANDOM_SEED
        )
        first_train_groups = slope_groups_train.iloc[first_train_idx].loc[first_train_processed_idx].nunique()
        first_val_groups = slope_groups_train.iloc[first_val_idx].nunique()
        print(
            f"Train: {len(first_train_processed_idx)} samples ({first_train_groups} slopes), "
            f"Val: {len(first_val_idx)} samples ({first_val_groups} slopes)"
        )
        print_slope_sample_breakdown(
            df_train.loc[first_train_processed_idx],
            slope_groups_train.loc[first_train_processed_idx], "First fold train"
        )
        print_slope_sample_breakdown(
            df_train.iloc[first_val_idx], slope_groups_train.iloc[first_val_idx],
            "First fold validation"
        )

        # y_train_processed_sample = df_train.loc[first_train_processed_idx, target].astype(float)
        # y_val_processed_sample = df_train.loc[first_val_idx, target].astype(float)

        # plot_target_unique_counts(
        #     y_series=y_train_processed_sample,
        #     target_name=target,
        #     slope_name=slope,
        #     split="Train",
        #     decimals=1,
        # )

        # plot_target_unique_counts(
        #     y_series=y_val_processed_sample,
        #     target_name=target,
        #     slope_name=slope,
        #     split="Val",
        #     decimals=1,
        # )
        # continue

        for regressor_model in regressors:
            model_name = regressor_model.__class__.__name__
            results = []

            for seed in RANDOM_STATES:
                # Apply random permutations to group identifiers to generate distinct validation splits.
                rng = np.random.RandomState(seed)
                unique_groups = slope_groups_train.unique()
                permuted_groups = rng.permutation(unique_groups)
                group_map = {orig: perm for orig, perm in zip(unique_groups, permuted_groups)}
                shuffled_groups = slope_groups_train.map(group_map)

                splitter = GroupKFold(n_splits=NUM_PARTITIONS)

                # Distribute the repeated feature selection process.
                fold_results = Parallel(n_jobs=N_CORES)(
                    delayed(evaluate_partition)(
                        train_idx,
                        validation_idx,
                        df_train,
                        feature_names,
                        slope_groups_train,
                        slope,
                        target,
                        regressor_model,
                        seed,
                    )
                    for train_idx, validation_idx in splitter.split(
                        X_train_full, y_train_full, groups=shuffled_groups
                    )
                )
                results.extend(fold_results)

            feature_counts = {feature: 0 for feature in feature_names}
            for result in results:
                for feature in result["features"]:
                    feature_counts[feature] += 1

            final_features = select_features(feature_counts)
            selected_features_all[slope][target][model_name] = final_features

            feature_frequency_df = pd.DataFrame(
                [
                    {
                        "feature": feature,
                        "selection_count": count,
                        "selection_frequency": count / NUM_SCORES,
                        "selected_final": feature in final_features,
                    }
                    for feature, count in feature_counts.items()
                ]
            ).sort_values(
                by=["selected_final", "selection_count"],
                ascending=[False, False],
            )

            os.makedirs(os.path.join(RESULTS_DIR, slope, target, model_name), exist_ok=True)
            feature_frequency_df.to_csv(
                os.path.join(RESULTS_DIR, slope, target, model_name, "feature_freq.csv"),
                index=False,
            )

            mean_r2 = np.mean([result["r2"] for result in results])
            mean_mae = np.mean([result["mae"] for result in results])
            std_r2 = np.std([result["r2"] for result in results])
            std_mae = np.std([result["mae"] for result in results])

            print(
                f"{model_name}: R2={mean_r2:.4f} +/- {std_r2:.4f} | MAE={mean_mae:.4f} +/- {std_mae:.4f} | features={final_features}"
            )

with open(FEATURES_PATH, "w", encoding="utf-8") as file:
    json.dump(selected_features_all, file, indent=4, ensure_ascii=False)

print(f"\nSelected features saved to: {FEATURES_PATH}")


# =========================================================
# PHASE 2: CROSS-VALIDATION AND FINAL MODEL TRAINING
# =========================================================
def evaluate_fold(train_idx, val_idx, df_train_source, features, slope_name, target_name, regressor_model, seed):
    """Train and evaluate the model on a single cross-validation fold with disjoint slopes."""
    df_cv_train = df_train_source.iloc[train_idx]
    df_cv_val = df_train_source.iloc[val_idx]

    kept_cv_train_idx = get_train_processed_indices(
        df_cv_train, target_name=target_name, slope_name=slope_name, random_seed=seed
    )

    X_cv_train = df_cv_train.loc[kept_cv_train_idx, features].astype(float)
    y_cv_train = df_cv_train.loc[kept_cv_train_idx, target_name].astype(float)

    X_cv_val = df_cv_val[features].astype(float)
    y_cv_val = df_cv_val[target_name].astype(float)

    scaler_cv = make_scaler()
    X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
    X_cv_val_scaled = scaler_cv.transform(X_cv_val)

    if SAMPLE_WEIGHTS:
        model_cv = AutoWeightedRegressor(regressor_model, target_name)
    else:
        model_cv = clone(regressor_model)

    model_cv.fit(X_cv_train_scaled, y_cv_train)
    y_cv_pred = model_cv.predict(X_cv_val_scaled)

    return r2_score(y_cv_val, y_cv_pred), mean_absolute_error(y_cv_val, y_cv_pred)


with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    selected_features_all = json.load(file)

trained_params_all = {}

for slope in ALL_SLOPES:
    trained_params_all[slope] = {}

    for target in ALL_TARGETS:
        trained_params_all[slope][target] = {}

        train_path = os.path.join(
            BASE_DIR, "dataset", "train", f"{slope}_{TARGET_TO_DATASET[target]}.csv"
        )
        test_path = os.path.join(
            BASE_DIR, "dataset", "test", f"{slope}_{TARGET_TO_DATASET[target]}.csv"
        )

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df_all = pd.concat([df_train, df_test], ignore_index=True)

        slope_groups_train = get_slope_groups(df_train)

        print(f"\n=== TRAINING & CROSS-VALIDATION | {slope} | {target} ===")

        first_partition_splitter = GroupKFold(n_splits=NUM_PARTITIONS)
        first_train_idx, first_val_idx = next(
            first_partition_splitter.split(df_train, groups=slope_groups_train)
        )
        first_train_processed_idx = get_train_processed_indices(
            df_train.iloc[first_train_idx], target_name=target, slope_name=slope, random_seed=RANDOM_SEED
        )
        first_train_groups = slope_groups_train.iloc[first_train_idx].loc[first_train_processed_idx].nunique()
        first_val_groups = slope_groups_train.iloc[first_val_idx].nunique()
        print(
            f"Train: {len(first_train_processed_idx)} samples ({first_train_groups} slopes), "
            f"Val: {len(first_val_idx)} samples ({first_val_groups} slopes)"
        )
        print_slope_sample_breakdown(
            df_train.loc[first_train_processed_idx],
            slope_groups_train.loc[first_train_processed_idx], "First fold train"
        )
        print_slope_sample_breakdown(
            df_train.iloc[first_val_idx], slope_groups_train.iloc[first_val_idx],
            "First fold validation"
        )

        target_model_scores_r2 = {}
        target_model_scores_mae = {}

        for regressor_model in regressors:
            model_name = regressor_model.__class__.__name__

            features = selected_features_all[slope][target][model_name]

            model_dir = os.path.join(RESULTS_DIR, slope, target, model_name)
            os.makedirs(model_dir, exist_ok=True)

            X_train_full = df_train[features].astype(float)
            y_train_full = df_train[target].astype(float)

            # ---------------------------------------------------------
            # REPEATED 10-FOLD GROUP CROSS-VALIDATION
            # ---------------------------------------------------------
            # Assess the model's generalization capabilities using 
            # independent grouped splits.
            results = []

            for seed in RANDOM_STATES:
                rng = np.random.RandomState(seed)
                unique_groups = slope_groups_train.unique()
                permuted_groups = rng.permutation(unique_groups)
                group_map = {orig: perm for orig, perm in zip(unique_groups, permuted_groups)}
                shuffled_groups = slope_groups_train.map(group_map)

                splitter = GroupKFold(n_splits=NUM_PARTITIONS)

                cv_results = Parallel(n_jobs=N_CORES)(
                    delayed(evaluate_fold)(
                        train_idx,
                        val_idx,
                        df_train,
                        features,
                        slope,
                        target,
                        regressor_model,
                        seed,
                    )
                    for train_idx, val_idx in splitter.split(
                        X_train_full, y_train_full, groups=shuffled_groups
                    )
                )
                results.extend(cv_results)

            cv_r2_scores = [res[0] for res in results]
            cv_mae_scores = [res[1] for res in results]

            target_model_scores_r2[model_name] = cv_r2_scores
            target_model_scores_mae[model_name] = cv_mae_scores

            mean_cv_r2 = np.mean(cv_r2_scores)
            std_cv_r2 = np.std(cv_r2_scores)
            mean_cv_mae = np.mean(cv_mae_scores)
            std_cv_mae = np.std(cv_mae_scores)

            # ---------------------------------------------------------
            # TEST MODEL (Train on full train set, evaluate on complete Hold-out Test)
            # ---------------------------------------------------------
            # Fit the architecture on the training partition 
            X_train_final = df_train.loc[:, features].astype(float)
            y_train_final = df_train.loc[:, target].astype(float)

            X_test = df_test[features].astype(float)
            y_test = df_test[target].astype(float)

            scaler_test = make_scaler()
            X_train_scaled = scaler_test.fit_transform(X_train_final)
            X_test_scaled = scaler_test.transform(X_test)

            if SAMPLE_WEIGHTS:
                test_model = AutoWeightedRegressor(regressor_model, target)
            else:
                test_model = clone(regressor_model)

            test_model.fit(X_train_scaled, y_train_final)

            if SAMPLE_WEIGHTS:
                test_model = test_model.estimator_

            model_params = test_model.get_params()
            clean_params = sanitize_params(model_params)
            trained_params_all[slope][target][model_name] = clean_params

            y_test_pred = test_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            test_data = {
                "model": test_model,
                "scaler": scaler_test,
                "features": features,
                "params": clean_params,
                "y_holdout_true": y_test,
                "y_holdout_pred": y_test_pred,
            }

            with open(os.path.join(model_dir, f"{model_name}_test.pkl"), "wb") as file:
                pickle.dump(test_data, file)

            # ---------------------------------------------------------
            # INFERENCE MODEL (Train on complete dataset)
            # ---------------------------------------------------------
            # Optimize the final model utilizing all available datasets.
            kept_all_idx = get_train_processed_indices(
                df_all, target_name=target, slope_name=slope, random_seed=RANDOM_SEED
            )

            X_all_final = df_all.loc[kept_all_idx, features].astype(float)
            y_all_final = df_all.loc[kept_all_idx, target].astype(float)

            scaler_inference = make_scaler()
            X_all_scaled = scaler_inference.fit_transform(X_all_final)

            if SAMPLE_WEIGHTS:
                inference_model = AutoWeightedRegressor(regressor_model, target)
            else:
                inference_model = clone(regressor_model)

            inference_model.fit(X_all_scaled, y_all_final)

            if SAMPLE_WEIGHTS:
                inference_model = inference_model.estimator_

            inference_data = {
                "model": inference_model,
                "scaler": scaler_inference,
                "features": features,
                "params": clean_params,
            }

            with open(os.path.join(model_dir, f"{model_name}_inference.pkl"), "wb") as file:
                pickle.dump(inference_data, file)

            print(
                f"{model_name}: R2={mean_cv_r2:.4f} +/- {std_cv_r2:.4f} | MAE={mean_cv_mae:.4f} +/- {std_cv_mae:.4f} | hold-out R2={test_r2:.4f} | hold-out MAE={test_mae:.4f}"
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
