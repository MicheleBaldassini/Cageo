# -*- coding: utf-8 -*-
"""Generate the training and test datasets."""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from config import (
    BASE_DIR,
    DATASET_DIR,
    RANDOM_SEED,
    TARGET_FOS,
    RETURN_PERIOD_COLUMN,
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
    ALL_TARGETS,
    TARGET_ZWD_FINAL,
    TARGET_ZWU_FINAL
)

TEST_SIZE = 0.2

COLUMNS = {
    "Unit weight [kN/m3]": "γ",
    "Effective cohesion [kPa]": "c'",
    "Effective friction angle [°]": "φ'",
    "Undrained Shear Strength [kPa]": "Cu",
    "Saturated permeability [m/s]": "ksat",
    "Soil Type [-]": "ST",
    "Slope angle [°]": "α",
    "Slope length [m]": "L",
    "Total length [m]": "B",
    "Slope height [m]": "H",
    "Total height Upstream [m]": "hm",
    "Total height Downstream [m]": "hd",
    "Soil depth upstream [m]": "hSu",
    "Soil depth downstream [m]": "hSd",
    "Bedrock depth upstream [m]": "hBu",
    "Bedrock depth downstream [m]": "hBd",
    "Initial piezometric surface depth - upstream [m]": "zwuinit",
    "Initial piezometric surface depth - downstream [m]": "zwdinit",
    "Return period of precipitation [years]": "Tr",
    "Accumulated precipitation [mm]": "hw",
    "Precipitation duration [hrs]": "tw",
    "Factor of safety [-]": "FoS",
    "Depth of slip surface [m]": "zs",
    "Final Piezometric surface depth - upstream [m]": "zwufinal",
    "Final Piezometric surface depth - downstream [m]": "zwdfinal",
}


def merge_datasets(prefix, condition_name):
    """Combine drained datasets."""
    dataframes = []
    file_names = []

    # Load all drained condition datasets
    for file_name in sorted(os.listdir(DATASET_DIR)):
        if file_name.startswith("D.1") or file_name.startswith("D.2"):
            full_path = os.path.join(DATASET_DIR, file_name)
            dataframes.append(pd.read_csv(full_path))
            file_names.append(file_name)

    if not dataframes:
        return

    # Identify the columns common to all loaded datasets
    common_cols = [
        col
        for col in dataframes[0].columns
        if all(col in df.columns for df in dataframes[1:])
    ]

    # Collect all unique columns
    all_cols = []
    for df in dataframes:
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)

    missing_cols = [col for col in all_cols if col not in common_cols]

    # Filter columns
    indexed_dataframes = []
    for dataset_index, df in enumerate(dataframes):
        df = df.loc[:, common_cols].copy()
        indexed_dataframes.append(df)

    combined = pd.concat(indexed_dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # Save the merged dataset to disk
    output_path = os.path.join(DATASET_DIR, "D_drained.csv")
    combined.to_csv(output_path, index=False, header=True, float_format="%.10f")
    print(f"Saved {output_path}: {len(combined)} samples, {len(combined.columns)} columns, removed {len(missing_cols)} non-shared features.")


def plot_unique_value_counts(target_datasets, condition, split="all", decimals=1):
    """Plot bar charts with the counts of unique values for each target."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
    fig.suptitle(
        f"Condition: {condition.upper()}, Split: {split.upper()}",
        fontsize=18,
        fontweight="bold",
    )

    # Define the mapping for each target variable
    targets = [
        ("fos", TARGET_FOS, axes[0, 0], "teal"),
        ("zs", TARGET_SLIP_DEPTH, axes[0, 1], "darkorange"),
        ("zwd", TARGET_ZWD_FINAL, axes[1, 0], "forestgreen"),
        ("zwu", TARGET_ZWU_FINAL, axes[1, 1], "crimson"),
    ]

    # Generate a bar plot for each target
    for dict_key, col_name, ax, color in targets:
        df = target_datasets[dict_key]

        # Round values and calculate frequencies for Fos
        if dict_key == "fos":
            rounded_values = df[col_name].round(decimals)
            val_counts = rounded_values.value_counts().sort_index()
            print(
                f"[{condition.upper()} - {split.upper()}] {col_name} has {len(val_counts)} unique values."
            )
            ax.set_title(f"{col_name} (Unique Values - Rounded {decimals} dec)")
        # Calculate frequencies for exact values for other targets
        else:
            val_counts = df[col_name].value_counts().sort_index()
            ax.set_title(f"{col_name} (All {len(val_counts)} Unique Values)")

        sns.barplot(x=val_counts.index.astype(str), y=val_counts.values, ax=ax, color=color)
        ax.set_xlabel("Unique Values")
        ax.set_ylabel("Count (Number of Samples)")
        ax.tick_params(axis="x", rotation=90)

        if len(val_counts) > 20:
            ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    plt.show()


# =============================================================
# CREATE TRAIN AND TEST DATASETS FOR EACH TARGET
# =============================================================
def split_datasets():
    """Split the data by grouping distinct slopes into train (80%) and test (20%)."""
    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Define the paths for the input datasets
    dataset_paths = {
        "drained": os.path.join(DATASET_DIR, "D_drained.csv"),
        "undrained": os.path.join(DATASET_DIR, "U_undrained.csv"),
    }

    # Define columns to exclude when generating the slope ID
    cols_to_ignore_for_slope_id = [
        RETURN_PERIOD_COLUMN,
        ACCUMULATED_RAIN_COLUMN,
        RAIN_DURATION_COLUMN,
    ] + ALL_TARGETS

    # Process each condition dataset
    for condition, dataset_path in dataset_paths.items():
        df = pd.read_csv(dataset_path)

        # Create specific datasets for each target
        target_datasets = {
            "fos": df.copy(),
            "zs": df.copy(),
            "zwd": df[df[TARGET_ZWD_FINAL].ne(0)].copy(),
            "zwu": df[df[TARGET_ZWU_FINAL].ne(0)].copy(),
        }

        # Filter out conditions not subjected to rainfall for ZWD and ZWU
        target_datasets["zwd"] = target_datasets["zwd"][
            target_datasets["zwd"][ACCUMULATED_RAIN_COLUMN] != 0
        ].copy()
        target_datasets["zwu"] = target_datasets["zwu"][
            target_datasets["zwu"][ACCUMULATED_RAIN_COLUMN] != 0
        ].copy()

        # plot_unique_value_counts(target_datasets, condition, split="all")

        train_datasets_for_plot = {}
        test_datasets_for_plot = {}

        # Group Shuffle Split for each target dataset
        for dataset_name, target_df in target_datasets.items():
            target_df = target_df.reset_index(drop=True)

            # Identify the intrinsic features of the slope (geometry + soil parameters)
            slope_features = [col for col in target_df.columns if col not in cols_to_ignore_for_slope_id]

            # Assign a unique ID to each slope configuration
            slope_groups = target_df.groupby(slope_features, observed=False).ngroup()
            total_distinct_slopes = slope_groups.nunique()

            # GroupShuffleSplit ensures that all records with the same slope ID
            # end up in the train set or in the test set
            gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
            train_idx, test_idx = next(gss.split(target_df, groups=slope_groups))

            # Extract the train and test dataframes using the generated indices
            train_df = target_df.iloc[train_idx].reset_index(drop=True)
            test_df = target_df.iloc[test_idx].reset_index(drop=True)

            # Safety check: empty intersection between train and test slopes
            train_slopes = set(slope_groups.iloc[train_idx])
            test_slopes = set(slope_groups.iloc[test_idx])
            assert len(train_slopes.intersection(test_slopes)) == 0, "Leakage detected: shared slopes between train and test!"

            train_datasets_for_plot[dataset_name] = train_df
            test_datasets_for_plot[dataset_name] = test_df

            train_path = os.path.join(train_dir, f"{condition}_{dataset_name}.csv")
            test_path = os.path.join(test_dir, f"{condition}_{dataset_name}.csv")

            train_df.to_csv(train_path, index=False, float_format="%.10f")
            test_df.to_csv(test_path, index=False, float_format="%.10f")

            print(
                f"[{condition.upper()} - {dataset_name}] Slopes: {total_distinct_slopes} "
                f"(Train={len(train_slopes)}, Test={len(test_slopes)}) | "
                f"Samples: Total={len(target_df)} -> Train={len(train_df)}, Test={len(test_df)}"
            )

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Combine the drained datasets.
merge_datasets("D", "drained")

# Generate the training and test splits.
split_datasets()