# -*- coding: utf-8 -*-
"""Generate the training and test datasets."""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import BASE_DIR, DATASET_DIR, RANDOM_SEED, TARGET_FOS


def merge_datasets(prefix, condition_name):
    """Combine intermediate datasets while keeping only the columns they share."""
    
    dataframes = []

    # Load all relevant files based on the specified prefix (e.g., D.1 and D.2).
    for file_name in sorted(os.listdir(DATASET_DIR)):
        if file_name.startswith("D.1") or file_name.startswith("D.2"):
            full_path = os.path.join(DATASET_DIR, file_name)
            dataframes.append(pd.read_csv(full_path))

    # Stop execution if no files were found.
    if not dataframes:
        return

    # Identify columns that are present in every loaded file.
    # This preserves the column order of the first file.
    common_cols = [
        col
        for col in dataframes[0].columns
        if all(col in df.columns for df in dataframes[1:])
    ]
    
    # Track all unique columns across all files to report what is removed.
    all_cols = []
    for df in dataframes:
        for col in df.columns:
            if col not in all_cols:
                all_cols.append(col)

    missing_cols = [col for col in all_cols if col not in common_cols]

    # Keep only the shared columns for each dataset.
    indexed_dataframes = []
    for dataset_index, df in enumerate(dataframes, start=1):
        df = df.loc[:, common_cols].copy()
        indexed_dataframes.append(df)

    # Combine all datasets into a single DataFrame.
    combined = pd.concat(indexed_dataframes, ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)

    # Remove any rows containing missing values.
    combined = combined.dropna(subset=combined.columns).reset_index(drop=True)

    # Save the merged dataset.
    output_path = os.path.join(DATASET_DIR, "D_drained.csv")
    combined.to_csv(output_path, index=False, header=True, float_format="%.10f")

    print(f"Saved {output_path}: {len(combined)} samples, {len(combined.columns)} columns, removed {len(missing_cols)} non-shared features.")


# =============================================================
# CREATE TRAIN AND TEST DATASETS FOR EACH TARGET
# =============================================================
def split_datasets():
    """Split the data into training and testing sets for each target variable."""
    
    # Create directories for the output files if they do not exist.
    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    dataset_paths = {
        "drained": os.path.join(DATASET_DIR, "D_drained.csv"),
        "undrained": os.path.join(DATASET_DIR, "U_undrained.csv"),
    }

    # Process both the drained and undrained datasets.
    for condition, dataset_path in dataset_paths.items():
        df = pd.read_csv(dataset_path)

        # Create specific datasets according to the physical constraints of the variables.
        target_datasets = {
            # The full dataset is suitable for FoS and z_s models.
            "fos": df.copy(),
            # The downstream piezometric model requires a non-zero initial downstream depth.
            "zwd": df[df[COL_ZWD_INIT].ne(0)].copy(),
            # The upstream piezometric model requires a non-zero initial upstream depth.
            "zwu": df[df[COL_ZWU_INIT].ne(0)].copy(),
        }

        # Perform the data split for each target variable.
        for dataset_name, target_df in target_datasets.items():
            target_df = target_df.reset_index(drop=True)

            # Stratify the data to maintain the ratio of stable (FoS >= 1) to unstable (FoS < 1) slopes.
            stratify = (target_df[COL_FOS] >= 1.0).astype(int)

            # Split the data.
            train_df, test_df = train_test_split(
                target_df,
                test_size=TEST_SIZE,
                random_state=RANDOM_SEED,
                stratify=stratify,
            )

            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            # Save the final training and testing datasets.
            train_path = os.path.join(train_dir, f"{condition}_{dataset_name}.csv")
            test_path = os.path.join(test_dir, f"{condition}_{dataset_name}.csv")

            train_df.to_csv(train_path, index=False, float_format="%.10f")
            test_df.to_csv(test_path, index=False, float_format="%.10f")

            print(f"{condition}_{dataset_name}: {len(target_df)} samples -> train={len(train_df)}, test={len(test_df)}")


if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Combine the drained datasets.
merge_datasets("D", "drained")

# Generate the training and test splits.
split_datasets()

# Process the external dataset.
preprocess_ext(os.path.join(BASE_DIR, "data", "real_cases.xlsx"))