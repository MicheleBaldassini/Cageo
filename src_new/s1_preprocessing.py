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
    TARGET_SLIP_DEPTH,
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


def clean_cell(val):
    """Clean and format string values to ensure valid numeric conversion."""
    # Convert the value to a string and strip any surrounding whitespace.
    val = str(val).strip()
    
    # Correct specific invalid punctuation sequences.
    val = re.sub(r"(\d+\.\d+)\.,", r"\1.", val)
    val = val.replace("..", ".")
    
    # Standardize all potential decimal separators to a dot.
    val = val.replace(",", ".")
    
    # Resolve instances with multiple decimal separators (e.g., "12.45.89").
    # Retain the first separator to preserve the primary decimal scale and eliminate the rest.
    if val.count(".") > 1:
        parts = val.split(".")
        val = parts[0] + "." + "".join(parts[1:])
        
    return val


def print_complete_slope_cases(slopes, dataset_name):
    """Count slopes having the requested base/rainfall combinations."""
    ignored = ALL_TARGETS + [
        RETURN_PERIOD_COLUMN, ACCUMULATED_RAIN_COLUMN, RAIN_DURATION_COLUMN
    ]
    slope_features = [col for col in slopes.columns if col not in ignored]
    base = (-1, 0)
    rain_100 = {(tr, 100) for tr in (30, 200, 500)}
    rain_15_30 = {(tr, tw) for tr in (30, 200, 500) for tw in (15, 30)}
    counts = {"4 samples": 0, "7 samples": 0, "3 rain only": 0, "6 rain only": 0}
    total_slopes = 0
    for _, group in slopes.groupby(slope_features, sort=False, dropna=False):
        total_slopes += 1
        cases = set()
        for tr, hw, tw in group[
            [RETURN_PERIOD_COLUMN, ACCUMULATED_RAIN_COLUMN, RAIN_DURATION_COLUMN]
        ].itertuples(index=False, name=None):
            # Tr=-1 and Tr=0 both identify the base; the stored Tr is preserved.
            if hw == 0 and tr in (-1, 0):
                cases.add(base)
            elif hw > 0:
                cases.add((tr, tw))
            else:
                cases.add((tr, hw, tw))  # invalid rain record cannot complete a grid
        if len(group) == 4 and cases == (rain_100 | {base}):
            counts["4 samples"] += 1
        elif len(group) == 7 and cases == (rain_15_30 | {base}):
            counts["7 samples"] += 1
        elif len(group) == 3 and cases == rain_100:
            counts["3 rain only"] += 1
        elif len(group) == 6 and cases == rain_15_30:
            counts["6 rain only"] += 1
    print(
        f"[{dataset_name}] Pendii totali={total_slopes}; \n"
        f"caso base + TR 30/200/500, duration=100 (4 campioni)={counts['4 samples']}; \n"
        f"caso base + TR 30/200/500, duration=15 e 30 (7 campioni)={counts['7 samples']}; \n"
        f"senza caso base: 3 piogge a 100 ore={counts['3 rain only']}, \n"
        f"6 piogge a 15/30 ore={counts['6 rain only']}; \n"
        f"altri/incompleti={total_slopes - sum(counts.values())}."
    )
    return counts


def preprocess_ext(excel_path):
    """Process the file containing real-world cases and save it in CSV format."""
    
    all_data = np.empty((0, len(COLUMNS)))
    xls = pd.ExcelFile(excel_path)

    # Read data from all sheets in the file.
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)

        # Remove unnamed columns.
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()

        # Isolate inputs and targets, then combine them.
        inputs = df.iloc[1:, 1:-4].values
        targets = df.iloc[1:, -4:].values
        data = np.concatenate([inputs, targets], axis=1)

        all_data = np.vstack([all_data, data])

    slopes = pd.DataFrame(all_data, columns=COLUMNS.values())

    # Map the short column symbols back to their full descriptive names.
    inv_map = {v: k for k, v in COLUMNS.items()}
    slopes.columns = slopes.columns.map(inv_map)

    # Clean text values and convert invalid entries to NaN.
    slopes = slopes.astype(str)
    slopes = slopes.apply(lambda s: s.map(clean_cell))
    slopes = slopes.replace("-", np.nan)
    slopes = slopes.replace("err", np.nan)
    slopes = slopes.replace(" ", np.nan)
    
    # Convert the cleaned data to numeric format.
    for col in slopes.columns:
        slopes[col] = pd.to_numeric(slopes[col], errors="coerce")

    # Remove columns that contain entirely NaN values.
    nan_cols = slopes.columns[slopes.isna().all()].tolist()
    file_stem = "real_cases"
    slopes = slopes.drop(columns=nan_cols, errors="ignore").reset_index(drop=True)

    # Save the final real cases dataset.
    output_path = os.path.join(DATASET_DIR, f"{file_stem}.csv")
    slopes.to_csv(output_path, index=False, header=True, float_format="%.10f")

    print(f"{len(slopes)} samples, {len(slopes.columns)} columns.")


def preprocess(data_dir, slope="drained"):
    """Process each source file for a specific slope condition."""

    for file_name in os.listdir(data_dir):
        if (file_name.startswith("D") and (slope == "drained")) or (
            file_name.startswith("U") and (slope == "undrained")):
            all_data = np.empty((0, len(COLUMNS)))

            excel_path = os.path.join(data_dir, file_name)
            xls = pd.ExcelFile(excel_path)

            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                df.columns = df.columns.str.strip()

                inputs = df.iloc[1:, 1:-4].values
                targets = df.iloc[1:, -4:].values
                data = np.concatenate([inputs, targets], axis=1)

                all_data = np.vstack([all_data, data])

            slopes = pd.DataFrame(all_data, columns=COLUMNS.keys())
            length = slopes.shape[0]

            slopes = slopes.astype(str)
            slopes = slopes.apply(lambda s: s.map(clean_cell))
            pd.set_option("future.no_silent_downcasting", True)
            # slopes = slopes.replace([" ", "-", "err", "", "nan", "NaN", "None"], np.nan) #.infer_objects(copy=False)
            slopes = slopes.replace(["err", "-"], np.nan) #.infer_objects(copy=False)

            slopes[TARGET_FOS] = slopes[TARGET_FOS].replace(fos_corrections)

            slopes[RETURN_PERIOD_COLUMN] = slopes[RETURN_PERIOD_COLUMN].fillna(-1)
            slopes[ACCUMULATED_RAIN_COLUMN] = slopes[ACCUMULATED_RAIN_COLUMN].fillna(0)
            slopes[RAIN_DURATION_COLUMN] = slopes[RAIN_DURATION_COLUMN].fillna(0)


            ZWU_INIT = "Initial piezometric surface depth - upstream [m]"
            ZWD_INIT = "Initial piezometric surface depth - downstream [m]"

            mask_hw_zero = slopes[ACCUMULATED_RAIN_COLUMN] == 0
            slopes.loc[mask_hw_zero, TARGET_ZWU_FINAL] = slopes.loc[mask_hw_zero, ZWU_INIT]
            slopes.loc[mask_hw_zero, TARGET_ZWD_FINAL] = slopes.loc[mask_hw_zero, ZWD_INIT]

            # slopes.loc[slopes[ZWU_INIT] == 0.0, TARGET_ZWU_FINAL] = 0.0
            # slopes.loc[slopes[ZWD_INIT] == 0.0, TARGET_ZWD_FINAL] = 0.0

            # mask_zwu = mask_hw_zero & slopes[TARGET_ZWU_FINAL] != slopes[ZWU_INIT]
            # mask_zwd = mask_hw_zero & slopes[TARGET_ZWD_FINAL] != slopes[ZWD_INIT]
            slopes.loc[mask_hw_zero, TARGET_ZWU_FINAL] = slopes.loc[mask_hw_zero, ZWU_INIT]
            slopes.loc[mask_hw_zero, TARGET_ZWD_FINAL] = slopes.loc[mask_hw_zero, ZWD_INIT]

            zwu_init_num.notna() & zwu_final_num.notna() & ~np.isclose(zwu_init_num, zwu_final_num)
            mask_zwu = slopes[TARGET_ZWU_FINAL] > slopes[ZWU_INIT]
            slopes.loc[mask_zwu, TARGET_ZWU_FINAL] = slopes.loc[mask_zwu, ZWU_INIT]

            mask_zwd = slopes[TARGET_ZWD_FINAL] > slopes[ZWD_INIT]
            slopes.loc[mask_zwd, TARGET_ZWD_FINAL] = slopes.loc[mask_zwd, ZWD_INIT]

            


            slopes = slopes.dropna(subset=[TARGET_FOS]).reset_index(drop=True)
            slopes = slopes.dropna(subset=[TARGET_SLIP_DEPTH]).reset_index(drop=True)
            slopes = slopes.dropna(subset=[TARGET_ZWU_FINAL]).reset_index(drop=True)
            slopes = slopes.dropna(subset=[TARGET_ZWD_FINAL]).reset_index(drop=True)
            
            nan_cols = slopes.columns[slopes.isna().all()].tolist()
            slopes = slopes.drop(columns=nan_cols, errors="ignore").reset_index(drop=True)


            if file_name.startswith("D.1"):
                file_stem = "D.1_drained"
            elif file_name.startswith("D.2"):
                file_stem = "D.2_drained"
            elif file_name.startswith("U"):
                file_stem = "U_undrained"

            print(f"\n{file_stem}")
            columns_with_nans = slopes.columns[slopes.isna().any()].tolist()
            if columns_with_nans:
                print(f"columns containing NaN values:\n{columns_with_nans}")

            # ============================================================
            # NUOVA LOGICA: TRACCIAMENTO VALORI DIVENTATI NaN
            # ============================================================
            nan_origins = {}
            
            for col in slopes.columns:
                # 1. Salva la colonna originale
                original_col = slopes[col].copy()
                
                # 2. Conversione numerica
                slopes[col] = pd.to_numeric(slopes[col], errors="coerce")
                
                # 3. Trova i NaN post-conversione
                mask_nan = slopes[col].isna()
                
                if mask_nan.any():
                    # Estrai i valori unici originali responsabili del NaN
                    bad_vals = original_col[mask_nan].unique()
                    
                    # Formatta l'output per renderlo leggibile
                    formatted_vals = []
                    for val in bad_vals:
                        if pd.isna(val) or str(val).lower() == "nan":
                            formatted_vals.append("NaN")
                        elif str(val) == " ":
                            formatted_vals.append("space")
                        elif str(val) == "":
                            formatted_vals.append("empty")
                        elif str(val).isspace():
                            formatted_vals.append(f"whitespace_({repr(val)})")
                        else:
                            formatted_vals.append(str(val))
                    
                    # Rimuovi eventuali duplicati generati dalla formattazione e salva
                    nan_origins[col] = list(set(formatted_vals))

            if nan_origins:
                print(f"columns containing NaN values after numeric conversion:\n{list(nan_origins.keys())}")
                print("Original values that caused NaN:")
                for col, vals in nan_origins.items():
                    print(f"  - {col}: {vals}")
            # ============================================================

            # print(f"{file_name} removed: {length - slopes.shape[0]}")
            
            slopes = slopes[(slopes[TARGET_FOS] <= 30)]
            slopes = slopes.drop_duplicates()

            mask_to_drop = (slopes[RAIN_DURATION_COLUMN] > 0) & (slopes[RAIN_DURATION_COLUMN] < 15)
            slopes = slopes[~mask_to_drop].reset_index(drop=True)

            # rows_with_nans = slopes[slopes.isna().any(axis=1)]
            
            # # Output the count and the contents of the rows containing missing values for inspection.
            # if not rows_with_nans.empty:
            #     print(f"[{file_stem}] Found {len(rows_with_nans)} rows containing NaN values:\n{rows_with_nans}")

            output_path = os.path.join(DATASET_DIR, f"{file_stem}.csv")
            slopes.to_csv(output_path, index=False, header=True, float_format="%.10f")

# def preprocess(data_dir, slope="drained"):
#     """Process each source file for a specific slope condition."""
#     fos_corrections = {
#         1725.0: 1.725,
#         1634.0: 1.634,
#         156.0: 1.560,
#         1492.0: 1.492,
#         1460.0: 1.460,
#     }

#     for file_name in os.listdir(data_dir):
#         if (file_name.startswith("D") and (slope == "drained")) or (
#             file_name.startswith("U") and (slope == "undrained")):
#             all_data = np.empty((0, len(COLUMNS)))

#             excel_path = os.path.join(data_dir, file_name)
#             xls = pd.ExcelFile(excel_path)

#             for sheet in xls.sheet_names:
#                 df = xls.parse(sheet)
#                 df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
#                 df.columns = df.columns.str.strip()

#                 inputs = df.iloc[1:, 1:-4].values
#                 targets = df.iloc[1:, -4:].values
#                 data = np.concatenate([inputs, targets], axis=1)

#                 all_data = np.vstack([all_data, data])

#             slopes = pd.DataFrame(all_data, columns=COLUMNS.keys())
#             length = slopes.shape[0]

#             slopes = slopes.astype(str)
#             slopes = slopes.apply(lambda s: s.map(clean_cell))
#             pd.set_option("future.no_silent_downcasting", True)
#             # slopes = slopes.replace([" ", "-", "err", "", "nan", "NaN", "None"], np.nan) #.infer_objects(copy=False)
#             slopes = slopes.replace(["err", "-"], np.nan) #.infer_objects(copy=False)

#             slopes[TARGET_FOS] = slopes[TARGET_FOS].replace(fos_corrections)

#             slopes[RETURN_PERIOD_COLUMN] = slopes[RETURN_PERIOD_COLUMN].fillna(-1)
#             slopes[ACCUMULATED_RAIN_COLUMN] = slopes[ACCUMULATED_RAIN_COLUMN].fillna(0)
#             slopes[RAIN_DURATION_COLUMN] = slopes[RAIN_DURATION_COLUMN].fillna(0)

#             ZWU_INIT = "Initial piezometric surface depth - upstream [m]"
#             ZWD_INIT = "Initial piezometric surface depth - downstream [m]"

#             mask_hw_zero = slopes[ACCUMULATED_RAIN_COLUMN] == 0
#             slopes.loc[mask_hw_zero, TARGET_ZWU_FINAL] = slopes.loc[mask_hw_zero, ZWU_INIT]
#             slopes.loc[mask_hw_zero, TARGET_ZWD_FINAL] = slopes.loc[mask_hw_zero, ZWD_INIT]

#             slopes.loc[slopes[ZWU_INIT] == 0.0, TARGET_ZWU_FINAL] = 0.0
#             slopes.loc[slopes[ZWD_INIT] == 0.0, TARGET_ZWD_FINAL] = 0.0

#             mask_zwu = slopes[TARGET_ZWU_FINAL] > slopes[ZWU_INIT]
#             slopes.loc[mask_zwu, TARGET_ZWU_FINAL] = slopes.loc[mask_zwu, ZWU_INIT]

#             mask_zwd = slopes[TARGET_ZWD_FINAL] > slopes[ZWD_INIT]
#             slopes.loc[mask_zwd, TARGET_ZWD_FINAL] = slopes.loc[mask_zwd, ZWD_INIT]

#             slopes = slopes.dropna(subset=[TARGET_FOS]).reset_index(drop=True)
#             slopes = slopes.dropna(subset=[TARGET_SLIP_DEPTH]).reset_index(drop=True)
#             slopes = slopes.dropna(subset=[TARGET_ZWU_FINAL]).reset_index(drop=True)
#             slopes = slopes.dropna(subset=[TARGET_ZWD_FINAL]).reset_index(drop=True)
            
#             nan_cols = slopes.columns[slopes.isna().all()].tolist()
#             slopes = slopes.drop(columns=nan_cols, errors="ignore").reset_index(drop=True)


#             if file_name.startswith("D.1"):
#                 file_stem = "D.1_drained"
#             elif file_name.startswith("D.2"):
#                 file_stem = "D.2_drained"
#             elif file_name.startswith("U"):
#                 file_stem = "U_undrained"

#             print(file_stem)
#             columns_with_nans = slopes.columns[slopes.isna().any()].tolist()
#             if columns_with_nans:
#                 print(f"columns containing NaN values:\n{columns_with_nans}")

#             for col in slopes.columns:
#                 slopes[col] = pd.to_numeric(slopes[col], errors="coerce")

#             columns_with_nans_after = slopes.columns[slopes.isna().any()].tolist()
#             if columns_with_nans_after:
#                 print(f"columns containing NaN values after numeric conversion:\n{columns_with_nans_after}")

#             # print(f"{file_name} removed: {length - slopes.shape[0]}")
            
#             slopes = slopes[(slopes[TARGET_FOS] <= 30)]
#             slopes = slopes.drop_duplicates()

#             mask_to_drop = (slopes[RAIN_DURATION_COLUMN] > 0) & (slopes[RAIN_DURATION_COLUMN] < 15)
#             slopes = slopes[~mask_to_drop].reset_index(drop=True)

         

#             rows_with_nans = slopes[slopes.isna().any(axis=1)]
            
#             # # Output the count and the contents of the rows containing missing values for inspection.
#             if not rows_with_nans.empty:
#                 print(f"Found {len(rows_with_nans)} rows containing NaN values:\n{rows_with_nans}")

#             output_path = os.path.join(DATASET_DIR, f"{file_stem}.csv")
#             slopes.to_csv(output_path, index=False, header=True, float_format="%.10f")
#             # print(f"{len(slopes)} samples, removed {nan_cols} empty columns.")


def merge_datasets(prefix, condition_name):
    """Combine drained datasets."""
    dataframes = []
    file_names = []

    # Load all drained condition datasets
    for file_name in sorted(os.listdir(DATASET_DIR)):
        if file_name.startswith("D.1_drained") or \
           file_name.startswith("D.2_drained"):
            full_path = os.path.join(DATASET_DIR, "preprocessed", file_name)

            df = pd.read_csv(full_path)
            
            # df = df.astype(str)
            # df = df[~df.isin(["err"]).any(axis=1)].reset_index(drop=True)

            mask_err = df.astype(str).apply(lambda col: col.str.strip() == "err").any(axis=1)
            df = df[~mask_err].reset_index(drop=True)
            # df = df.apply(lambda s: s.map(clean_cell))
            # df = df.drop(columns="Undrained Shear Strength [kPa]", errors="ignore").reset_index(drop=True)
            # df = df.replace(["err"], np.nan)
            # 1. Salviamo una copia per ricordare l'aspetto originale dei dati testuali
            # df_str = df.copy()

            # Convert the cleaned data to numeric format.
            for col in df.columns:
                df[col] = pd.to_numeric(df[col]) #, errors="coerce")
                
                # # 2. Identifichiamo i valori che sono stati "coercizzati" a NaN.
                # # Escludiamo le classiche stringhe che rappresentano già un dato mancante.
                # is_nan_now = df[col].isna()
                # was_not_nan_str = ~df_str[col].astype(str).str.strip().isin(["nan", "None", "", "NA", "<NA>"])
                
                # # Estraiamo i valori unici "colpevoli" del NaN
                # bad_values = df_str.loc[is_nan_now, col].unique()
                
                # if len(bad_values) > 0:
                #     print(f"[{file_name}] Colonna '{col}' -> Stringhe convertite in NaN: {bad_values}")

            # fos_corrections = {
            #     1725.0: 1.725,
            #     1634.0: 1.634,
            #     156.0: 1.560,
            #     1492.0: 1.492,
            #     1460.0: 1.460,
            # }
            # df[TARGET_FOS] = df[TARGET_FOS].replace(fos_corrections)

            nan_counts = df.isna().sum()
            remaining_nans = nan_counts[nan_counts > 0]

            if not remaining_nans.empty:
                print("-" * 40)
                print(f"\nRemaining NaN {file_name}")
                print(remaining_nans)
                print("-" * 40)

            dataframes.append(df)
            file_names.append(file_name)

    if not dataframes:
        return

    # Identify the columns common to all loaded datasets
    # common_cols = [
    #     col
    #     for col in dataframes[0].columns
    #     if all(col in df.columns for df in dataframes[1:])
    # ]

    # # Collect all unique columns
    # all_cols = []
    # for df in dataframes:
    #     for col in df.columns:
    #         if col not in all_cols:
    #             all_cols.append(col)

    # missing_cols = [col for col in all_cols if col not in common_cols]

    # # Filter columns
    # indexed_dataframes = []
    # for dataset_index, df in enumerate(dataframes):
    #     df = df.loc[:, common_cols].copy()
    #     indexed_dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True).drop_duplicates().reset_index(drop=True)

    print_complete_slope_cases(combined, "D_drained")

    output_path = os.path.join(DATASET_DIR, "D_drained.csv")
    combined.to_csv(output_path, index=False, header=True, float_format="%.10f")
    # print(f"Saved {output_path}: {len(combined)} samples, {len(combined.columns)} columns, removed {len(missing_cols)} non-shared features.")


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
        "drained": os.path.join(DATASET_DIR, MODE, "D_drained.csv"),
        "undrained": os.path.join(DATASET_DIR, MODE, "U_undrained.csv"),
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
        target_datasets["zwd"] = target_datasets["zwd"][target_datasets["zwd"][ACCUMULATED_RAIN_COLUMN] != 0].copy()
        target_datasets["zwu"] = target_datasets["zwu"][target_datasets["zwu"][ACCUMULATED_RAIN_COLUMN] != 0].copy()

        # plot_unique_value_counts(target_datasets, condition, split="all")

        train_datasets_for_plot = {}
        test_datasets_for_plot = {}

        # Group Shuffle Split for each target dataset
        for dataset_name, target_df in target_datasets.items():
            target_df = target_df.reset_index(drop=True)

            # Identify the intrinsic features of the slope (geometry + soil parameters)
            slope_features = [col for col in target_df.columns if col not in cols_to_ignore_for_slope_id]

            # Assign a unique ID to each slope configuration
            slope_groups = target_df.groupby(slope_features, observed=True).ngroup()
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

            print(f"[{condition.upper()} - {dataset_name}] Slopes: {total_distinct_slopes} "
                f"(Train={len(train_slopes)}, Test={len(test_slopes)}) | "
                f"Samples: Total={len(target_df)} -> Train={len(train_df)}, Test={len(test_df)}")


if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# # Preprocessing source files
# preprocess(os.path.join(BASE_DIR, "data"), "drained")
# preprocess(os.path.join(BASE_DIR, "data"), "undrained")

# Merge datasets
# merge_datasets("D", "drained")

MODE = "real_freq"

d1_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", MODE, "D.1_drained.csv"))
print(len(d1_df))
d1_df = d1_df.dropna()

print(len(d1_df))

d2_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", MODE, "D.2_drained.csv"))
print(len(d2_df))
d2_df = d2_df.dropna()

print(len(d2_df))

d_df = pd.concat([d1_df, d2_df])
print(len(d_df))
d_df.to_csv(os.path.join(BASE_DIR, "dataset", MODE, 'D_drained.csv'), index=False, float_format="%.10f")

u_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", MODE, "U_undrained.csv"))
print(len(u_df))
u_df = u_df.dropna()
print(len(u_df))
u_df.to_csv(os.path.join(BASE_DIR, "dataset", MODE, 'U_undrained.csv'), index=False, float_format="%.10f")


# Split datasets (80% train / 20% test).
split_datasets()

# # Process the external dataset.
# preprocess_ext(os.path.join(BASE_DIR, "data", "real_cases.xlsx"))