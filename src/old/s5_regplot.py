# -*- coding: utf-8 -*-
"""Create observed-versus-predicted regression plots from saved models."""

import os
import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from PIL import Image
from sklearn.metrics import r2_score, mean_absolute_error

from config import (
    ALL_SLOPES,
    ALL_TARGETS,
    BASE_DIR,
    FIGURES_DIR,
    MODEL_ABBREVIATIONS,
    RESULTS_DIR,
    TARGET_TO_DATASET,
    TARGET_LABELS,
    TARGET_UNITS,
)


# =========================================================
# FUNCTIONS
# =========================================================
def get_model_directories(stats_dir):
    """Retrieve directories containing the corresponding test model pickle files."""
    if not os.path.isdir(stats_dir):
        return []

    model_directories = []

    for item_name in sorted(os.listdir(stats_dir)):
        item_path = os.path.join(stats_dir, item_name)

        if not os.path.isdir(item_path):
            continue

        pkl_path = os.path.join(item_path, f"{item_name}_test.pkl")

        if os.path.isfile(pkl_path):
            model_directories.append((item_name, item_path, pkl_path))

    return model_directories


def load_saved_test_predictions(pkl_path):
    """Extract the true and predicted target arrays from the serialized test model."""
    with open(pkl_path, "rb") as file:
        saved_data = pickle.load(file)

    required_keys = ["y_holdout_true", "y_holdout_pred"]

    missing_keys = [key for key in required_keys if key not in saved_data]
    if missing_keys:
        raise KeyError(f"Missing keys in {pkl_path}: {missing_keys}")

    y_true = np.asarray(saved_data["y_holdout_true"], dtype=float).reshape(-1)
    y_pred = np.asarray(saved_data["y_holdout_pred"], dtype=float).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch in {pkl_path}: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Apply a boolean mask to isolate finite numeric values.
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite_mask]
    y_pred = y_pred[finite_mask]

    if len(y_true) == 0:
        raise ValueError(f"No valid true/pred pairs found in {pkl_path}")

    # Retrieve pre-calculated metrics or compute them from the extracted arrays.
    r2 = saved_data.get("holdout_test_r2", r2_score(y_true, y_pred))
    mae = saved_data.get("holdout_test_mae", mean_absolute_error(y_true, y_pred))

    return (y_true, y_pred, float(r2), float(mae))


def format_metric(value):
    """Format numerical metric values to three decimal places for plot annotations."""
    formatted = f"{value:.3f}"
    return formatted.replace("1.000", "1.0").replace("0.000", "0.0")


def create_regression_plot(y_true, y_pred, r2, mae, target, model_name, output_path):
    """Generate and save an observed-versus-predicted regression scatter plot."""
    
    # Calculate the absolute error between true and predicted values.
    error = np.abs(y_true - y_pred)
    plot_df = pd.DataFrame({"True": y_true, "Pred": y_pred})

    figure, axis = plt.subplots(figsize=(5.5, 5.5), dpi=300)

    # Normalize the absolute error array to define the scatter point transparency levels.
    point_alpha = (error - error.min()) / (error.max() - error.min() + 1e-8)
    point_alpha = np.clip(point_alpha, 0.10, 1.0)

    # Define the coordinate limits and ticks based on the specific target variable.
    min_lim = 0.0
    if target == "Factor of safety [-]":
        max_lim = 6.0
    else:
        max_lim = 60.0

    tick_spacing = max(1, int(np.floor(max_lim / 6)))
    ticks = np.arange(min_lim, max_lim + tick_spacing, tick_spacing).astype(int)

    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xlim(min_lim, max_lim)
    axis.set_ylim(min_lim, max_lim)

    # Draw the reference regression line on the axis.
    sns.regplot(
        x="True",
        y="Pred",
        data=plot_df,
        ax=axis,
        scatter=False,
        truncate=False,
        line_kws={
            "color": "#f97306",
            "linewidth": 1.5,
            "alpha": 0.75,
        },
    )

    # Plot the individual data points.
    axis.scatter(
        plot_df["True"],
        plot_df["Pred"],
        facecolors="#2ecc71",
        edgecolors="#001f3f",
        alpha=point_alpha,
        s=70,
        linewidths=1,
        zorder=2,
    )

    # Adjust the drawing order of the axis collections.
    for collection in axis.collections:
        collection.set_zorder(1)

    def padded_ytick_formatter(value, position):
        """Format a vertical tick label."""
        value_int = int(round(value))
        if value_int < 10:
            return f"{value_int:>5}"
        if value_int < 100:
            return f"{value_int:>4}"
        return f"{value_int}"

    def padded_xtick_formatter(value, position):
        """Format a horizontal tick label."""
        value_int = int(round(value))
        if value_int < 10:
            return f"{value_int:>3}"
        if value_int < 100:
            return f"{value_int:>2}"
        return f"{value_int}"

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(padded_xtick_formatter))
    axis.yaxis.set_major_formatter(ticker.FuncFormatter(padded_ytick_formatter))

    # Apply a spatial offset to the x-axis labels.
    if target == "Factor of safety [-]":
        offset = transforms.ScaledTranslation(-6 / 72, 0, figure.dpi_scale_trans)
        for label in axis.get_xticklabels():
            label.set_transform(label.get_transform() + offset)

    # Construct the axis labels and the plot title strings.
    x_label = rf"{TARGET_LABELS[target]} " rf"{TARGET_UNITS[target]} " r"${true}$"
    y_label = rf"{TARGET_LABELS[target]} " rf"{TARGET_UNITS[target]} " r"${pred}$"
    
    r2_text = format_metric(r2)
    mae_text = format_metric(mae)
    
    r2_title = rf"$R^2: {r2_text} " rf"\quad — \quad$"
    mae_title = rf"$MAE: {mae_text}$"

    axis.tick_params(axis="both", labelsize=20)
    axis.set_xlabel(x_label, fontsize=20, labelpad=0)
    axis.set_ylabel(y_label, fontsize=20, labelpad=-10)
    axis.set_title(f"{r2_title} {mae_title}", fontsize=20, pad=5)
    axis.set_aspect("equal", "box")

    figure.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    # Crop the surrounding whitespace from the saved image.
    with Image.open(output_path) as image:
        left = 40
        top = 30
        right = max(left + 1, image.width - 45)
        bottom = max(top + 1, image.height - 45)

        image.crop((left, top, right, bottom)).save(output_path)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    
    for slope in ALL_SLOPES:
        for target in ALL_TARGETS:
            stats_dir = os.path.join(RESULTS_DIR, slope, target)

            # Retrieve available model paths for the current target.
            model_directories = get_model_directories(stats_dir)

            if not model_directories:
                print(f"No _test.pkl found in: {stats_dir}")
                continue

            target_abbreviation = TARGET_TO_DATASET[target]

            # Process the prediction data for each identified model.
            for model_name, model_directory, pkl_path in model_directories:
                try:
                    (y_true, y_pred, r2, mae) = load_saved_test_predictions(pkl_path)
                except Exception as error:
                    print(f"{model_name}: {error}")
                    continue

                model_abbreviation = MODEL_ABBREVIATIONS.get(model_name, model_name)
                safe_model_name = model_abbreviation.replace(" ", "_").replace("/", "_")

                output_directory = os.path.join(FIGURES_DIR, "holdout", slope, target)
                
                os.makedirs(output_directory, exist_ok=True)
                plot_filename = f"{safe_model_name}_{target_abbreviation}.png"
                output_path = os.path.join(output_directory, plot_filename)

                create_regression_plot(
                    y_true=y_true,
                    y_pred=y_pred,
                    r2=r2,
                    mae=mae,
                    target=target,
                    model_name=model_name,
                    output_path=output_path,
                )