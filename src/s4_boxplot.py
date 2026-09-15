# -*- coding: utf-8 -*-
"""Generate combined boxplots and Wilcoxon significance matrices."""

import itertools
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from PIL import Image

from config import (
    ALL_SLOPES,
    ALL_TARGETS,
    FIGURES_DIR,
    METRIC_LABELS,
    MODEL_ABBREVIATIONS,
    MODEL_COLORS,
    RESULTS_DIR,
    TARGET_LABELS,
    SIGNIFICANCE_COLORS
)


def load_target_distributions(slope, target, metric):
    """Load the evaluation scores for a specified target variable."""
    csv_path = os.path.join(RESULTS_DIR, slope, target, f"scores_{metric}.csv")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Scores file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)

    model_order = list(dict.fromkeys(MODEL_ABBREVIATIONS.values()))
    
    distributions = {model: np.full(len(dataframe), np.nan, dtype=float) for model in model_order}

    for original_model_name in dataframe.columns:
        short_model_name = MODEL_ABBREVIATIONS.get(original_model_name, original_model_name)

        if short_model_name not in distributions:
            continue

        # Convert the column data to numeric format.
        distributions[short_model_name] = pd.to_numeric(dataframe[original_model_name], errors="coerce").to_numpy(dtype=float)

    return distributions


# =========================================================
# MODELS RANKING
# =========================================================
def get_ranked_models(distributions, metric):
    """Rank the models by the median value of a metric."""
    model_order = list(dict.fromkeys(MODEL_ABBREVIATIONS.values()))
    
    medians = {}

    for model in model_order:
        values = np.asarray(distributions[model], dtype=float)
        
        # Filter out non-finite values.
        finite_values = values[np.isfinite(values)]
        
        if finite_values.size == 0:
            medians[model] = -np.inf if metric == "r2" else np.inf
        else:
            # Compute the median of the valid scores.
            medians[model] = np.median(finite_values)

    # Return the models sorted by median values in descending order for R2, and ascending order for MAE.
    if metric == "r2":
        return sorted(model_order, key=lambda m: (medians[m], -model_order.index(m)), reverse=True)
    else:
        return sorted(model_order, key=lambda m: (medians[m], model_order.index(m)))


# =========================================================
# WILCOXON + HOLM
# =========================================================
def get_significance_matrix(distributions, ordered_models):
    """Calculate pairwise Wilcoxon signed-rank test statistics and apply the Holm correction."""
    num_models = len(ordered_models)
    
    # Initialize matrices for significance levels and text labels.
    significance_matrix = np.zeros((num_models, num_models), dtype=int)
    text_matrix = np.full((num_models, num_models), "ns", dtype=object)

    # Generate all unique pairwise combinations of the models.
    pairs = list(itertools.combinations(range(num_models), 2))
    raw_p_values = []

    for idx1, idx2 in pairs:
        m1, m2 = ordered_models[idx1], ordered_models[idx2]
        v1, v2 = np.asarray(distributions[m1], dtype=float), np.asarray(distributions[m2], dtype=float)
        
        # Extract the subset of finite values shared by both arrays.
        mask = np.isfinite(v1) & np.isfinite(v2)
        first, second = v1[mask], v2[mask]

        if first.size == 0:
            p_value = 1.0
        else:
            differences = first - second
            if np.allclose(differences, 0.0):
                p_value = 1.0
            else:
                # Perform the Wilcoxon signed-rank test on the paired distributions.
                try:
                    _, p_value = stats.wilcoxon(first, second, alternative="two-sided", zero_method="wilcox", method="auto")
                except TypeError:
                    _, p_value = stats.wilcoxon(first, second, alternative="two-sided", zero_method="wilcox")
                except ValueError:
                    p_value = 1.0

                if not np.isfinite(p_value):
                    p_value = 1.0

        raw_p_values.append(float(p_value))

    # Apply the Holm method to adjust the raw p-values.
    adjusted_p_values = multipletests(raw_p_values, alpha=0.05, method="holm")[1] if raw_p_values else np.array([], dtype=float)

    # Map the adjusted p-values to corresponding significance symbols and numeric levels.
    for pair_idx, (idx1, idx2) in enumerate(pairs):
        adj_p = float(adjusted_p_values[pair_idx])

        if adj_p <= 0.02:
            symbol, color_level = "**", 2
        elif adj_p <= 0.05:
            symbol, color_level = "*", 1
        else:
            symbol, color_level = "ns", 0

        text_matrix[idx1, idx2] = text_matrix[idx2, idx1] = symbol
        significance_matrix[idx1, idx2] = significance_matrix[idx2, idx1] = color_level

    # Assign default values to the diagonal elements of the matrices.
    for i in range(num_models):
        text_matrix[i, i] = "-"
        significance_matrix[i, i] = 0

    return significance_matrix, text_matrix


def draw_boxplot(axis, distributions_by_target, metric):
    """Draw the four target distributions on a single boxplot axis."""
    positions = []
    all_data = []
    colors = []

    gap_between_targets = 8
    box_width = 0.7

    for target_index, target in enumerate(ALL_TARGETS):
        distributions = distributions_by_target[target]
        
        # Retrieve the sorted list of models.
        ordered_models = get_ranked_models(distributions=distributions, metric=metric)

        # Compute the starting position on the x-axis for the current target.
        base_position = target_index * gap_between_targets

        for model_index, model in enumerate(ordered_models):
            positions.append(base_position + model_index)

            values = np.asarray(distributions[model], dtype=float)
            finite_values = values[np.isfinite(values)]

            if finite_values.size == 0:
                finite_values = np.array([np.nan], dtype=float)

            all_data.append(finite_values)
            colors.append(MODEL_COLORS[model])

    # Plot the distributions as boxplots.
    boxplot = axis.boxplot(
        all_data, positions=positions, widths=box_width, patch_artist=True, showfliers=True,
        medianprops={"color": "black", "linewidth": 2}, whiskerprops={"linewidth": 1.5}, capprops={"linewidth": 1.5}
    )

    # Apply the colors and transparency to the boxplot patches.
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    model_order = list(dict.fromkeys(MODEL_ABBREVIATIONS.values()))
    centers = [target_index * gap_between_targets + (len(model_order) - 1) / 2 for target_index in range(len(ALL_TARGETS))]

    # Configure the x-axis and y-axis ticks, labels, and gridlines.
    axis.set_xticks(centers)
    axis.set_xticklabels([TARGET_LABELS[target] for target in ALL_TARGETS], fontsize=28)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=28)
    axis.tick_params(axis="x", labelsize=28)
    axis.tick_params(axis="y", labelsize=28)
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    # Place the legend at the top of the axis.
    handles = [Line2D([0], [0], color=MODEL_COLORS[model], lw=8, alpha=0.6) for model in model_order]
    axis.legend(handles, model_order, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=len(model_order), fontsize=20, frameon=False)


def draw_heatmap(axis, significance_matrix, text_matrix, ordered_models, target_label):
    """Draw the pairwise significance matrix on the supplied axis."""
    bounds = [0, 1, 2, 3]
    normalization = plt.cm.colors.BoundaryNorm(bounds, SIGNIFICANCE_COLORS.N)
    number_of_models = len(ordered_models)

    # Plot the significance matrix as an image.
    axis.imshow(significance_matrix, cmap=SIGNIFICANCE_COLORS, norm=normalization)
    axis.set_xticks(np.arange(number_of_models))
    axis.set_yticks(np.arange(number_of_models))
    axis.set_xticklabels(ordered_models, fontsize=18, rotation=45)
    axis.set_yticklabels(ordered_models, fontsize=18)
    axis.set_title(target_label, fontsize=22, pad=10)

    # Write the symbols in each cell of the heatmap.
    for row in range(number_of_models):
        for column in range(number_of_models):
            text_color = "white" if significance_matrix[row, column] >= 2 else "black"
            axis.text(column, row, text_matrix[row, column], ha="center", va="center", color=text_color, fontsize=12, fontweight="bold")

    axis.set_xticks(np.arange(-0.5, number_of_models, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, number_of_models, 1), minor=True)
    axis.grid(which="minor", color="white", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)


def draw_heatmaps(heatmap_axes, distributions_by_target, metric):
    """Generate and plot the Wilcoxon significance matrix for each target variable."""
    for target_index, target in enumerate(ALL_TARGETS):
        distributions = distributions_by_target[target]
        ordered_models = get_ranked_models(distributions=distributions, metric=metric)
        significance_matrix, text_matrix = get_significance_matrix(distributions=distributions, ordered_models=ordered_models)

        draw_heatmap(axis=heatmap_axes[target_index], significance_matrix=significance_matrix,
                     text_matrix=text_matrix, ordered_models=ordered_models, target_label=TARGET_LABELS[target])


def create_figure(slope, metric, distributions_by_target):
    """Create a figure with four boxplot groups and four Wilcoxon matrices."""
    
    figure = plt.figure(figsize=(24, 10), dpi=300)
    grid = figure.add_gridspec(2, 4, width_ratios=[1.8, 1.8, 1.0, 1.0], wspace=0.3, hspace=0.7)

    boxplot_axis = figure.add_subplot(grid[:, 0:2])
    draw_boxplot(axis=boxplot_axis, distributions_by_target=distributions_by_target, metric=metric)

    heatmap_axes = [figure.add_subplot(grid[0, 2]), figure.add_subplot(grid[0, 3]),
                    figure.add_subplot(grid[1, 2]), figure.add_subplot(grid[1, 3])]

    draw_heatmaps(heatmap_axes=heatmap_axes, distributions_by_target=distributions_by_target, metric=metric)

    legend_row_1 = [Line2D([0], [0], marker="s", color="w", markerfacecolor="#eeeeee", markersize=15, label=r"ns ($p > 0.05$)")]
    legend_row_2 = [Line2D([0], [0], marker="s", color="w", markerfacecolor="#9ecae1", markersize=15, label=r"* ($p \leq 0.05$)"),
                    Line2D([0], [0], marker="s", color="w", markerfacecolor="#4292c6", markersize=15, label=r"** ($p \leq 0.02$)")]

    first_legend = figure.legend(handles=legend_row_1, loc="upper center", bbox_to_anchor=(0.775, 0.54), ncol=1, fontsize=20, frameon=False)
    figure.legend(handles=legend_row_2, loc="upper center", bbox_to_anchor=(0.775, 0.50), ncol=2, fontsize=20, frameon=False)
    figure.add_artist(first_legend)

    output_directory = os.path.join(FIGURES_DIR, "holdout", slope)
    os.makedirs(output_directory, exist_ok=True)

    output_path = os.path.join(output_directory, f"boxplot_wilcoxon_{metric}.png")
    figure.savefig(output_path)
    plt.close(figure)

    with Image.open(output_path) as image:
        left = 400
        top = 250
        right = max(left + 1, image.width - 500)
        bottom = max(top + 1, image.height - 130)

        image.crop((left, top, right, bottom)).save(output_path)



if __name__ == "__main__":
    """Generate a combined figure for each metric."""
    for slope in ALL_SLOPES:
        for metric in ['r2', 'mae']:
            distributions_by_target = {}

            for target in ALL_TARGETS:
                try:
                    distributions_by_target[target] = load_target_distributions(slope=slope, target=target, metric=metric)
                except FileNotFoundError:
                    break

            if len(distributions_by_target) != len(ALL_TARGETS):
                continue

            create_figure(slope=slope, metric=metric, distributions_by_target=distributions_by_target)
