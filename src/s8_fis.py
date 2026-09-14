# -*- coding: utf-8 -*-
"""Evaluate fuzzy inference systems and generate ranking figures."""

import os
import pickle
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from skfuzzy import control as ctrl

from config import (
    regressors,
    BASE_DIR,
    DATASET_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    TARGET_FOS,
    TARGET_SLIP_DEPTH,
    TARGET_ZWD_FINAL,
    TARGET_ZWU_FINAL,
)

# Inject compatibility structures to allow the deserialization of legacy scikit-learn GradientBoosting models.
gb_losses = types.ModuleType("sklearn.ensemble._gb_losses")


class IdentityLink:
    """Implement a dummy identity link class to satisfy legacy model dependencies."""
    def link(self, x): return x
    def inverse(self, x): return x


class LeastSquaresError:
    """Implement a compatibility stub for the legacy LeastSquaresError class."""
    K = 1
    link = IdentityLink()

    def __init__(self, *args, **kwargs): pass
    
    def __call__(self, y, pred, sample_weight=None): 
        # Compute the mean squared error between true and predicted values.
        return np.mean((y - pred) ** 2)
        
    def negative_gradient(self, y, pred, **k): 
        # Calculate the residual errors.
        return y - pred
        
    def update_terminal_regions(self, *args, **kwargs): pass
    
    def get_init_raw_predictions(self, X, estimator): 
        # Extract initial raw predictions and reshape the output array.
        return estimator.predict(X).reshape(-1, 1)


# Register the compatibility classes within the system modules.
gb_losses.LeastSquaresError = LeastSquaresError
sys.modules["sklearn.ensemble._gb_losses"] = gb_losses


def patch_monotonic_cst(model):
    """Traverse the model estimators and append missing monotonic-constraint attributes."""
    if hasattr(model, "estimators_"):
        # Flatten the estimator array and update each individual tree.
        for est in np.ravel(model.estimators_):
            if not hasattr(est, "monotonic_cst"):
                est.monotonic_cst = None
    else:
        # Update the base model directly if it lacks the attribute.
        if not hasattr(model, "monotonic_cst"):
            model.monotonic_cst = None
    return model


# Define global parameters for the fuzzy inference system.
TYPES = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
MF_TYPE = "trapmf" 
ALPHAS = [0.5, 0.6] #, 0.7, 0.8, 0.9, 1.0]


def mf(universe, points):
    """Generate a triangular or trapezoidal membership function over a specified universe array."""
    if MF_TYPE == "trimf": 
        return fuzz.trimf(universe, points)
    elif MF_TYPE == "trapmf": 
        return fuzz.trapmf(universe, points)
    raise ValueError("MF_TYPE must be 'trimf' or 'trapmf'")


# Initialize the fuzzy logic antecedents and consequents with their numerical ranges.
depth = ctrl.Antecedent(np.arange(0, 20.1, 0.1), "depth")
piezo = ctrl.Antecedent(np.arange(0, 20.1, 0.1), "piezo")
eff = ctrl.Consequent(np.arange(0, 1.01, 0.01), "eff")

d, p, e = depth.universe, piezo.universe, eff.universe

# Map linguistic labels to specific coordinate arrays for the 'depth' antecedent.
depth["superficial"] = mf(d, [0.0, 0.0, 1.0, 2.0])
depth["shallow"]     = mf(d, [1.0, 2.0, 3.0, 4.0])
depth["medium"]      = mf(d, [3.0, 4.0, 8.0, 9.0])
depth["deep"]        = mf(d, [8.0, 9.0, 15.0, 16.0])
depth["very_deep"]   = mf(d, [15.0, 16.0, 20.0, 20.0])

# Map linguistic labels to specific coordinate arrays for the 'piezo' antecedent.
piezo["high"]   = mf(p, [0.0, 0.0, 3.0, 4.0])
piezo["low"]    = mf(p, [3.0, 4.0, 12.0, 13.0])
piezo["absent"] = mf(p, [12.0, 13.0, 20.0, 20.0])

# Map linguistic labels to specific coordinate arrays for the 'eff' consequent.
eff["low"]      = mf(e, [0.00, 0.00, 0.125, 0.25])
eff["medium"]   = mf(e, [0.125, 0.2, 0.4, 0.5])
eff["high"]     = mf(e, [0.4, 0.50, 0.7, 0.8])
eff["veryhigh"] = mf(e, [0.75, 0.8, 1.0, 1.00])

depth_labels = ["superficial", "shallow", "medium", "deep", "very_deep"]
piezo_labels = ["high", "low", "absent"]


def build_type(TYPE: str):
    """Construct effectiveness matrices and map them to their respective mitigation subsystems."""
    if TYPE == "G1":
        # Define matrices for mitigation category G1.
        G11 = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        G12 = np.array([[0.5, 0.5, 0], [0.25, 0.25, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        G13 = np.array([[1, 0.5, 0.5], [0.5, 0.25, 0.25], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
        # Link the matrices to control systems and assign raw applicability scores.
        systems = {"G1.1": create_system(G11), "G1.2": create_system(G12), "G1.3": create_system(G13), "G1.4": create_system(G12.copy()), "G1.5": create_system(G13.copy())}
        app_raw = {"G1.1": 3.5, "G1.2": 4.0, "G1.3": 4.0, "G1.4": 3.5, "G1.5": 4.0}
        return systems, app_raw

    elif TYPE == "G2":
        # Define matrices for mitigation category G2.
        G21 = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        G22 = np.array([[0.25, 0.5, 0.5], [0.25, 0.5, 0.5], [0.5, 1, 1], [0.25, 0.5, 0.5], [0.25, 0.5, 0.5], [0, 0, 0]])
        G24 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [1, 1, 1], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0, 0, 0]])
        
        systems = {"G2.1": create_system(G21), "G2.2": create_system(G22), "G2.3": create_system(G22.copy()), "G2.4": create_system(G24)}
        app_raw = {"G2.1": 3.5, "G2.2": 3.5, "G2.3": 2.0, "G2.4": 4.0}
        return systems, app_raw

    elif TYPE == "G3a":
        # Define matrices for mitigation category G3a.
        G3a1 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0]])
        G3a2 = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0]])
        G3a3 = np.array([[0.5, 0.5, 1], [0.5, 0.5, 1], [0.25, 0.25, 0.5], [0, 0, 0], [0, 0, 0]])
        G3a4 = np.array([[0.5, 1, 1], [0.5, 1, 1], [0.25, 0.5, 0.5], [0, 0, 0], [0, 0, 0]])
        G3a5 = np.array([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        
        systems = {"G3a.1": create_system(G3a1), "G3a.2": create_system(G3a2), "G3a.3": create_system(G3a3), "G3a.4": create_system(G3a4), "G3a.5": create_system(G3a5)}
        app_raw = {"G3a.1": 4.0, "G3a.2": 4.0, "G3a.3": 4.0, "G3a.4": 4.0, "G3a.5": 3.0}
        return systems, app_raw

    elif TYPE == "G3b":
        # Define matrices for mitigation category G3b.
        G3b1 = np.array([[0.5, 0, 0], [0.5, 0, 0], [0.25, 0, 0], [0, 0, 0], [0, 0, 0]])
        G3b2 = np.array([[0.5, 0.25, 0], [0.5, 0.25, 0], [0.5, 0.25, 0], [0.5, 0.25, 0], [0, 0, 0]])
        G3b3 = np.array([[0, 0, 0], [0, 0, 0], [0.25, 0.5, 0], [0.25, 0.5, 0], [0.25, 0.5, 0]])
        G3b4 = np.array([[0, 0, 0], [0.5, 0.25, 0], [0.5, 0.25, 0], [1, 0.5, 0], [0.5, 0.25, 0]])
        G3b5 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 0], [1, 1, 0]])
        
        systems = {"G3b.1": create_system(G3b1), "G3b.2": create_system(G3b2), "G3b.3": create_system(G3b3), "G3b.4": create_system(G3b4), "G3b.5": create_system(G3b5)}
        app_raw = {"G3b.1": 2.5, "G3b.2": 2.5, "G3b.3": 2.0, "G3b.4": 2.0, "G3b.5": 1.5}
        return systems, app_raw

    elif TYPE == "G4":
        # Define matrices for mitigation category G4.
        G41 = np.array([[0, 0, 0], [0.25, 0.5, 0.5], [0.5, 1, 1], [0.25, 0.5, 0.5], [0, 0, 0]])
        G42 = np.array([[0, 0, 0], [0, 0, 0], [0.25, 0.5, 0.5], [0.5, 1, 1], [0.25, 0.5, 0.5]])
        G43 = np.array([[0, 0.5, 1], [0, 0.5, 1], [0, 0.25, 0.5], [0, 0, 0], [0, 0, 0]])
        G44 = np.array([[0, 0, 0], [0, 0, 0], [0.25, 0.5, 0.5], [0.5, 1, 1], [0.25, 0.5, 0.5], [0, 0, 0]])
        
        systems = {"G4.1": create_system(G41), "G4.2": create_system(G42), "G4.3": create_system(G43), "G4.4": create_system(G44)}
        app_raw = {"G4.1": 3.0, "G4.2": 3.0, "G4.3": 2.0, "G4.4": 2.0}
        return systems, app_raw

    elif TYPE == "G5":
        # Define matrices for mitigation category G5.
        G51 = np.array([[0, 0, 0], [0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [0.25, 0.5, 0.25], [0, 0, 0], [0, 0, 0]])
        G52 = np.array([[0.5, 0.5, 0.5], [1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0]])
        G53 = np.array([[0, 0, 0], [1, 1, 0.5], [0.5, 0.5, 0.25], [0, 0, 0], [0, 0, 0]])
        G54 = np.array([[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5], [0, 0, 0], [0, 0, 0]])
        
        systems = {"G5.1": create_system(G51), "G5.2": create_system(G52), "G5.3": create_system(G53), "G5.4": create_system(G54)}
        app_raw = {"G5.1": 3.5, "G5.2": 4.0, "G5.3": 3.5, "G5.4": 3.0}
        return systems, app_raw

    raise ValueError(f"Unsupported TYPE: {TYPE}")


def build_rules(matrix):
    """Generate a list of fuzzy rules by mapping matrix values to consequent linguistic terms."""
    rules = []
    
    # Iterate through every combination of antecedent labels and map the matrix value.
    for i, dlab in enumerate(depth_labels):
        for j, plab in enumerate(piezo_labels):
            val = matrix[i, j]
            
            # Translate numeric matrix values to qualitative labels.
            if val == 0: out = "low"
            elif val == 0.25: out = "medium"
            elif val == 0.5: out = "high"
            elif val == 1.0: out = "veryhigh"
            
            # Construct and append the formal fuzzy rule.
            rules.append(ctrl.Rule(depth[dlab] & piezo[plab], eff[out]))
            
    return rules


def create_system(matrix):
    """Instantiate a fuzzy control system environment and assign the matrix."""
    system = ctrl.ControlSystem(build_rules(matrix))
    system.matrix = matrix
    return system


def normalize_dict(d):
    """Scale dictionary values."""
    vals = np.array(list(d.values()))
    return {k: v / vals.max() for k, v in d.items()}


def compute_eff(system, depth_val, piezo_val):
    """Execute the fuzzy simulation and compute the crisp effectiveness score."""
    try:
        sim = ctrl.ControlSystemSimulation(system)
        
        # Confine input values to the predefined universes of discourse.
        sim.input["depth"] = np.clip(depth_val, 0.0, 20.0)
        sim.input["piezo"] = np.clip(piezo_val, 0.0, 20.0)
        
        sim.compute()
        return float(sim.output["eff"])
    except ValueError:
        return 0.0


def surface_data(system, depth_range=(0, 20), piezo_range=(0, 20), n_depth=100, n_piezo=100):
    """Evaluate a fuzzy system over a continuous 2D grid to generate surface coordinates."""
    D, P = np.linspace(*depth_range, n_depth), np.linspace(*piezo_range, n_piezo)
    Z = np.zeros((n_depth, n_piezo))
    sim = ctrl.ControlSystemSimulation(system)

    # Compute the system output at each node in the grid.
    for i, d in enumerate(D):
        for j, p in enumerate(P):
            sim.input["depth"], sim.input["piezo"] = d, p
            try:
                sim.compute()
                Z[i, j] = sim.output["eff"]
            except ValueError:
                Z[i, j] = 0.0

    # Generate the meshgrid format required for 3D plotting.
    Dg, Pg = np.meshgrid(D, P, indexing="ij")
    return Dg, Pg, Z


def get_crisp_eff(matrix, d, p):
    """Extract a rigid, discrete effectiveness value based on strict input intervals."""
    depth_bins = [0, 1.0, 3.0, 8.0, 15.0, 20.0]
    piezo_bins = [0, 3.0, 12.0, 20.0]
    
    # Locate the matrix index corresponding to the input value intervals.
    di = min(max(np.digitize(d, depth_bins) - 1, 0), 4)
    pj = min(max(np.digitize(p, piezo_bins) - 1, 0), 2)
    return matrix[di, pj]


def crispy_surface_from_matrix(matrix, depth_range=(0, 20), piezo_range=(0, 20), n_depth=200, n_piezo=200):
    """Evaluate a discrete matrix mapping over a 2D grid to generate stepped surface coordinates."""
    D, P = np.linspace(*depth_range, n_depth), np.linspace(*piezo_range, n_piezo)
    Z = np.zeros((n_depth, n_piezo))
    
    for i, d in enumerate(D):
        for j, p in enumerate(P):
            Z[i, j] = get_crisp_eff(matrix, d, p)
    
    Dg, Pg = np.meshgrid(D, P, indexing="ij")
    return Dg, Pg, Z


def final_ranking(depth_val, piezo_val, alpha=0.8):
    """Compute and sort the final rankings combining fuzzy and applicability scores."""
    rows = []
    
    for name, system in ALL_SYSTEMS.items():
        e = compute_eff(system, depth_val, piezo_val)
        # Compute the weighted composite score.
        rows.append((name, e, ALL_APP_RAW[name], alpha * e + (1 - alpha) * ALL_APP_NORM[name]))
        
    return sorted(rows, key=lambda x: x[3], reverse=True)


def plot_surfaces(name, system, out_dir_f, out_dir_c, point=None):
    """Plot 3D projections of the fuzzy and discrete effectiveness surfaces."""
    # Define a color palette for the surface mapping.
    custom_cmap = LinearSegmentedColormap.from_list("crispy_fuzzy_cmap", [(0.0, "red"), (0.25, "orange"), (0.5, "yellow"), (1.0, "green")])

    def crop_manual(img_path):
        img = Image.open(img_path).convert("RGB")
        bbox = (60, 60, img.width - 10, img.height - 50)
        if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
            img.crop(bbox).save(img_path)

    def draw_single_surface(Dg, Pg, Z, pt_z, filepath):
        """Build the 3D axis object."""
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "3d"})
        
        ax.plot_surface(Dg, Pg, Z, cmap=custom_cmap, vmin=0, vmax=1, edgecolor="none", alpha=0.65, zorder=0)

        # Overlay the exact predicted coordinate point on the surface mesh.
        if point is not None:
            pt_d, pt_p = point
            z_text_top, z_arrow_start, z_arrow_end, z_text_bot = 1.15, 1.08, pt_z + 0.1, pt_z - 0.15

            # Draw an indicator arrow and the point geometry.
            ax.plot([pt_d, pt_d], [pt_p, pt_p], [z_arrow_start, z_arrow_end], color="blue", linewidth=3, zorder=100)
            ax.plot([pt_d], [pt_p], [z_arrow_end], marker="v", color="blue", markersize=10, zorder=101)
            ax.plot([pt_d], [pt_p], [pt_z], marker="o", markeredgecolor="black", markerfacecolor="blue", markersize=12, zorder=102)

            bbox_props = dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5)
            val_str = f"eff: {pt_z:.2f}" if pt_z > 0 else "eff: 0.00"
            coord_str = f"($z_s$, $z_w^{{final}}$)\n({pt_d:.2f}, {pt_p:.2f})"
            
            # Place the numerical annotation texts.
            ax.text(pt_d, pt_p, z_text_top, val_str, color="black", fontsize=14, fontweight="bold", ha="center", va="center", zorder=104, bbox=bbox_props)
            ax.text(pt_d, pt_p, z_text_bot, coord_str, color="black", fontsize=14, fontweight="bold", ha="center", va="top", zorder=104, bbox=bbox_props)

        # Format axis limits, labels, and ticks.
        ax.set_xlabel(r"$z_{s}$", fontsize=24, labelpad=15)
        ax.set_ylabel(r"$z_{w}^{final}$", fontsize=24, labelpad=15)
        ax.set_zlabel("Effectiveness", fontsize=24, labelpad=10)
        ax.set_zlim(0, 1)
        ax.tick_params(axis="both", labelsize=24)
        ax.tick_params(axis="z", labelsize=24)

        ax.text(10, 10, 1.55, name, color="black", fontsize=18, fontweight="bold", ha="center", va="top", zorder=104, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5))

        fig.savefig(filepath, bbox_inches="tight", pad_inches=0.5)
        plt.close(fig)
        crop_manual(filepath)

    # Process the continuous fuzzy surface.
    Dg_f, Pg_f, Z_f = surface_data(system)
    pt_z_f = compute_eff(system, point[0], point[1]) if point else None
    draw_single_surface(Dg_f, Pg_f, Z_f, pt_z_f, os.path.join(out_dir_f, f"{name}.png"))

    # Process the discrete crisp matrix surface.
    Dg_c, Pg_c, Z_c = crispy_surface_from_matrix(system.matrix)
    pt_z_c = get_crisp_eff(system.matrix, point[0], point[1]) if point else None
    draw_single_surface(Dg_c, Pg_c, Z_c, pt_z_c, os.path.join(out_dir_c, f"{name}.png"))


def plot_ranking_bars(results, path):
    """Plot a grouped bar chart showing the final rankings."""
    names, eff_values = [r[0] for r in results], [r[1] for r in results]
    app_values, score_values = [ALL_APP_NORM[n] for n in names], [r[3] for r in results]
    x, width, text_color = np.arange(len(names)), 0.3, "#2C3E50"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(axis="y", linestyle="--", alpha=0.6, color="#BDC3C7", zorder=0)

    b1 = ax.bar(x - width - 0.01, eff_values, width, label="Effectiveness", color="#4A90E2", zorder=3, edgecolor="white", linewidth=1)
    b2 = ax.bar(x, app_values, width, label="Applicability", color="#F39C12", zorder=3, edgecolor="white", linewidth=1)
    b3 = ax.bar(x + width + 0.01, score_values, width, label="Final score", color="#E74C3C", zorder=3, edgecolor="white", linewidth=1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BDC3C7")
    ax.spines["bottom"].set_color("#BDC3C7")

    ax.set_ylabel("Score", fontsize=16, fontweight="bold", color=text_color, labelpad=5)
    ax.set_xlabel("Guidelines", fontsize=16, fontweight="bold", color=text_color, labelpad=0)
    ax.tick_params(axis="both", labelsize=16, colors=text_color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=16, color=text_color, rotation=15, ha="right")
    ax.set_xlim(x[0] - width * 2, x[-1] + width * 2)
    ax.margins(x=0)

    def add_values(bars):
        """Iterate over bar objects and overlay numerical value annotations."""
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                s = f"{int(height)}" if height == 1.0 else f"{height:.2f}"
                ax.annotate(s, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom", fontsize=16, color=text_color, fontweight="500")

    add_values(b1)
    add_values(b2)
    add_values(b3)

    legend = fig.legend(handles=[b1, b2, b3], labels=["Effectiveness", "Applicability", "Final score"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.975), fontsize=16, frameon=False)
    for text in legend.get_texts(): text.set_color(text_color)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _predict_target_for_row(row: pd.Series, base_dir: str, slope: str, model_name: str, target_name: str):
    """Load a target model, extract required features, and compute the prediction for a single row."""
    pkl_path = os.path.join(base_dir, "results", slope, target_name, model_name, f"{model_name}_inference.pkl")
    
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    if model_name == 'GradientBoostingRegressor':
        regressor = patch_monotonic_cst(d.get("model"))
    else:
        regressor = d.get("model")
        
    features, scaler = d["features"], d.get("scaler")

    # Extract the input features required by the target model and scale them.
    X_full = row[features].to_frame().T.reindex()
    X_scaled = scaler.transform(X_full) 

    # Return the scalar prediction value and the list of feature names.
    return float(np.squeeze(regressor.predict(X_scaled))), features


def infer_one_sample(row: pd.Series, slope:str, model: str, base_dir: str):
    """Compute the predictions for all four target for a given input row."""    
    fos_pred, feats_fos = _predict_target_for_row(row, base_dir, slope, model, TARGET_FOS)
    zs_pred, feats_zs = _predict_target_for_row(row, base_dir, slope, model, TARGET_SLIP_DEPTH)
    zwuf_pred, feats_zwuf = _predict_target_for_row(row, base_dir, slope, model, TARGET_ZWU_FINAL)
    zwdf_pred, feats_zwdf = _predict_target_for_row(row, base_dir, slope, model, TARGET_ZWD_FINAL)

    # Output predictions, feature sets, and true values.
    return fos_pred, zs_pred, zwuf_pred, zwdf_pred, feats_fos, feats_zs, feats_zwuf, feats_zwdf, {
        "fos": row.get(TARGET_FOS), "zs": row.get(TARGET_SLIP_DEPTH), "zwufinal": row.get("zwufinal"), "zwdfinal": row.get("zwdfinal")
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    
    slope = 'drained'

    # Store fuzzy systems and applicability scores.
    ALL_SYSTEMS, ALL_APP_RAW = {}, {}
    for TYPE in TYPES:
        sys_dict, app_raw = build_type(TYPE)
        ALL_SYSTEMS.update(sys_dict)
        ALL_APP_RAW.update(app_raw)

    ALL_APP_NORM = normalize_dict(ALL_APP_RAW)
    TOPK = 6

    in_csv = os.path.join(DATASET_DIR, "real_cases.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = pd.read_csv(in_csv)

    for reg_model in regressors:
        model_name = reg_model.__class__.__name__
        out_csv = os.path.join(RESULTS_DIR, f"predictions_{model_name}.csv")
        out_txt = os.path.join(RESULTS_DIR, f"features_{model_name}.txt")

        # Track the sets of input features used by each model for each target.
        used_feats = {"drained": {"fos": set(), "zs": set(), "zwufinal": set(), "zwdfinal": set()}}
                      # "undrained": {"fos": set(), "zs": set(), "zwufinal": set(), "zwdfinal": set()}}

        df_out = pd.DataFrame({
            "Tr": df["Return period of precipitation [years]"],
            "fos_true": np.nan, "zs_true": np.nan, "zwuf_true": np.nan, "zwdf_true": np.nan,
            "fos_pred": np.nan, "zs_pred": np.nan, "zwuf_pred": np.nan, "zwdf_pred": np.nan,
        })

        for idx, row in df.iterrows():
            f_p, z_p, u_p, d_p, f_f, z_f, u_f, d_f, true = infer_one_sample(row, slope, model_name, BASE_DIR)

            # Record predictions and corresponding true values.
            df_out.loc[idx, ["fos_pred", "zs_pred", "zwuf_pred", "zwdf_pred"]] = [float(f_p), z_p, u_p, d_p]
            df_out.loc[idx, ["fos_true", "zs_true", "zwuf_true", "zwdf_true"]] = [true.get("fos"), true.get("zs"), true.get("zwufinal"), true.get("zwdfinal")]

            used_feats['drained']["fos"].update(f_f)
            used_feats['drained']["zs"].update(z_f)
            used_feats['drained']["zwufinal"].update(u_f)
            used_feats['drained']["zwdfinal"].update(d_f)

            # Average the piezometric heads to calculate the aggregate input coordinate for the fuzzy system.
            zwf_pred = (u_p + d_p) / 2
            sample_base_prefix = f"{row['Return period of precipitation [years]']}"

            if sample_base_prefix == 'nan':
                sample_base_prefix = 'TR -'
            else:
                sample_base_prefix = 'TR ' + sample_base_prefix.split('.')[0]


            for alpha in ALPHAS:
                alpha_str = f"alpha_{str(alpha).replace('.', '')}"
                top = final_ranking(z_p, zwf_pred, alpha=alpha)[:TOPK]

                alpha_base_dir = os.path.join(FIGURES_DIR, "fis", sample_base_prefix, alpha_str)
                ranking_dir = os.path.join(alpha_base_dir, "ranking")
                fuzzy_dir = os.path.join(alpha_base_dir, "fuzzy_surface")
                crispy_dir = os.path.join(alpha_base_dir, "crispy_surface")

                os.makedirs(ranking_dir, exist_ok=True)
                os.makedirs(fuzzy_dir, exist_ok=True)
                os.makedirs(crispy_dir, exist_ok=True)

                plot_ranking_bars(top, os.path.join(ranking_dir, f"{sample_base_prefix}.png"))

                for gname, e, a_raw, score in top:
                    plot_surfaces(gname, ALL_SYSTEMS[gname], fuzzy_dir, crispy_dir, point=(z_p, zwf_pred))

        df_out.to_csv(out_csv, index=False)

        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("Drained\n")
            f.write("FoS: " + ", ".join(sorted(used_feats["drained"]["fos"])) + "\n")
            f.write("zs: "  + ", ".join(sorted(used_feats["drained"]["zs"]))  + "\n")
            f.write("zwf upstream: " + ", ".join(sorted(used_feats["drained"]["zwufinal"])) + "\n")
            f.write("zwf downstream: " + ", ".join(sorted(used_feats["drained"]["zwdfinal"])) + "\n")
            # f.write("Undrained\n")
            # f.write("FoS: " + ", ".join(sorted(used_feats["undrained"]["fos"])) + "\n")
            # f.write("zs: "  + ", ".join(sorted(used_feats["undrained"]["zs"]))  + "\n")
            # f.write("zwf upstream: " + ", ".join(sorted(used_feats["undrained"]["zwufinal"])) + "\n")
            # f.write("zwf downstream: " + ", ".join(sorted(used_feats["undrained"]["zwdfinal"])) + "\n")