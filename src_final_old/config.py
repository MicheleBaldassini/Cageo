# -*- coding: utf-8 -*-
"""Experiment constants shared by the project scripts."""

import os
from matplotlib.colors import ListedColormap

from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module=r"xgboost(\..*)?")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="X does not have valid feature names, but LGBMRegressor was fitted with feature names")

from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor


# Define directory paths.
BASE_DIR = os.path.abspath(os.path.join(__file__, "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_R2_DIR = os.path.join(BASE_DIR, "results_r2")
FIGURES_DIR = os.path.join(BASE_DIR, "fig")

NORMALIZATION = "z-score"
RANDOM_SEED = 32651
SAMPLE_WEIGHTS = False

# Soil conditions and the prediction targets.
ALL_SLOPES = ["undrained", "drained"]
ALL_TARGETS = [
    "Factor of safety [-]",
    "Depth of slip surface [m]",
    "Final Piezometric surface depth - upstream [m]",
    "Final Piezometric surface depth - downstream [m]",
]

TARGET_FOS, TARGET_SLIP_DEPTH, TARGET_ZWU_FINAL, TARGET_ZWD_FINAL = ALL_TARGETS

# Columns related to precipitation parameters.
RETURN_PERIOD_COLUMN = "Return period of precipitation [years]"
ACCUMULATED_RAIN_COLUMN = "Accumulated precipitation [mm]"
RAIN_DURATION_COLUMN = "Precipitation duration [hrs]"
RAINFALL_COLUMNS = [
    RETURN_PERIOD_COLUMN,
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
]

# Map the full target descriptions to their respective datasets.
TARGET_TO_DATASET = {
    "Factor of safety [-]": "fos",
    "Depth of slip surface [m]": "zs",
    "Final Piezometric surface depth - upstream [m]": "zwu",
    "Final Piezometric surface depth - downstream [m]": "zwd",
}

# Map target variables to their mathematical formatting symbols for plotting.
TARGET_LABELS = {
    "Factor of safety [-]": r"$FoS$",
    "Depth of slip surface [m]": r"$z_{s}$",
    "Final Piezometric surface depth - upstream [m]": r"$z_{wu}^{final}$",
    "Final Piezometric surface depth - downstream [m]": r"$z_{wd}^{final}$",
}

# Map target variables to their physical units.
TARGET_UNITS = {
    "Factor of safety [-]": r"$[-]$",
    "Depth of slip surface [m]": r"$[m]$",
    "Final Piezometric surface depth - upstream [m]": r"$[m]$",
    "Final Piezometric surface depth - downstream [m]": r"$[m]$",
}

# Map standard regressor class names to abbreviations.
MODEL_ABBREVIATIONS = {
    "GradientBoostingRegressor": "GB",
    "RandomForestRegressor": "RF",
    "XGBRegressor": "XGB",
    "LGBMRegressor": "LGBM",
    "LightGBMRegressor": "LGBM",
    "DecisionTreeRegressor": "DT",
    "KNeighborsRegressor": "k-NN",
}

# Categorical colors for each models.
MODEL_COLORS = {
    "GB": "#4C72B0",
    "RF": "#55A868",
    "XGB": "#C44E52",
    "LGBM": "#8172B2",
    "DT": "#CCB974",
    "k-NN": "#64B5CD",
}

# Define labels for performance metrics.
METRIC_LABELS = {
    "r2": r"$R^2$",
    "mae": "MAE",
}

# Map raw input feature names to their respective mathematical symbols.
NAME_TO_SYMBOL = {
    "Unit weight [kN/m3]": r"$\gamma$",
    "Effective cohesion [kPa]": r"$c'$",
    "Effective friction angle [°]": r"$\phi'$",
    "Undrained Shear Strength [kPa]": r"$s_u$",
    "Saturated permeability [m/s]": r"$k_{sat}$",
    "Soil Type [-]": r"$ST$",
    "Slope angle [°]": r"$\alpha$",
    "Slope length [m]": r"$L$",
    "Total length [m]": r"$B$",
    "Slope height [m]": r"$H$",
    "Total height Upstream [m]": r"$h_u$",
    "Total height Downstream [m]": r"$h_d$",
    "Soil depth upstream [m]": r"$h_{Su}$",
    "Soil depth downstream [m]": r"$h_{Sd}$",
    "Bedrock depth upstream [m]": r"$h_{Bu}$",
    "Bedrock depth downstream [m]": r"$h_{Bd}$",
    "Initial piezometric surface depth - upstream [m]": r"$z_{wu}^{init}$",
    "Initial piezometric surface depth - downstream [m]": r"$z_{wd}^{init}$",
    "Return period of precipitation [years]": r"$T_r$",
    "Accumulated precipitation [mm]": r"$h_w$",
    "Precipitation duration [hrs]": r"$t_w$",
}

# Build a discrete colormap representing significance threshold levels for statistical tests.
SIGNIFICANCE_COLORS = ListedColormap([
    "#eeeeee",  # Not significant.
    "#9ecae1",  # p <= 0.05.
    "#4292c6",  # p <= 0.02.
])


# The regression models.
regressors = [
    tree.DecisionTreeRegressor(random_state=RANDOM_SEED),
    KNeighborsRegressor(),
    RandomForestRegressor(n_jobs=1, random_state=RANDOM_SEED),
    GradientBoostingRegressor(random_state=RANDOM_SEED),
    XGBRegressor(n_jobs=1, random_state=RANDOM_SEED),
    LGBMRegressor(verbose=-1, n_jobs=1, random_state=RANDOM_SEED)
]


# Define the hyperparameter search space configurations for nested cross-validation tuning.
parameters_reg = {
    "DecisionTreeRegressor": {
        "max_depth": [None, 3], #, 5, 10],
        "min_samples_split": [2, 4], #, 6],
        # "min_samples_leaf": [1, 2, 3],
        # "max_features": [None, "sqrt", "log2"],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
        "leaf_size": [20, 30, 40],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_features": [1.0, "sqrt", "log2"],
        "bootstrap": [True],
    },
    "GradientBoostingRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [1, 2],
        "max_features": [None, "sqrt"],
    },
    "XGBRegressor": {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.1, 0.2, 0.3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 2, 3],
        "gamma": [0, 0.05, 0.1],
    },
    "LGBMRegressor": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1, 0.15],
        "num_leaves": [20, 31, 50],
        "max_depth": [-1, 10, 20],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 0.05, 0.1],
        "reg_lambda": [0.0, 0.05, 0.1],
    },
}