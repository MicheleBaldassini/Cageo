# -*- coding: utf-8 -*-
"""Shared utilities."""

import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import shap
import logging
from PIL import Image
import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, RegressorMixin, clone

from plots.beeswarm import beeswarm
from plots.waterfall import waterfall
from config import (
    BASE_DIR,
    NORMALIZATION,
    RETURN_PERIOD_COLUMN,
    ACCUMULATED_RAIN_COLUMN,
    RAIN_DURATION_COLUMN,
    ALL_TARGETS
)


def sanitize_params(params):
    """
    Ensure all extracted model parameters are formatted as standard Python types,
    preventing runtime errors during JSON serialization.
    """
    sanitized = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            sanitized[k] = v
        elif isinstance(v, np.integer):
            sanitized[k] = int(v)
        elif isinstance(v, np.floating):
            sanitized[k] = float(v)
        else:
            sanitized[k] = str(v)
    return sanitized


def get_slope_groups(df):
    # Group instances based on slope geometry and geotechnical characteristics.
    cols_to_ignore = [
        RETURN_PERIOD_COLUMN,
        ACCUMULATED_RAIN_COLUMN,
        RAIN_DURATION_COLUMN,
    ] + ALL_TARGETS
    slope_features = [col for col in df.columns if col not in cols_to_ignore]
    return df.groupby(slope_features, observed=False).ngroup()


def get_train_processed_indices(train_df, target_name, slope_name, random_seed=42):
    # Subsample the dataset by isolating early stages of rainfall accumulation.
    # num_samples_to_keep = 2 # if target_name == TARGET_FOS else 2

    # cols_to_ignore = [
    #     RETURN_PERIOD_COLUMN,
    #     ACCUMULATED_RAIN_COLUMN,
    #     RAIN_DURATION_COLUMN,
    # ]
    # subset_cols = [col for col in train_df.columns if col not in cols_to_ignore]

    # filtered_train = (
    #     train_df.groupby(subset_cols, group_keys=False)[train_df.columns]
    #     .apply(
    #         lambda g: g.sort_values(
    #             by=ACCUMULATED_RAIN_COLUMN, ascending=True
    #         ).head(num_samples_to_keep)
    #     )
    # )

    # return filtered_train.index
    return train_df.index


def make_scaler(normalization=NORMALIZATION):
    """Make the scikit-learn scaling object."""
    if normalization == "min-max":
        return MinMaxScaler()
    if normalization == "z-score":
        return StandardScaler()
    raise ValueError(f"Unknown normalization method: {normalization}")


def get_sample_weights(y, target_name, decimals=1):
    # Compute inverse frequency sample weights.
    if target_name == TARGET_FOS:
        y_series = pd.Series(np.round(y, decimals=decimals))
    else:
        y_series = pd.Series(y)

    val_counts = y_series.value_counts()

    counts_per_sample = y_series.map(val_counts)

    weights = 1.0 / counts_per_sample

    weights = weights / np.mean(weights)

    return weights.to_numpy()


class AutoWeightedRegressor(BaseEstimator, RegressorMixin):
    # Custom meta-estimator designed to dynamically compute and integrate 
    # inverse frequency sample weights during the optimization process.
    def __init__(self, estimator, target_name, decimals=2):
        self.estimator = estimator
        self.target_name = target_name
        self.decimals = decimals

    def fit(self, X, y, **kwargs):
        self.estimator_ = clone(self.estimator)
        model_name = self.estimator_.__class__.__name__

        if model_name != "KNeighborsRegressor":
            weights = get_sample_weights(
                y, target_name=self.target_name, decimals=self.decimals
            )
            self.estimator_.fit(X, y, sample_weight=weights, **kwargs)
        else:
            self.estimator_.fit(X, y, **kwargs)

        return self

    def predict(self, X):
        return self.estimator_.predict(X)



def build_explainer(model, model_name, X_background, feature_names, seed):
    """Initialize and return the appropriate SHAP Explainer."""
    np.random.seed(seed)
    
    if model_name == "KNeighborsRegressor":
        # Generate a representative background sample using the SHAP utility.
        background = shap.sample(X_background, 500, random_state=seed)
        explainer = shap.KernelExplainer(model.predict, background)
    else:
        # Use the TreeExplainer for tree-based ensemble models.
        explainer = shap.TreeExplainer(model, data=X_background, feature_names=list(feature_names))
        
    return explainer


def build_explanation(explainer, model_name, X_samples, X_samples_raw, feature_names):
    """Build a SHAP Explanation object using an already initialized explainer."""
    if len(X_samples) == 0:
        raise ValueError("No samples available to explain.")

    n_samples = len(X_samples)

    if model_name == "KNeighborsRegressor":
        raw_vals = explainer.shap_values(X_samples)
        
        shap_values = raw_vals[0] if isinstance(raw_vals, list) else np.asarray(raw_vals)
        base_val = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        base_values = np.full(n_samples, base_val)
    else:
        try:
            tree_exp = explainer(X_samples, check_additivity=False)
        except TypeError:
            tree_exp = explainer(X_samples)

        shap_values = tree_exp.values[0] if isinstance(tree_exp.values, list) else np.asarray(tree_exp.values)
        
        base_values = np.asarray(tree_exp.base_values)
        if base_values.ndim == 0 or base_values.size == 1:
            base_values = np.full(n_samples, float(np.asarray(base_values).reshape(-1)[0]))

        # Standardize the dimensions of the SHAP values.
        if shap_values.ndim == 3:
            shap_values = shap_values[..., 0]
        elif shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

    return shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=X_samples_raw,
        feature_names=list(feature_names),
    )

def save_beeswarm_plot(explanation, output_path, seed):
    """Generate SHAP beeswarm plot."""
    np.random.seed(seed)
    plt.close("all")
    
    figure, axis = plt.subplots(figsize=(5, 4))
    plt.sca(axis)
    
    beeswarm(explanation, show=False)
    
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def save_waterfall_plot(local_explanation, output_path):
    """Generate SHAP waterfall plot."""
    plt.close("all")
    
    figure, axis = plt.subplots(figsize=(7, 5.25))
    waterfall(local_explanation, ax=axis, show=False)
    
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    with Image.open(output_path) as img:
        img.crop((50, 40, img.width, img.height - 45)).save(output_path)


# Inject compatibility structures to allow the deserialization of legacy scikit-learn models.
class IdentityLink:
    def link(self, x): return x
    def inverse(self, x): return x

class LeastSquaresError:
    K = 1
    link = IdentityLink()

    def __init__(self, *args, **kwargs): pass
    def __call__(self, y, pred, sample_weight=None): 
        return np.mean((y - pred) ** 2)
    def negative_gradient(self, y, pred, **k): 
        return y - pred
    def update_terminal_regions(self, *args, **kwargs): pass
    def get_init_raw_predictions(self, X, estimator): 
        return estimator.predict(X).reshape(-1, 1)

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

# --- Custom Unpickler ---
class LegacySklearnUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.ensemble._gb_losses":
            if name == "LeastSquaresError":
                return LeastSquaresError
            if name == "IdentityLink":
                return IdentityLink
        return super().find_class(module, name)

def load_legacy_model(file_path):
    with open(file_path, "rb") as f:
        data = LegacySklearnUnpickler(f).load()
    # Applica la patch a prescindere dal tipo di dato restituito
    d = {"model": patch_monotonic_cst(data["model"]), "features": data["features"], "scaler": data["scaler"]}
    return d

'''
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
'''