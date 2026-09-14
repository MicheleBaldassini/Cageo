"""Provide ordering, formatting, and clustering helpers for SHAP plots."""

import sys
import re
import numpy as np
import matplotlib.pyplot as plt

import colors
import shap

from collections.abc import Mapping


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = [""]

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError(
                "class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'"
            )

        # Splits on last occurrence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def format_value(s, format_str):
    """Strips trailing zeros and uses a unicode minus sign."""

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r"\.?0+$", "", s)
    if s[0] == "-":
        s = "\u2212" + s[1:]
    return s


# From: https://groups.google.com/forum/m/#!topic/openrefine/G7_PSdUeno0
def ordinal_str(n):
    """Converts a number to and ordinal string."""
    return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10, "th"
    )


def convert_color(color):
    """Convert a named SHAP color or return the supplied color value."""
    if color == "shap_red":
        return colors.red_rgb
    if color == "shap_blue":
        return colors.blue_rgb

    try:
        return plt.get_cmap(color)
    except ValueError:
        return color


# def convert_ordering(ordering, shap_values):
#     if issubclass(type(ordering), shap.utils.OpChain):
#         ordering = ordering.apply(shap.Explanation(shap_values))
#     if issubclass(type(ordering), shap.Explanation):
#         if "argsort" in [op["name"] for op in ordering.op_history]:
#             ordering = ordering.values
#         else:
#             ordering = ordering.argsort.flip.values
#     return ordering
def _op_name(op):
    """Read an operation name from a mapping or operation object."""
    # Support mappings and objects such as OpHistoryItem.
    if isinstance(op, Mapping):
        return op.get("name", None)
    return getattr(op, "name", None)


def _op_kwargs(op):
    """Read operation keyword arguments from a mapping or object."""
    if isinstance(op, Mapping):
        return op.get("kwargs", {})
    return getattr(op, "kwargs", {})


def _op_get(op, key, default=None):
    """Safely access a key or attribute of an operation-history item."""
    if isinstance(op, Mapping):
        return op.get(key, default)
    return getattr(op, key, default)


def convert_ordering(ordering, shap_values):
    """Resolve an ordering specification into feature indices."""
    if issubclass(type(ordering), shap.utils.OpChain):
        ordering = ordering.apply(shap.Explanation(shap_values))

    if issubclass(type(ordering), shap.Explanation):
        # Operation history may contain legacy mappings or newer objects.
        op_hist = getattr(ordering, "op_history", []) or []
        names = [_op_name(op) for op in op_hist]
        if "argsort" in [n for n in names if n]:
            # If argsort was applied in the history, values is already ordered.
            ordering = ordering.values
        else:
            # Standard path: Explanation.argsort -> .flip -> .values.
            # In some versions argsort and flip are OpChain objects.
            chain = getattr(ordering, "argsort", None)
            if chain is None:
                # Fall back to sorting by the mean absolute value.
                vals = np.atleast_2d(ordering.values)
                ordering = np.argsort(np.mean(np.abs(vals), axis=0))[::-1]
                return ordering

            # Use flip when available; otherwise retain the current chain.
            chain = getattr(chain, "flip", chain)
            values = getattr(chain, "values", None)
            if values is not None:
                ordering = values
            else:
                # Some versions require apply(...).
                ordering = chain.apply(shap.Explanation(shap_values)).values
    return ordering


def get_sort_order(dist, clust_order, cluster_threshold, feature_order):
    """Returns a sorted order of the values where we respect the clustering order when dist[i,j] < cluster_threshold"""

    # feature_imp = np.abs(values)

    # if partition_tree is not None:
    #     new_tree = fill_internal_max_values(partition_tree, shap_values)
    #     clust_order = sort_inds(new_tree, np.abs(shap_values))
    clust_inds = np.argsort(clust_order)

    feature_order = feature_order.copy()  # order.apply(shap.Explanation(shap_values))
    # print("feature_order", feature_order)
    for i in range(len(feature_order) - 1):
        ind1 = feature_order[i]
        next_ind = feature_order[i + 1]
        next_ind_pos = i + 1
        for j in range(i + 1, len(feature_order)):
            ind2 = feature_order[j]

            # if feature_imp[ind] >
            # if ind1 == 2:
            #     print(ind1, ind2, dist[ind1,ind2])
            if dist[ind1, ind2] <= cluster_threshold:
                # if ind1 == 2:
                #     print(clust_inds)
                #     print(ind1, ind2, next_ind, dist[ind1,ind2], clust_inds[ind2], clust_inds[next_ind])
                if (
                    dist[ind1, next_ind] > cluster_threshold
                    or clust_inds[ind2] < clust_inds[next_ind]
                ):
                    next_ind = ind2
                    next_ind_pos = j
            # print("next_ind", next_ind)
            # print("next_ind_pos", next_ind_pos)

        # insert the next_ind next
        for j in range(next_ind_pos, i + 1, -1):
            # print("j", j)
            feature_order[j] = feature_order[j - 1]
        feature_order[i + 1] = next_ind
        # print(feature_order)

    return feature_order


def merge_nodes(values, partition_tree):
    """This merges the two clustered leaf nodes with the smallest total value."""
    M = partition_tree.shape[0] + 1

    ptind = 0
    min_val = np.inf
    for i in range(partition_tree.shape[0]):
        ind1 = int(partition_tree[i, 0])
        ind2 = int(partition_tree[i, 1])
        if ind1 < M and ind2 < M:
            val = np.abs(values[ind1]) + np.abs(values[ind2])
            if val < min_val:
                min_val = val
                ptind = i
                # print("ptind", ptind, min_val)

    ind1 = int(partition_tree[ptind, 0])
    ind2 = int(partition_tree[ptind, 1])
    if ind1 > ind2:
        tmp = ind1
        ind1 = ind2
        ind2 = tmp

    partition_tree_new = partition_tree.copy()
    for i in range(partition_tree_new.shape[0]):
        i0 = int(partition_tree_new[i, 0])
        i1 = int(partition_tree_new[i, 1])
        if i0 == ind2:
            partition_tree_new[i, 0] = ind1
        elif i0 > ind2:
            partition_tree_new[i, 0] -= 1
            if i0 == ptind + M:
                partition_tree_new[i, 0] = ind1
            elif i0 > ptind + M:
                partition_tree_new[i, 0] -= 1

        if i1 == ind2:
            partition_tree_new[i, 1] = ind1
        elif i1 > ind2:
            partition_tree_new[i, 1] -= 1
            if i1 == ptind + M:
                partition_tree_new[i, 1] = ind1
            elif i1 > ptind + M:
                partition_tree_new[i, 1] -= 1
    partition_tree_new = np.delete(partition_tree_new, ptind, axis=0)

    # update the counts to be correct
    fill_counts(partition_tree_new)

    return partition_tree_new, ind1, ind2


def dendrogram_coords(leaf_positions, partition_tree):
    """Returns the x and y coords of the lines of a dendrogram where the leaf order is given.

    Note that scipy can compute these coords as well, but it does not allow you to easily specify
    a specific leaf order, hence this reimplementation.
    """

    xout = []
    yout = []
    _dendrogram_coords_rec(
        partition_tree.shape[0] - 1, leaf_positions, partition_tree, xout, yout
    )

    return np.array(xout), np.array(yout)


def _dendrogram_coords_rec(pos, leaf_positions, partition_tree, xout, yout):
    """Recursively construct dendrogram line coordinates."""
    M = partition_tree.shape[0] + 1

    if pos < 0:
        return leaf_positions[pos + M], 0

    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M

    x_left, y_left = _dendrogram_coords_rec(
        left, leaf_positions, partition_tree, xout, yout
    )
    x_right, y_right = _dendrogram_coords_rec(
        right, leaf_positions, partition_tree, xout, yout
    )

    y_curr = partition_tree[pos, 2]

    xout.append([x_left, x_left, x_right, x_right])
    yout.append([y_left, y_curr, y_curr, y_right])

    return (x_left + x_right) / 2, y_curr


def fill_internal_max_values(partition_tree, leaf_values):
    """This fills the forth column of the partition tree matrix with the max leaf value in that cluster."""
    M = partition_tree.shape[0] + 1
    new_tree = partition_tree.copy()
    for i in range(new_tree.shape[0]):
        val = 0
        if new_tree[i, 0] < M:
            ind = int(new_tree[i, 0])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i, 0]) - M
            val = max(val, np.abs(new_tree[ind, 3]))  # / partition_tree[ind,2])
        if new_tree[i, 1] < M:
            ind = int(new_tree[i, 1])
            val = max(val, np.abs(leaf_values[ind]))
        else:
            ind = int(new_tree[i, 1]) - M
            val = max(val, np.abs(new_tree[ind, 3]))  # / partition_tree[ind,2])
        new_tree[i, 3] = val
    return new_tree


def fill_counts(partition_tree):
    """This updates the"""
    M = partition_tree.shape[0] + 1
    for i in range(partition_tree.shape[0]):
        val = 0
        if partition_tree[i, 0] < M:
            ind = int(partition_tree[i, 0])
            val += 1
        else:
            ind = int(partition_tree[i, 0]) - M
            val += partition_tree[ind, 3]
        if partition_tree[i, 1] < M:
            ind = int(partition_tree[i, 1])
            val += 1
        else:
            ind = int(partition_tree[i, 1]) - M
            val += partition_tree[ind, 3]
        partition_tree[i, 3] = val


def sort_inds(partition_tree, leaf_values, pos=None, inds=None):
    """Sort leaf indices according to a hierarchical partition tree."""
    if inds is None:
        inds = []

    if pos is None:
        partition_tree = fill_internal_max_values(partition_tree, leaf_values)
        pos = partition_tree.shape[0] - 1

    M = partition_tree.shape[0] + 1

    if pos < 0:
        inds.append(pos + M)
        return

    left = int(partition_tree[pos, 0]) - M
    right = int(partition_tree[pos, 1]) - M

    left_val = partition_tree[left, 3] if left >= 0 else leaf_values[left + M]
    right_val = partition_tree[right, 3] if right >= 0 else leaf_values[right + M]

    if left_val < right_val:
        tmp = right
        right = left
        left = tmp

    sort_inds(partition_tree, leaf_values, left, inds)
    sort_inds(partition_tree, leaf_values, right, inds)

    return inds
