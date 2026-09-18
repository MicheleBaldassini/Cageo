"""Render customized SHAP beeswarm plots."""

import os
import sys

sys.path.append(os.path.realpath(__file__ + "/../assets"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import scipy.cluster
import scipy.sparse
import scipy.spatial
from scipy.stats import gaussian_kde

import colors
from utils import (
    safe_isinstance,
    format_value,
    ordinal_str,
    convert_color,
    convert_ordering,
    dendrogram_coords,
    get_sort_order,
    merge_nodes,
    sort_inds,
)
import shap


labelsize = 28
fontsize = 28


# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def beeswarm(
    shap_values,
    max_display=20,
    order=shap.Explanation.abs.mean(0),
    clustering=None,
    cluster_threshold=0.5,
    color=None,
    axis_color="#333333",
    alpha=1,
    show=True,
    log_scale=False,
    color_bar=True,
    plot_size="auto",
    color_bar_label="Feature value",
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    show : bool
        Whether ``matplotlib.pyplot.show()`` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via plt.gca().

    color_bar : bool
        Whether to draw the color bar (legend).

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged.

    Examples
    --------

    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """

    if not isinstance(shap_values, shap.Explanation):
        emsg = (
            "The beeswarm plot requires an `Explanation` object as the "
            "`shap_values` argument."
        )
        raise TypeError(emsg)

    sv_shape = shap_values.shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
        raise ValueError(emsg)

    shap_exp = shap_values
    # we make a copy here, because later there are places that might modify this array
    values = np.copy(shap_exp.values)
    features = shap_exp.data
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names = shap_exp.feature_names

    order = convert_ordering(order, values)

    # default color:
    if color is None:
        if features is not None:
            color = colors.red_blue
        else:
            color = colors.blue_rgb
    color = convert_color(color)

    idx2cat = None
    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape "
            "of the provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            print(shape_msg)
            pass
        if num_features != features.shape[1]:
            print(shape_msg)
            pass

    if feature_names is None:
        feature_names = np.array([str(i) for i in range(num_features)])

    if log_scale:
        plt.xscale("symlog")

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering

    if partition_tree is not None:
        if partition_tree.shape[1] != 4:
            emsg = (
                "The clustering provided by the Explanation object does not seem to "
                "be a partition tree (which is all shap.plots.bar supports)!"
            )
            raise ValueError(emsg)

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    # iteratively merge nodes until we can cut off the smallest feature values to stay within
    # num_features without breaking a cluster tree
    orig_inds = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()
    while True:
        feature_order = convert_ordering(order, shap.Explanation(np.abs(values)))
        if partition_tree is not None:
            # compute the leaf order if we were to show (and so have the ordering respect) the whole partition tree
            clust_order = sort_inds(partition_tree, np.abs(values))

            # now relax the requirement to match the partition tree ordering for connections above cluster_threshold
            dist = scipy.spatial.distance.squareform(
                scipy.cluster.hierarchy.cophenet(partition_tree)
            )
            feature_order = get_sort_order(
                dist, clust_order, cluster_threshold, feature_order
            )

            # if the last feature we can display is connected in a tree the next feature then we can't just cut
            # off the feature ordering, so we need to merge some tree nodes and then try again.
            if (
                max_display < len(feature_order)
                and dist[feature_order[max_display - 1], feature_order[max_display - 2]]
                <= cluster_threshold
            ):
                # values, partition_tree, orig_inds = merge_nodes(values, partition_tree, orig_inds)
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for i in range(len(values)):
                    values[:, ind1] += values[:, ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    feature_names_new = []
    for pos, inds in enumerate(orig_inds):
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(
                feature_names[inds[max_ind]] + " + %d other features" % (len(inds) - 1)
            )
    feature_names = feature_names_new

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features < len(values[0]):
        num_cut = np.sum(
            [
                len(orig_inds[feature_order[i]])
                for i in range(num_features - 1, len(values[0]))
            ]
        )
        values[:, feature_order[num_features - 1]] = np.sum(
            [
                values[:, feature_order[i]]
                for i in range(num_features - 1, len(values[0]))
            ],
            0,
        )

    # build our y-tick labels
    yticklabels = [feature_names[i] for i in feature_inds]
    if num_features < len(values[0]):
        yticklabels[-1] = "Sum of %d other features" % num_cut

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(
            8, min(len(feature_order), max_display) * row_height + 1.5
        )
    elif type(plot_size) in (list, tuple):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(
            8, min(len(feature_order), max_display) * plot_size + 1.5
        )
    plt.axvline(x=0, color="#999999", zorder=-1)

    # make the beeswarm dots
    for pos, i in enumerate(reversed(feature_inds)):
        plt.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]
        inds = np.arange(len(shaps))
        np.random.shuffle(inds)
        if fvalues is not None:
            fvalues = fvalues[inds]
        shaps = shaps[inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:  # check categorical feature
                colored_feature = False
            else:
                fvalues = np.array(
                    fvalues, dtype=np.float64
                )  # make sure this can be numeric
        except Exception:
            colored_feature = False
        N = len(shaps)
        # hspacing = (np.max(shaps) - np.min(shaps)) / 200
        # curr_bin = []
        nbins = 100
        quant = np.round(
            nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
        )
        inds = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)
        for ind in inds:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        if (
            safe_isinstance(color, "matplotlib.colors.Colormap")
            and features is not None
            and colored_feature
        ):
            # trim the color range, but prevent the color range from collapsing
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:  # fixes rare numerical precision issues
                vmin = vmax

            if features.shape[0] != len(shaps):
                emsg = "Feature and SHAP matrices must have the same number of rows!"
                print(emsg)
                pass

            # plot the nan fvalues in the interaction feature as grey
            nan_mask = np.isnan(fvalues)
            plt.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                s=16,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

            # plot the non-nan fvalues colored by the trimmed feature value
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            plt.scatter(
                shaps[np.invert(nan_mask)],
                pos + ys[np.invert(nan_mask)],
                cmap=color,
                vmin=vmin,
                vmax=vmax,
                s=16,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )
        else:
            plt.scatter(
                shaps,
                pos + ys,
                s=16,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color=color if colored_feature else "#777777",
                rasterized=len(shaps) > 500,
            )

    # draw the color bar
    if (
        safe_isinstance(color, "matplotlib.colors.Colormap")
        and color_bar
        and features is not None
    ):
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = plt.colorbar(m, ax=plt.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels(["Low", "High"])
        cb.set_label(color_bar_label, size=labelsize, labelpad=0)
        cb.ax.tick_params(labelsize=labelsize, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)
        # bbox = cb.ax.get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        # cb.ax.set_aspect((bbox.height - 0.9) * 20)
        # cb.draw_all()

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(range(len(feature_inds)), reversed(yticklabels), fontsize=fontsize)
    plt.gca().tick_params("y", length=20, width=0.5, which="major")
    plt.gca().tick_params("x", labelsize=labelsize)
    plt.ylim(-1, len(feature_inds))
    plt.xlabel("SHAP value (impact on model output)", fontsize=fontsize)
    if show:
        plt.show()
    else:
        return plt.gca()
