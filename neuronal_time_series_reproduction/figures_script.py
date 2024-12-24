import os
from typing import Tuple, List

import neurotorch as nt
import numpy as np
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from neurotorch.visualisation.time_series_visualisation import (
    Visualise,
    VisualiseKMeans,
    VisualisePCA,
    VisualiseUMAP,
)
from neurotorch.visualisation.report import mix_report

from constants import MPL_RC_BIG_FONT_PARAMS, MPL_RC_SMALL_FONT_PARAMS
from train_script import pvar_mean_std


@torch.no_grad()
def complete_report(
        model: nt.SequentialRNN,
        y_pred, y_target, spikes=None,
        **kwargs
) -> Tuple[nt.Visualise, nt.Visualise]:
    """
    Plots the report for the given model and data.

    :param model: The given model
    :type model: nt.SequentialRNN
    :param y_pred: The predicted time series
    :type y_pred: torch.Tensor
    :param y_target: The target time series
    :type y_target: torch.Tensor
    :param spikes: The spikes of the last layer
    :type spikes: torch.Tensor
    :param kwargs: Keyword arguments

    :return: The predicted and target visualisations
    :rtype: Tuple[nt.Visualise, nt.Visualise]
    """
    import matplotlib
    matplotlib.rcParams.update(MPL_RC_SMALL_FONT_PARAMS)

    y_pred = nt.to_numpy(y_pred).squeeze()
    y_target = nt.to_numpy(y_target).squeeze()
    shape = kwargs.get(
        "shape",
        nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time steps [-]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Activity [-]"),
            ]
        )
    )

    # Visualises
    viz_target = Visualise(y_target, shape=shape)
    viz_pred = Visualise(y_pred, shape=shape)
    viz_target_kmeans = VisualiseKMeans(y_target, shape=shape)
    viz_pred_kmeans = Visualise(viz_target_kmeans.permute_timeseries(viz_pred.timeseries), shape=shape)

    # Report
    viz_pred.plot_timeseries_comparison_report(
        viz_target.timeseries,
        title=f"Prediction",
        filename=f"{model.checkpoint_folder}/figures/timeseries_comparison_report.png",
        show=kwargs.get("show", False), dpi=600,
    )

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    _, ax = viz_target_kmeans.heatmap(fig=fig, ax=axes[0], title="True time series")
    ax.images[-1].colorbar.remove()  # Workaround to remove the colorbar. TODO: add a parameter to remove the colorbar
    viz_pred_kmeans.heatmap(
        fig=fig, ax=axes[1],
        title="Predicted time series",
        filename=f"{model.checkpoint_folder}/figures/heatmap.png",
        show=kwargs.get("show", False), dpi=600,
    )

    # PCA & UMAP
    nt.visualisation.report.UMAP_PCA_report(
        pred_viz=viz_pred,
        target_viz=viz_target,
        filename=f"{model.checkpoint_folder}/figures/UMAP_PCA_report.png",
        show=kwargs.get("show", False), dpi=600,
    )
    mix_report(
        viz_pred, viz_target,
        show=kwargs.get("show", False), dpi=600,
        filename=f"{model.checkpoint_folder}/figures/Mix_UMAP_report.png",
    )

    # Animation
    if kwargs.get("animate", False):
        viz_pred.animate(
            time_interval=1.0,
            weights=nt.to_numpy(model.get_layer("Encoder").forward_weights),
            dt=0.1,
            show=kwargs.get("show", False),
            filename="data/figures/wc_eprop/animation.gif",
            writer=None,
        )
    return viz_pred, viz_target


@torch.no_grad()
def plot_model_comparison(
        model0: nt.SequentialRNN,
        model1: nt.SequentialRNN,
        visualise0: nt.Visualise,
        visualise1: nt.Visualise,
        visualise_target: nt.Visualise,
        **kwargs
):
    plt.close("all")
    mpl.rcParams.update(MPL_RC_BIG_FONT_PARAMS)
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=kwargs.get("figsize", (20, 10)))

    visualise0.plot_timeseries_comparison(
        target=visualise_target.timeseries,
        traces_to_show=["typical_0"],
        traces_to_show_names=[rf"Typical neuronal activity"],
        title="Typical neuron activity",
        desc="Prediction",
        show=False,
        numbered=False,
        fig=fig,
        axes=np.asarray([axes[0, 1]]),
        legend_loc="lower left"
    )
    viz_target_umap = VisualiseUMAP(
        timeseries=visualise_target.timeseries, shape=visualise_target.shape, random_state=kwargs.get("seed", 42)
    )
    VisualiseUMAP(
        timeseries=visualise0.timeseries,
        shape=visualise0.shape,
        umap_transform=viz_target_umap.umap_transform,
    ).trajectory_umap(
        target=visualise_target,
        UMAPS=(1, 2),
        traces="UMAP_space",
        fig=fig, axes=np.asarray([axes[0, 0]]),
        show=False,
    )
    visualise1.plot_timeseries_comparison(
        target=visualise_target.timeseries,
        traces_to_show=["typical_0"],
        traces_to_show_names=[f"Typical neuronal activity"],
        title="Typical neuron activity",
        desc="Prediction",
        show=False,
        numbered=False,
        fig=fig,
        axes=np.asarray([axes[1, 1]]),
        legend_loc="lower left"
    )
    VisualiseUMAP(
        timeseries=visualise1.timeseries,
        shape=visualise1.shape,
        umap_transform=viz_target_umap.umap_transform,
    ).trajectory_umap(
        target=visualise_target,
        UMAPS=(1, 2),
        traces="UMAP_space",
        fig=fig, axes=np.asarray([axes[1, 0]]),
        show=False,
    )
    fig, axes = format_comparison_figure(
        fig=fig, axes=axes,
        model0=model0, model1=model1,
        visualise0=visualise0, visualise1=visualise1,
        visualise_target=visualise_target,
        model_name_to_repr={"SNN-LPF": "SpyLIF-LPF"},
        **kwargs
    )

    save_folder = kwargs.get("save_folder", "figures")
    os.makedirs(f"{save_folder}", exist_ok=True)
    ext_list = ["pdf", "png"]
    filename = f"{model0.name}_vs_{model1.name}"
    for ext in ext_list:
        fig.savefig(f"{save_folder}/{filename}.{ext}", bbox_inches='tight', pad_inches=0.1, dpi=900)
    if kwargs.get("show", False):
        plt.show()

    _, __, heat_files = plot_heatmap_comparison(
        model0=model0, model1=model1,
        visualise0=visualise0, visualise1=visualise1,
        visualise_target=visualise_target,
        **kwargs
    )
    return [f"{save_folder}/{filename}.{ext}" for ext in ext_list] + heat_files


@torch.no_grad()
def format_comparison_figure(
        fig: plt.Figure,
        axes: np.ndarray,
        model0: nt.SequentialRNN,
        model1: nt.SequentialRNN,
        visualise0: nt.Visualise,
        visualise1: nt.Visualise,
        visualise_target: nt.Visualise,
        **kwargs
):
    # Add titles
    mean_repr = r"$\mathbb{E}$"
    for i, (model_i, viz_i, text_y_loc) in enumerate(zip([model0, model1], [visualise0, visualise1], [1.0, 0.48])):
        pvar = nt.losses.PVarianceLoss()(viz_i.timeseries, visualise_target.timeseries)
        mean, std = pvar_mean_std(viz_i.timeseries, visualise_target.timeseries)
        pvar, mean, std = float(nt.to_numpy(pvar)), float(nt.to_numpy(mean)), float(nt.to_numpy(std))
        # fig.text(
        #     0.42, text_y_loc,
        #     rf'{model_i.name} Prediction (pVar: {pvar:.3f}, {mean_repr}[pVar]: {mean:.3f} $\pm$ {std:.3f})',
        #     ha='center', va='center',
        #     fontsize=kwargs.get("fontsize", mpl.rcParams["font.size"])
        # )
        axes[i, 0].text(
            0, 0,
            rf'pVar: {pvar:.2f}',
            ha='left', va='bottom',
            fontsize=kwargs.get("fontsize", mpl.rcParams["font.size"]),
            transform=axes[i, 0].transAxes
        )

    # Set line styles and colours
    for ax in np.ravel(axes):
        for line in ax.get_lines():
            if "real" in line.get_label().lower():
                line.set_linestyle("--")
                line.set_color("tab:orange")
                line.set_zorder(1)
                line.set_linewidth(line.get_linewidth() * 1.1)
            elif "pred" in line.get_label().lower():
                line.set_linestyle("-")
                line.set_color("tab:blue")
                line.set_zorder(0)

    # Format x ticks
    if kwargs.get("timesteps", None) is not None:
        timesteps = kwargs["timesteps"]
        for ax in [axes[0, 1], axes[1, 1]]:
            old_ticks = ax.get_xticks()
            new_ticks = timesteps[np.linspace(0, len(timesteps) - 1, len(old_ticks), dtype=int)]
            ax.set_xticks(new_ticks)
            ax.set_xticklabels([f"{t:.0f}" for t in new_ticks])
            ax.set_xlim(new_ticks[0], new_ticks[-1])
            ax.set_xlabel("Time [s]")

    # Format UMAP plots
    for umap_ax in [axes[0, 0], axes[1, 0]]:
        umap_ax.set_box_aspect(1)
        umap_ax.set_xticklabels([])
        umap_ax.set_yticklabels([])
        points_lines = [line for line in umap_ax.get_lines() if len(line.get_xdata()) == 1]
        x_points = [line.get_xdata() for line in points_lines]
        annotations = [child for child in umap_ax.get_children() if isinstance(child, mpl.text.Annotation)]
        annot_sorted_by_text = list(sorted(annotations, key=lambda x: x.get_text()))
        # the x_point associated with each annotation is the nearest x_point of the annotation x position
        text_to_point = {
            annot.get_text(): points_lines[np.argmin(np.abs(np.asarray(x_points) - annot.get_position()[0]))]
            for annot in annot_sorted_by_text
        }
        points_sorted_by_text = [text_to_point[text] for text in sorted(text_to_point.keys())]
        points_sorted_by_text[0].set_marker("^")
        points_sorted_by_text[0].set_markersize(10)
        points_sorted_by_text[0].set_color("black")
        points_sorted_by_text[0].set_label(r"$t_0$")
        for pt in points_sorted_by_text[1:]:
            pt.remove()
        for annotation in annotations:
            annotation.remove()

    # Format the legend
    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    for i, label in enumerate(legend_labels):
        if "real" in label.lower():
            legend_labels[i] = "Real"
        elif "pred" in label.lower():
            legend_labels[i] = "Prediction"

    for ax in [a for a in np.ravel(axes) if a.get_legend() is not None]:
        ax.get_legend().remove()
    # fig.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(0.95, 1.0))
    axes[0, -1].legend(
        legend_handles, legend_labels,
        # loc="upper left",
    )

    # Add titles to the axes
    for ax in np.ravel(axes):
        ax.set_title("")
    axes[0, -1].set_title("Typical prediction of single neuron time series")
    axes[0, 0].set_title("Whole network activity")
    model_name_to_repr = kwargs.get("model_name_to_repr", {})
    for i, model in enumerate([model0, model1]):
        model_repr = model_name_to_repr.get(model.name, model.name)
        axes[i, 0].set_ylabel(f"{model_repr}\n{axes[i, 0].get_ylabel()}")

    # Add titles to the axes
    nt.Visualise.number_axes(axes, method="text", fontsize=mpl.rcParams['font.size'], x=0, y=1)
    # fig.subplots_adjust(hspace=0.35, wspace=0.35)
    fig.tight_layout()
    return fig, axes


@torch.no_grad()
def plot_heatmap_comparison(
        model0: nt.SequentialRNN,
        model1: nt.SequentialRNN,
        visualise0: nt.Visualise,
        visualise1: nt.Visualise,
        visualise_target: nt.Visualise,
        **kwargs
) -> Tuple[plt.Figure, np.ndarray, List[str]]:
    # Visualises
    viz_target_kmeans = VisualiseKMeans(visualise_target.timeseries, shape=visualise_target.shape)
    viz0_kmeans = Visualise(viz_target_kmeans.permute_timeseries(visualise0.timeseries), shape=visualise0.shape)
    viz1_kmeans = Visualise(viz_target_kmeans.permute_timeseries(visualise1.timeseries), shape=visualise1.shape)

    # Heatmap
    fig, axes = plt.subplots(1, 3, figsize=kwargs.get("figsize", (20, 10)), sharey="all")
    # nt.Visualise.number_axes(axes, method="title", fontsize=kwargs.get("fontsize", mpl.rcParams["font.size"]))
    _, ax = viz_target_kmeans.heatmap(fig=fig, ax=axes[0], title="Neuronal measurements")
    viz0_kmeans.heatmap(fig=fig, ax=axes[1], title=f"{model0.name} prediction")
    viz1_kmeans.heatmap(fig=fig, ax=axes[2], title=f"{model1.name} prediction")
    axes[0].images[-1].colorbar.remove()  # Workaround to remove the colorbar.
    axes[1].images[-1].colorbar.remove()  # Workaround to remove the colorbar.
    axes[2].images[-1].colorbar.remove()  # Workaround to remove the colorbar.
    # axes[-1].images[-1].colorbar.set_label("Activity [-]", rotation=90, labelpad=20)
    # add the colobar to another axis
    # divider = make_axes_locatable(axes[-1])
    # cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    cbar_ax = fig.add_axes([0.98, 0.1, 0.02, 0.83])
    fig.colorbar(axes[0].images[-1], cax=cbar_ax, label="Activity [-]", orientation="vertical")

    for ax in np.ravel(axes):
        ax.set_xlabel("")
        ax.set_ylabel("")
    axes[0].set_ylabel("Neuron ID [-]")
    axes[1].set_xlabel("Time steps [-]")

    # Format x ticks
    if kwargs.get("timesteps", None) is not None:
        timesteps = kwargs["timesteps"]
        for ax in np.ravel(axes):
            old_ticks = ax.get_xticks()
            new_ticks = timesteps[np.linspace(0, len(timesteps) - 1, len(old_ticks), dtype=int)]
            ax.set_xticks(new_ticks)
            ax.set_xticklabels([f"{t:.0f}" for t in new_ticks])
            ax.set_xlim(new_ticks[0], new_ticks[-1])
        axes[1].set_xlabel("Time [s]")

    save_folder = kwargs.get("save_folder", "figures")
    fig.tight_layout()
    os.makedirs(f"{save_folder}", exist_ok=True)
    ext_list = ["pdf", "png", "svg"]
    filename = f"heatmap_{model0.name}-{model1.name}"
    for ext in ext_list:
        fig.savefig(f"{save_folder}/{filename}.{ext}", bbox_inches='tight', pad_inches=0.1, dpi=900)
    if kwargs.get("show", False):
        plt.show()
    return fig, axes, [f"{save_folder}/{filename}.{ext}" for ext in ext_list]


def plot_simple_report(
        model: nt.SequentialRNN,
        y_pred, y_target, spikes=None,
        **kwargs
) -> Tuple[nt.Visualise, nt.Visualise]:
    """
    Plots a simple version of the report for the given model and data.

    :param model:
    :param y_pred:
    :param y_target:
    :param spikes:
    :param kwargs:
    :return:
    """
    import matplotlib
    matplotlib.rcParams.update(MPL_RC_SMALL_FONT_PARAMS)

    fig, axes = kwargs.get("fig", None), kwargs.get("axes", None)
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            ncols=2, nrows=3, figsize=kwargs.get("figsize", (16, 8)),
            sharex="all", sharey="all",
        )
    else:
        axes = np.asarray(axes)

    y_pred = nt.to_numpy(y_pred).squeeze()
    y_target = nt.to_numpy(y_target).squeeze()
    shape = kwargs.get(
        "shape",
        nt.Size(
            [
                nt.Dimension(None, nt.DimensionProperty.TIME, "Time steps [-]"),
                nt.Dimension(None, nt.DimensionProperty.NONE, "Activity [-]"),
            ]
        )
    )

    # Visualises
    viz_target = Visualise(y_target, shape=shape)
    viz_pred = Visualise(y_pred, shape=shape)

    viz_pred.plot_timeseries_comparison(
        viz_target.timeseries, spikes,
        title="",
        fig=fig, axes=axes[:, 0],
        traces_to_show=["best", "most_var", "worst"],
        traces_to_show_names=[
            "Neuron (1)", "Neuron (2)", "Neuron (3)"
        ],
        show=False,
    )

    viz_pred.plot_timeseries_comparison(
        viz_target.timeseries, spikes,
        title="",
        fig=fig, axes=axes[:, 1],
        traces_to_show=[f"typical_{i}" for i in range(3)],
        traces_to_show_names=[
            "Neuron (4)", "Neuron (5)", "Neuron (6)"
        ],
        show=False,
    )
    xlabel = kwargs.get("xlabel", "Time steps [-]")
    ylabel = kwargs.get("ylabel", "Activity [-]")
    for ax in np.ravel(axes):
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)

    fig.suptitle("Neuronal activity prediction")

    # viz_pred.number_axes(np.ravel(axes), **kwargs)
    fig.tight_layout()

    filename = kwargs.get("filename", f"{model.checkpoint_folder}/figures/timeseries_comparison_simple_report.pdf")
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=kwargs.get("dpi", 300))
    if kwargs.get("show", False):
        plt.show()
    if kwargs.get("close", False):
        plt.close(fig)

    return viz_pred, viz_target

