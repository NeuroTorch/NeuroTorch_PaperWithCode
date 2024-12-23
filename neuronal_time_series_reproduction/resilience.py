import os
import pathlib
import shutil
from collections import namedtuple, defaultdict
from copy import deepcopy
from typing import List, Dict, Any, Union, Optional, Tuple

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import neurotorch as nt
import numpy as np
import pandas as pd
import pythonbasictools as pbt
from scipy import stats

from _filter_models import _filter_models_
from models import ModelType
from constants import MPL_RC_DEFAULT_PARAMS
from train_script import test_model
from utils import (
    get_cmd_kwargs,
    gather_models_from_folders,
    sparsify_matrix,
    _copy_figures,
    _save_figures,
)


def ablation_on_layer_(model: nt.SequentialRNN, layer_name: str, dataloader, **kwargs):
    initial_sparsity = kwargs.get("initial_sparsity", 0.0)
    final_sparsity = kwargs.get("final_sparsity", 1.0)
    n_pts, n_test = kwargs.get("n_pts", 100), kwargs.get("n_test", 10)
    sparsity_space = kwargs.get("sparsity_space", np.linspace(initial_sparsity, final_sparsity, num=n_pts))
    n_pts = len(sparsity_space)
    ablation_strategy = kwargs.get("ablation_strategy", kwargs.get("strategy", "smallest"))
    kwargs["strategy"] = ablation_strategy
    if layer_name is None:
        layer_name = model.get_all_layers()[0].name
    elif isinstance(layer_name, int):
        layer_name = model.get_all_layers()[layer_name].name
    seed = kwargs.get("seed", None)
    model_ckpt_folder = os.path.basename(model.checkpoint_folder)
    save_path = os.path.join(
        kwargs.get("save_folder", "data/temp_df_ablation"),
        f"{model_ckpt_folder}_{layer_name}_{ablation_strategy}{seed}_p{n_pts}_t{n_test}.csv"
    )
    if os.path.exists(save_path) and kwargs.get("load_df", True):
        return pd.read_csv(save_path)

    initial_weights = nt.to_numpy(model.get_layer(layer_name).get_forward_weights_data())
    init_nb_zeros = np.sum(np.isclose(initial_weights, 0.0))
    df = pd.DataFrame()
    for sparsity in sparsity_space:
        sparsified_weights = sparsify_matrix(deepcopy(initial_weights), sparsity, **kwargs)
        nb_zeros = np.sum(np.isclose(sparsified_weights, 0.0))
        model.get_layer(layer_name).set_forward_weights_data(nt.to_tensor(sparsified_weights))
        real_sparsity = np.mean(np.isclose(nt.to_numpy(model.get_layer(layer_name).forward_weights), 0.0))
        y_pred, y_target, pvar, pvar_mean, pvar_std = test_model(
            model, dataloader.dataset, n_test=n_test, verbose=False, load=False,
        )
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "sparsity": sparsity,
                        "connection_removed": init_nb_zeros - nb_zeros,
                        "pvar": nt.to_numpy(pvar),
                        "pvar_mean": nt.to_numpy(pvar_mean),
                        "pvar_std": nt.to_numpy(pvar_std),
                        "real_sparsity": nt.to_numpy(real_sparsity),
                    }, index=[0]
                )
            ]
        )

    model.get_layer(layer_name).set_forward_weights_data(nt.to_tensor(initial_weights))
    df["layer"] = [layer_name for _ in range(df.shape[0])]
    df["model_name"] = [model.name for _ in range(df.shape[0])]
    df["ablation_strategy"] = [ablation_strategy for _ in range(df.shape[0])]
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    except Exception as e:
        pass
    return df


def compute_resilience_analyse_on_models_df(
        model_objects_list: List[Dict[str, Any]],
        layers: List[str] = None,
        **kwargs
) -> pd.DataFrame:
    savepath = kwargs.get("savepath", None)
    if savepath is not None and kwargs.get("load", True):
        if os.path.exists(savepath):
            return pd.read_csv(savepath)

    if not isinstance(layers, (list, tuple)):
        layers = [layers for _ in range(len(model_objects_list))]

    n_pts, n_test = kwargs.get("n_pts", 40), kwargs.get("n_test", 32)
    verbose = kwargs.get("verbose", True)

    if kwargs.get("filter_models", True):
        model_objects_list = _filter_models_(model_objects_list, layers=layers, **kwargs)

    sparsity_space = np.linspace(0.0, 1.0, n_pts)
    ablation_strategies = kwargs.get("ablation_strategies", ["smallest", "random"])
    n_seed = kwargs.get("n_seed", n_test)
    seeds = np.arange(n_seed)
    temp_df_folder = os.path.join(os.path.dirname(savepath), kwargs.get("temp_df_folder", "data/temp_df_ablation"))
    kwgs_list, args_list, out_model_objects_list = [], [], []
    for i, m_objects in enumerate(model_objects_list):
        for ablation_strategy in ablation_strategies:
            if "random" in ablation_strategy.lower():
                for seed in seeds:
                    kwgs_list.append(dict(
                        sparsity_space=sparsity_space, n_test=n_test, ablation_strategy=ablation_strategy,
                        seed=seed, save_folder=temp_df_folder,
                    ))
                    args_list.append((m_objects["model"], layers[i], m_objects["dataloader"]))
                    out_model_objects_list.append(m_objects)
            else:
                kwgs_list.append(dict(
                    sparsity_space=sparsity_space, n_test=n_test, ablation_strategy=ablation_strategy,
                    seed=seeds[0], save_folder=temp_df_folder,
                ))
                args_list.append((m_objects["model"], layers[i], m_objects["dataloader"]))
                out_model_objects_list.append(m_objects)
    new_df_list = pbt.multiprocessing.apply_func_multiprocess(
        ablation_on_layer_,
        iterable_of_args=args_list,
        iterable_of_kwargs=kwgs_list,
        desc=f"Resilience analysis [{n_test=}, {n_pts=}, {n_seed=}]",
        verbose=verbose,
        nb_workers=kwargs.get("nb_workers", -2),
    )
    df = pd.DataFrame()
    for i, (m_objects, new_df) in enumerate(zip(out_model_objects_list, new_df_list)):
        # model_objects_list[i]["layer"] = layers[i]
        new_df["model_id"] = i
        new_df["dale"] = m_objects["params"]["force_dale_law"]
        new_df["model_type"] = m_objects["params"]["model_type"]
        df = pd.concat([df, new_df])

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        df.to_csv(savepath, index=False)
        print(f"Saved resilience analysis results to {savepath}.")

    return df


def resilience_analyse_on_models(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)
    verbose = kwargs.get("verbose", True)
    confidence_interval = kwargs.get("confidence_interval", 0.95)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)

    # Plot the results
    x_axis_key = kwargs.get("x_axis_key", "sparsity")
    x_axis_name = kwargs.get("x_axis_name", "Sparsity [-]")
    model_id_key = kwargs.get("model_id_key", "model_id")
    perf_key = kwargs.get("perf_key", "pvar")
    colors = mpl.colormaps["tab10"].colors
    dale_color = {True: colors[0], False: colors[1]}

    ablation_strategy_key = kwargs.get("ablation_strategy_key", "ablation_strategy")
    ablation_strategy = kwargs.get("ablation_strategy", None)
    if ablation_strategy is not None and ablation_strategy_key in df.columns:
        df = df[df[ablation_strategy_key] == ablation_strategy]

    model_type_linestyle = {m: ls for m, ls in zip(df["model_type"].unique(), ["-", "--", "-.", ":"])}
    model_type_hatch = {m: h for m, h in zip(df["model_type"].unique(), ['', '|', '|', '-', '+', 'x', 'o', '.', '*'])}
    fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    model_names = []
    model_type_reprs = kwargs.get("model_type_reprs", {})
    normalize_perf = kwargs.get("normalize_perf", True)
    perf_as_auc = kwargs.get("perf_as_auc", False)
    keep_only_pos_perf = kwargs.get("keep_only_pos_perf", perf_as_auc)
    group = df.groupby(["model_type", "dale"])
    for j, (model_desc, df_model) in enumerate(group):
        model_type, dale = model_desc
        model_type_repr = model_type_reprs.get(model_type, ModelType.from_name(model_type).value)
        dale_name = " + Dale" if dale else ""
        model_names.append(f"{model_type_repr}{dale_name}")

        if keep_only_pos_perf:
            df_model = df_model.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{
                    perf_key: x[perf_key].iloc[:np.argwhere(np.less_equal(np.array(x[perf_key]), 0.0))[0].item()]
                })
            )

        if perf_as_auc:
            df_model = df_model.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{perf_key: np.nancumsum(x[perf_key]) / np.nansum(x[perf_key])})
            )

        df_mean_metrics = df_model.groupby(x_axis_key)[perf_key].mean()
        df_std_metrics = df_model.groupby(x_axis_key)[perf_key].std()
        df_mean_metrics = df_mean_metrics.sort_index(ascending=True)
        df_std_metrics = df_std_metrics.sort_index(ascending=True)

        if normalize_perf:
            df_perf_ratio = df_mean_metrics.div(
                df_mean_metrics.iloc[0], axis=0
            )
            df_perf_ratio.index = df_mean_metrics.index
            df_std_metrics = df_std_metrics.div(
                df_mean_metrics.iloc[0], axis=0
            )
        else:
            df_perf_ratio = df_mean_metrics
        df_std_metrics = df_std_metrics.fillna(0)
        conf_int_a, conf_int_b = stats.norm.interval(
            confidence_interval, loc=df_perf_ratio, scale=df_std_metrics / np.sqrt(df_model.groupby(x_axis_key).size())
        )
        ax.plot(
            df_perf_ratio.index, df_perf_ratio, model_type_linestyle[model_type],
            color=dale_color[dale], label="macro pVar"
        )
        ax.fill_between(
            df_perf_ratio.index,
            conf_int_a,
            conf_int_b,
            color=dale_color[dale],
            alpha=0.2,
        )
        ax.fill_between(
            df_perf_ratio.index,
            conf_int_a,
            conf_int_b,
            color="None",
            edgecolor=dale_color[dale],
            hatch=model_type_hatch[model_type],
            alpha=0.2,
        )
        ax.set_xlabel(x_axis_name)
        ax.set_ylim(kwargs.get("ylim", [0, 1.1]))
        ax.set_xlim(kwargs.get("xlim", [0, 1.0]))
        ylabel = "Performance ratio [-]" if normalize_perf else "Performance [-]"
        ylabel = f"AUC {ylabel}" if perf_as_auc else ylabel
        ax.set_ylabel(kwargs.get("ylabel", ylabel))

    legend_elements = [
        *[
            mpl.lines.Line2D([0], [0], color=dc, linestyle="-", label={True: "Dale", False: "No Dale"}[dv])
            for dv, dc in dale_color.items()
        ],
        *[
            (
                mpl.patches.Patch(
                    facecolor="k", edgecolor="k", alpha=0.2, hatch=model_type_hatch[mt],
                    label=model_type_reprs.get(mt, ModelType.from_name(mt).value)
                ),
                mpl.lines.Line2D(
                    [0], [0], color='k', linestyle=ms, label=model_type_reprs.get(mt, ModelType.from_name(mt).value)
                ),
            )
            for mt, ms in model_type_linestyle.items()
        ]
    ]
    ax.legend(
        handles=legend_elements,
        labels=[e.get_label() if not isinstance(e, tuple) else e[0].get_label() for e in legend_elements],
        handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=1)}
    )
    fig.tight_layout()

    if kwargs.get("save", True):
        post_name = f"_norm" if normalize_perf else ""
        ext_list = ["pdf", "png", "svg"]
        savefig_filenames = [f"resilience_{x_axis_key}{post_name}.{ext}" for ext in ext_list]
        original_fig_dir = _save_figures(fig, model_objects_list[0], savefig_filenames)
        print(f"Figures saved in {original_fig_dir}")
        pbt.multiprocessing.apply_func_main_process(
            _copy_figures,
            iterable_of_args=[(m_objects, original_fig_dir, savefig_filenames) for m_objects in model_objects_list[1:]],
            desc="Saving figures",
            unit="model",
            verbose=verbose,
        )
    if kwargs.get("show", False):
        plt.show()
    return fig, ax


def plot_all_resilience_analyse_on_models(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)
    verbose = kwargs.get("verbose", True)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)

    # Plot the results
    x_axis_key = "sparsity"
    x_axis_name = "Sparsity [-]"
    model_id_key = kwargs.get("model_id_key", "model_id")
    perf_key = kwargs.get("perf_key", "pvar")
    cmap = kwargs.get("cmap", "viridis")
    if isinstance(cmap, str):
        cmap = matplotlib.cm.get_cmap(cmap)

    ablation_strategy_key = kwargs.get("ablation_strategy_key", "ablation_strategy")
    ablation_strategy = kwargs.get("ablation_strategy", None)
    if ablation_strategy is not None and ablation_strategy_key in df.columns:
        df = df[df[ablation_strategy_key] == ablation_strategy]

    model_type_linestyle = {m: ls for m, ls in zip(df["model_type"].unique(), ["-", "--", "-.", ":"])}
    model_type_hatch = {m: h for m, h in zip(df["model_type"].unique(), ['', '|', '|', '-', '+', 'x', 'o', '.', '*'])}
    fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    model_names = []
    model_type_reprs = kwargs.get("model_type_reprs", {})
    normalize_perf = kwargs.get("normalize_perf", True)
    perf_as_auc = kwargs.get("perf_as_auc", False)
    keep_only_pos_perf = kwargs.get("keep_only_pos_perf", perf_as_auc)
    group = df.groupby(["model_type", "dale"])
    for j, (model_desc, df_model) in enumerate(group):
        model_type, dale = model_desc
        model_type_repr = model_type_reprs.get(model_type, ModelType.from_name(model_type).value)
        dale_name = " + Dale" if dale else ""
        model_names.append(f"{model_type_repr}{dale_name}")

        if keep_only_pos_perf:
            df_model = df_model.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{
                    perf_key: x[perf_key].iloc[:np.argwhere(np.less_equal(np.array(x[perf_key]), 0.0))[0].item()]
                })
            )

        if perf_as_auc:
            df_model = df_model.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{perf_key: np.nancumsum(x[perf_key]) / np.nansum(x[perf_key])})
            )

        if normalize_perf:
            df_perf_ratio = df_model.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{perf_key: x[perf_key] / x[perf_key].iloc[0]})
            )
            df_perf_ratio.index = df_model.index
        else:
            df_perf_ratio = df_model

        for sub_m_id, df_sub_model in df_perf_ratio.groupby(model_id_key, group_keys=False):
            ax.plot(
                df_sub_model[x_axis_key], df_sub_model[perf_key],
                color=cmap(df_sub_model[perf_key].iloc[0]),
            )
        ax.set_xlabel(x_axis_name)
        ax.set_ylim(kwargs.get("ylim", [0, 1]))
        ax.set_xlim(kwargs.get("xlim", [0, 1]))
        ylabel = "Performance ratio [-]" if normalize_perf else "Performance [-]"
        ylabel = f"AUC {ylabel}" if perf_as_auc else ylabel
        ax.set_ylabel(kwargs.get("ylabel", ylabel))
    # fig.tight_layout()

    if kwargs.get("save", True):
        post_name = f"_norm" if normalize_perf else ""
        ext_list = ["pdf", "png", "svg"]
        savefig_filenames = [f"resilience_{x_axis_key}{post_name}.{ext}" for ext in ext_list]
        original_fig_dir = _save_figures(fig, model_objects_list[0], savefig_filenames)
        print(f"Figures saved in {original_fig_dir}")
        pbt.multiprocessing.apply_func_main_process(
            _copy_figures,
            iterable_of_args=[(m_objects, original_fig_dir, savefig_filenames) for m_objects in model_objects_list[1:]],
            desc="Saving figures",
            unit="model",
            verbose=verbose,
        )
    if kwargs.get("show", False):
        plt.show()
    return fig, ax


def resilience_analyse_on_models_auc_dist(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, np.ndarray]]:
    from sklearn.metrics import auc

    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)
    verbose = kwargs.get("verbose", True)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)
    x_axis_key = kwargs.get("x_axis_key", "sparsity")
    perf_metric_key = kwargs.get("perf_metric_key", "pvar")
    threshold = kwargs.get("threshold", 0.9)
    model_id_key = kwargs.get("model_id_key", "model_id")
    normalize_perf = kwargs.get("normalize_perf", True)

    ablation_strategy_key = kwargs.get("ablation_strategy_key", "ablation_strategy")
    ablation_strategy = kwargs.get("ablation_strategy", None)
    if ablation_strategy is not None and ablation_strategy_key in df.columns:
        df = df[df[ablation_strategy_key] == ablation_strategy]

    model_names = []
    colors = mpl.colormaps["tab10"].colors
    dale_color = {True: colors[0], False: colors[1]}
    model_type_reprs = kwargs.get("model_type_reprs", {})
    model_names_to_dist = {}
    group = df.groupby(["model_type", "dale"])
    for j, (model_desc, df_model) in enumerate(group):
        model_type, dale = model_desc
        model_type_repr = model_type_reprs.get(model_type, ModelType.from_name(model_type).value)
        dale_name = " + Dale" if dale else ""
        model_name = f"{model_type_repr}{dale_name}"
        model_names.append(model_name)

        df_model[x_axis_key] = df_model[x_axis_key].round(2)
        # sort the groups by the dist_metric_key
        df_perf_ratio = df_model.groupby(model_id_key, group_keys=False).apply(lambda x: x.sort_values(x_axis_key))

        if normalize_perf:
            # normalize the threshold_metric_key by the first value of the group
            df_perf_ratio = df_perf_ratio.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{perf_metric_key: x[perf_metric_key] / x[perf_metric_key].iloc[0]})
            )

        df_perf_ratio = df_perf_ratio.groupby(model_id_key, group_keys=False).apply(
            lambda x: x.assign(**{
                perf_metric_key: x[perf_metric_key].iloc[
                                 :np.argwhere(np.less_equal(np.array(x[perf_metric_key]), 0.0))[0].item()]
            })
        )
        dist = np.array([
            auc(
                x[x_axis_key][~np.isnan(x[perf_metric_key])],
                x[perf_metric_key][~np.isnan(x[perf_metric_key])],
            )
            for _, x in df_perf_ratio.groupby(model_id_key, group_keys=False)
        ])
        model_names_to_dist[model_name] = dist

    fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    # plot a pairwise matrix of the p-values between the distributions using t-test

    from scipy import stats

    # p_values = np.zeros((len(model_names), len(model_names)))
    # for i, model_name_i in enumerate(model_names):
    # 	for j, model_name_j in enumerate(model_names):
    # 		if i == j:
    # 			p_values[i, j] = np.NaN
    # 			continue
    # 		p_values[i, j] = stats.ttest_ind(
    # 			model_names_to_dist[model_name_i], model_names_to_dist[model_name_j],
    # 			equal_var=False,
    # 		).pvalue

    p_values = stats.tukey_hsd(*[model_names_to_dist[m] for m in model_names]).pvalue
    np.fill_diagonal(p_values, np.NaN)
    p_values[np.triu_indices_from(p_values, k=1)] = np.NaN

    # plot the pairwise matrix of the p-values and annotate the values
    import matplotlib.patheffects as patheffects
    im = ax.imshow(p_values[1:, :-1], cmap=kwargs.get("cmap", "Oranges"), vmin=0.0, vmax=1.0, aspect='auto')
    for i in range(1, len(model_names)):
        for j in range(i):
            text = ax.text(
                j, i - 1, f"{p_values[i, j]:.2f}",
                ha="center", va="center", color="w",
                path_effects=[patheffects.withStroke(linewidth=1, foreground='black')]
            )
    ax.set_xticks(np.arange(len(model_names) - 1))
    ax.set_yticks(np.arange(len(model_names) - 1))
    ax.set_xticklabels(model_names[:-1])
    ax.set_yticklabels(model_names[1:])

    add_cbar = kwargs.get("add_cbar", True)
    if add_cbar:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("p-value", rotation=90, va="bottom", labelpad=30)
    # cbar.ax.set_title("p-value", pad=3)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    # ax.set_title("p-values of the pairwise t-test between the distributions")

    # make the return named tuple
    return_named_tuple = namedtuple("ResilienceAnalyseOnModels", ["fig", "ax", "dists"])
    return return_named_tuple(fig, ax, model_names_to_dist)


def resilience_analyse_on_models_violinplot(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    from sklearn.metrics import auc

    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)
    verbose = kwargs.get("verbose", True)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)
    dist_metric_key = kwargs.get("dist_metric_key", "sparsity")
    threshold_metric_key = kwargs.get("threshod_metric_key", "pvar")
    threshold = kwargs.get("threshold", 0.9)
    model_id_key = kwargs.get("model_id_key", "model_id")
    normalize_perf = kwargs.get("normalize_perf", True)

    ablation_strategy_key = kwargs.get("ablation_strategy_key", "ablation_strategy")
    ablation_strategy = kwargs.get("ablation_strategy", None)
    if ablation_strategy is not None and ablation_strategy_key in df.columns:
        df = df[df[ablation_strategy_key] == ablation_strategy]

    model_names = []
    dists = defaultdict(dict)
    colors = mpl.colormaps["tab10"].colors
    dale_color = {True: colors[0], False: colors[1]}
    model_type_reprs = kwargs.get("model_type_reprs", {})
    perf_as_auc = kwargs.get("perf_as_auc", False)
    keep_only_pos_perf = kwargs.get("keep_only_pos_perf", perf_as_auc)
    fig, ax = kwargs.get("fig", None), kwargs.get("ax", None)
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    group = df.groupby(["model_type", "dale"])
    for j, (model_desc, df_model) in enumerate(group):
        model_type, dale = model_desc
        model_type_repr = model_type_reprs.get(model_type, ModelType.from_name(model_type).value)
        dale_name = " + Dale" if dale else ""
        model_name = f"{model_type_repr}{dale_name}"
        model_names.append(model_name)

        df_model[dist_metric_key] = df_model[dist_metric_key].round(2)
        # sort the groups by the dist_metric_key
        df_perf_ratio = df_model.groupby(model_id_key, group_keys=False).apply(lambda x: x.sort_values(dist_metric_key))

        if normalize_perf:
            # normalize the threshold_metric_key by the first value of the group
            df_perf_ratio = df_perf_ratio.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{threshold_metric_key: x[threshold_metric_key] / x[threshold_metric_key].iloc[0]})
            )

        if keep_only_pos_perf:
            df_perf_ratio = df_perf_ratio.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.assign(**{
                    threshold_metric_key: (
                        x[threshold_metric_key].iloc[
                        :np.argwhere(np.less_equal(np.array(x[threshold_metric_key]), 0.0))[0].item()]
                        if np.any(np.less_equal(np.array(x[threshold_metric_key]), 0.0)) else x[threshold_metric_key]
                    )
                })
            )

        if perf_as_auc:
            # dist_shapes = [
            # 	x[dist_metric_key][~np.isnan(x[threshold_metric_key])].shape[0]
            # 	for _, x in df_perf_ratio.groupby(model_id_key, group_keys=False)
            # ]
            # if 1 in dist_shapes:
            # 	break_point = True
            dist = np.array([
                auc(
                    x[dist_metric_key][~np.isnan(x[threshold_metric_key])],
                    x[threshold_metric_key][~np.isnan(x[threshold_metric_key])],
                )
                for _, x in df_perf_ratio.groupby(model_id_key, group_keys=False)
                if x[dist_metric_key][~np.isnan(x[threshold_metric_key])].shape[0] > 1
            ])
        else:
            # take the df where the values are nearest the threshold
            df_perf_ratio_th = df_perf_ratio.groupby(model_id_key, group_keys=False).apply(
                lambda x: x.iloc[np.argmin(np.abs(x[threshold_metric_key] - threshold))]
            )
            dist = df_perf_ratio_th[dist_metric_key].values

        dists[model_type][dale] = {"dist": dist, "idx": j}

        # show the distribution of the "dist_metric_key" for the df_perf_ratio_th
        violin_parts = ax.violinplot(
            dist,
            showmeans=False,
            showmedians=True,
            showextrema=True,
            vert=True,
            positions=[j],
        )
        for pc in violin_parts['bodies']:
            pc.set_facecolor(dale_color[dale])
            pc.set_edgecolor(dale_color[dale])
            pc.set_alpha(0.2)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
            vp = violin_parts.get(partname, None)
            if vp is not None:
                vp.set_edgecolor('black')

    ax.set_xticks(np.arange(len(model_names)))
    ax.set_xticklabels(model_names, rotation=0, ha="center")
    dist_metric_key_name = dist_metric_key.replace("_", " ").capitalize()
    dist_metric_key_name = f"AUC {dist_metric_key_name}" if perf_as_auc else dist_metric_key_name
    threshold_metric_key_y_label = f"{threshold_metric_key}" + " ratio" if normalize_perf else ""
    ylabel = kwargs.get("ylabel", f"{dist_metric_key_name} ({threshold_metric_key_y_label}={threshold})")
    ax.set_ylabel(ylabel)

    for mt, md in dists.items():
        dists[mt]["pvalue"] = stats.ttest_ind(md[True]["dist"], md[False]["dist"], equal_var=False).pvalue
        ax.hlines(
            np.max([md[True]["dist"].max(), md[False]["dist"].max()]) + 0.02,
            xmin=np.min([md[True]["idx"], md[False]["idx"]]),
            xmax=np.max([md[True]["idx"], md[False]["idx"]]),
            color="black",
            linestyle="-",
            linewidth=mpl.rcParams["lines.linewidth"],
        )
        stars_ths = [0.05, 0.01, 0.001, 0.0001]
        n_stars = np.sum(np.less_equal(dists[mt]["pvalue"], stars_ths))
        ax.text(
            np.mean([md[True]["idx"], md[False]["idx"]]),
            # np.max(line.get_ydata()) + 0.02,
            np.max([md[True]["dist"].max(), md[False]["dist"].max()]) + 0.02,
            # f"p={md['pvalue']:.2e}",
            (f"{n_stars * '*'}" if n_stars > 0 else "n.s."),
            ha="center",
            va="bottom",
            color="black",
            fontsize=mpl.rcParams["font.size"],
        )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.05 * (ymax - ymin))
    fig.tight_layout()

    if kwargs.get("save", True):
        post_name = f"_norm" if normalize_perf else ""
        ext_list = ["pdf", "png", "svg"]
        th_str = f"{threshold_metric_key}-{str(threshold).replace('.', '_')}"
        savefig_filenames = [f"resilience_violin_{dist_metric_key}_{th_str}{post_name}.{ext}" for ext in ext_list]
        original_fig_dir = _save_figures(fig, model_objects_list[0], savefig_filenames)
        print(f"Figures saved in {original_fig_dir}")
        pbt.multiprocessing.apply_func_multiprocess(
            _copy_figures,
            iterable_of_args=[(m_objects, original_fig_dir, savefig_filenames) for m_objects in model_objects_list[1:]],
            desc="Saving figures",
            unit="model",
            verbose=verbose,
            nb_workers=kwargs.get("nb_workers", 0)
        )
    if kwargs.get("show", False):
        plt.show()
    return fig, ax


def resilience_analyse_report_graph(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
):
    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)
    ablation_strategies = kwargs.get("ablation_strategies", ["smallest", "random"])
    ax_kwargs = deepcopy(kwargs)
    ax_kwargs["save"] = False
    ax_kwargs["show"] = False
    ax_kwargs["model_type_reprs"] = {ModelType.SNN_LPF.name: "SpyLIF-LPF", ModelType.WILSON_COWAN.name: "WC"}
    fig, axes = plt.subplots(nrows=2, ncols=len(ablation_strategies), figsize=(20, 12))
    nt.visualisation.Visualise.number_axes(axes, method="text", fontsize=mpl.rcParams['font.size'], x=0, y=1)
    ablation_strategies_titles = kwargs.get("ablation_strategies_titles", {})
    cmap = mpl.colormaps["Oranges"]
    for i, ablation_strategy in enumerate(ablation_strategies):
        ax_title = ablation_strategies_titles.get(ablation_strategy, f"{ablation_strategy.capitalize()} ablation")
        axes[0, i].set_title(ax_title)
        fig, ax = resilience_analyse_on_models(
            model_objects_list, df.copy(True), ablation_strategy=ablation_strategy, fig=fig, ax=axes[0, i],
            xlim=([0, 1.0] if i == 0 else [0, 0.6]),
            # xlim=[0, 1.0],
            perf_as_auc=False, keep_only_pos_perf=False,
            **ax_kwargs
        )
        fig, ax = resilience_analyse_on_models_violinplot(
            model_objects_list, df.copy(True), ablation_strategy=ablation_strategy, fig=fig, ax=axes[1, i],
            threshold=0.8,
            # perf_as_auc=False, ylabel="Sustainable sparsity [-]",
            perf_as_auc=True, ylabel="AUC [-]", keep_only_pos_perf=True,
            **ax_kwargs
        )
        # fig, ax, dists = resilience_analyse_on_models_auc_dist(
        # 	model_objects_list, df, ablation_strategy=ablation_strategy, fig=fig, ax=axes[2, i],
        # 	cmap=cmap, add_cbar=(i == len(ablation_strategies) - 1),
        # 	**ax_kwargs
        # )
        if i != 0:
            axes[0, i].set_ylabel("")
            axes[1, i].set_ylabel("")
            axes[0, i].get_legend().remove()
    fig.tight_layout()
    if kwargs.get("save", True):
        ext_list = ["pdf", "png", "svg"]
        save_name = kwargs.get("save_name", "resilience_report")
        savefig_filenames = [f"{save_name}.{ext}" for ext in ext_list]
        original_fig_dir = _save_figures(fig, model_objects_list[0], savefig_filenames)
        print(f"Figures saved in {original_fig_dir}")
        pbt.multiprocessing.apply_func_multiprocess(
            _copy_figures,
            iterable_of_args=[(m_objects, original_fig_dir, savefig_filenames) for m_objects in model_objects_list[1:]],
            desc="Saving figures",
            unit="model",
            verbose=kwargs.get("verbose", False),
            nb_workers=kwargs.get("nb_workers", 0),
        )
    if kwargs.get("show", False):
        plt.show()
    return fig, axes


def plot_all_resilience_analyse_report_graph(
        model_objects_list: List[Dict[str, Any]],
        df: Optional[Union[pd.DataFrame, str]] = None,
        **kwargs
):
    if df is None:
        df = compute_resilience_analyse_on_models_df(model_objects_list, **kwargs)
    elif isinstance(df, str):
        df = pd.read_csv(df)

    # mpl.use('Agg')
    mpl.rcParams.update(MPL_RC_DEFAULT_PARAMS)
    ax_kwargs = deepcopy(kwargs)
    ax_kwargs["save"] = False
    ax_kwargs["show"] = False
    ax_kwargs["model_type_reprs"] = {ModelType.SNN_LPF.name: "SNN-LPF", ModelType.WILSON_COWAN.name: "WC"}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 12))
    nt.visualisation.Visualise.number_axes(axes, method="text", fontsize=mpl.rcParams['font.size'], x=0, y=1)
    _cmap = mpl.colormaps["viridis"]
    normalize = mpl.colors.Normalize(vmin=0.8, vmax=1)
    cmap = lambda x: _cmap(normalize(x))

    for i, (model_type, df_model) in enumerate(df.groupby("model_type")):
        for j, (model_dale, df_model_dale) in enumerate(df_model.groupby("dale")):
            axes[j, i].set_title(f"{ax_kwargs['model_type_reprs'][model_type]} {'+ Dale' if model_dale else ''}")
            fig, ax = plot_all_resilience_analyse_on_models(
                model_objects_list, df_model_dale.copy(True), fig=fig, ax=axes[j, i],
                xlim=[0, 1],
                perf_as_auc=False, keep_only_pos_perf=False,
                ablation_strategy="smallest",
                cmap=cmap,
                **ax_kwargs
            )
    cax = fig.add_axes([0.93, 0.1, 0.01, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=_cmap), cax=cax, label="pVar [-]", aspect=10, shrink=0.5)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    if kwargs.get("save", True):
        ext_list = ["pdf", "png", "svg"]
        save_name = kwargs.get("save_name", "resilience_report_all")
        savefig_filenames = [f"{save_name}.{ext}" for ext in ext_list]
        original_fig_dir = _save_figures(fig, model_objects_list[0], savefig_filenames)
        print(f"Figures saved in {original_fig_dir}")
        pbt.multiprocessing_tools.apply_func_multiprocess(
            _copy_figures,
            iterable_of_args=[(m_objects, original_fig_dir, savefig_filenames) for m_objects in model_objects_list[1:]],
            desc="Saving figures",
            unit="model",
            verbose=kwargs.get("verbose", False),
            nb_workers=kwargs.get("nb_workers", 0),
        )
    if kwargs.get("show", False):
        plt.show()
    return fig, axes


if __name__ == '__main__':
    import sys
    import json

    is_running_in_terminal = sys.stdout.isatty()
    sys_kwgs = get_cmd_kwargs({
        1: "./data/tr_eprop_50tr_filtered",
        "n_pts": 100,
        "n_seed": 2,
        "n_test": 32,
        # "nb_workers": max(0, psutil.cpu_count(logical=False) - 4),
        "nb_workers": 0,
    })
    print("sys_kwgs:\n", json.dumps(sys_kwgs, indent=4), "\n")
    _n_pts = int(sys_kwgs["n_pts"])
    _n_seed = int(sys_kwgs["n_seed"])
    _n_test = int(sys_kwgs["n_test"])
    try:
        df_path = os.path.join(sys_kwgs[1], "figures", f"resilience_analysis_data_p{_n_pts}_t{_n_test}_s{_n_seed}.csv")
        _models = gather_models_from_folders([
            str(p)
            for p in pathlib.Path(sys_kwgs[1]).glob("*")
            if p.is_dir() and any([f.suffix == ".pth" for f in p.glob("*")]) and "eprop" in str(p)
        ], nb_workers=int(sys_kwgs["nb_workers"]))
        _df = compute_resilience_analyse_on_models_df(
            _models,
            load=False,
            filter_models="filtered" not in sys_kwgs[1],
            savepath=df_path,
            n_pts=_n_pts,
            n_test=_n_test,
            n_seed=_n_seed,
            min_perf=0.8,
            ablation_strategies=["smallest_rm", "random_rm"],
            temp_df_folder=f"data/temp_df_ablation_p{_n_pts}_t{_n_test}",
            nb_workers=int(sys_kwgs["nb_workers"]),
        )
        resilience_analyse_report_graph(
            _models,
            # df=df_path if os.path.exists(df_path) else None,
            df=_df,
            ablation_strategies=["smallest_rm", "random_rm"],
            ablation_strategies_titles={
                "smallest": "Hierarchical connection ablations",
                "random": "Random connection ablations",
                "smallest_rm": "Hierarchical connection ablations",
                "random_rm": "Random connection ablations",
            },
            normalize_perf=True,
            # x_axis_key="connection_removed",
            # x_axis_name="Connection removed [-]",
            nb_workers=int(sys_kwgs["nb_workers"]),
            save_name=f"resilience_report_p{_n_pts}_t{_n_test}_s{_n_seed}",
            show=True,
        )
        # plot_all_resilience_analyse_report_graph(
        #     _models,
        #     df=_df,
        #     ablation_strategies_titles={
        #         "smallest": "Hierarchical connection ablations",
        #         "random": "Random connection ablations",
        #     },
        #     normalize_perf=False,
        #     nb_workers=int(sys_kwgs["nb_workers"]),
        #     save_name=f"resilience_report_all_p{_n_pts}_t{_n_test}_s{_n_seed}",
        #     save=True,
        #     show=True,
        # )
    except Exception as e:
        if is_running_in_terminal:
            print(e)
        else:
            raise e
    if is_running_in_terminal:
        _ = input("Press Enter to close.")
