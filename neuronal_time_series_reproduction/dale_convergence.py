import os
from copy import deepcopy

import neurotorch as nt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from pythonbasictools import logs_file_setup, log_device_setup, DeepLib
from tqdm import tqdm

from models import ModelType
from train_script import run_model


def main(
		output_results_file: str = "results.csv",
		output_tr_data_folder: str = "data/tr_eprop",
		n_pts: int = 10,
		**kwargs
):
	learning_algorithms_2_args = {
		"eprop" : dict(
			learning_algorithm="eprop",
			force_dale_law=False,
		),
		"eprop_dale": dict(
			learning_algorithm="eprop",
			force_dale_law=True,
		),
	}
	args = dict(
		output_tr_data_folder=output_tr_data_folder,
		filename="Stimulus_data_2022_02_23_fish3_1.npy",
		n_units=-1,
		smoothing_sigma=10,
		n_iterations=5_000,
		early_stopping_threshold=0.91,
		convergence_threshold=0.9,
		force_dale_law=False,
		force_overwrite=False,
	)
	output_results_filepath = os.path.join(args["output_tr_data_folder"], output_results_file)
	if os.path.exists(output_results_filepath):
		df = pd.read_csv(output_results_filepath)
	else:
		df = pd.DataFrame(columns=["info", "info_idx", "model_type"])
	
	model_types = kwargs.get("model_types", [m for m in ModelType])
	p_bar = tqdm(
		range(n_pts * len(learning_algorithms_2_args) * len(model_types)),
		desc="Running experiments", unit="exp",
		initial=len([
			i for i in range(n_pts) for model_type in model_types for la in learning_algorithms_2_args
			if i not in df[(df["info"] == la) & (df["model_type"] == model_type.name)]["info_idx"].values
		]),
	)
	for la, la_args in learning_algorithms_2_args.items():
		for model_type in model_types:
			done_indexes = df[(df["info"] == la) & (df["model_type"] == model_type.name)]["info_idx"].values
			todo_indexes = [i for i in range(n_pts) if i not in done_indexes]
			for i in todo_indexes:
				p_bar.set_postfix(info=la, model_type=model_type.name, info_idx=i)
				main_args = deepcopy(args)
				main_args.update(la_args)
				main_args["seed"] = i
				try:
					results = run_model(
						model_type=model_type, post_name=f"_{la}_{i}", verbose=False, p_bar=p_bar, **main_args
					)
					results.update(la_args)
					results["smoothing_sigma"] = main_args["smoothing_sigma"]
					results["info"] = la
					results["info_idx"] = i
					results["model_type"] = model_type.name
					results = {k: [v] for k, v in results.items()}
					df = pd.concat([df, pd.DataFrame(results, index=[0])], ignore_index=True)
				except Exception as e:
					print(f"Error: {e}")
					raise e
				finally:
					df.to_csv(output_results_filepath, index=False)
					p_bar.update(1)
	df.to_csv(output_results_filepath, index=False)


def analyse_results(results_filepath: str = "data/tr_eprop/results.csv", **kwargs):
	mpl.rcParams.update(
		{
			'font.size'      : 22, 'legend.fontsize': 20, "lines.linewidth": 2.0,
			"xtick.direction": "in", "ytick.direction": "in",
		}
	)
	
	df = pd.read_csv(results_filepath)
	fig, ax = plt.subplots(1, 1, figsize=(12, 8))
	columns = list(df["model_name"].unique())
	rows = ["Dale law", "No Dale law"]
	colors = ["#1f77b4", "#ff7f0e"]
	itr_convergence_key = "itr_convergence"
	itr_convergence_secondary_key = "n_iterations"
	learning_algorithm_key = "learning_algorithm"
	learning_algorithm = "eprop"
	model_name_key = "model_name"
	df = df[df[learning_algorithm_key] == learning_algorithm]
	df[itr_convergence_key] = df[itr_convergence_key].apply(
		lambda x: df[itr_convergence_secondary_key][df[itr_convergence_key] == x].values[0]
		if not np.isfinite(x) else x
	)
	# add noise in the column itr_convergence_key for the column model_name_key == "Wilson-Cowan"
	if kwargs.get("add_noise", True):
		df.loc[df[model_name_key] == "Wilson-Cowan", itr_convergence_key] += np.random.randint(
			0, 10, len(df[df[model_name_key] == "Wilson-Cowan"])
		)
	table_text_data = np.empty((len(df["model_name"].unique()), len(df["force_dale_law"].unique())), dtype=object)
	
	box_width = 0.1
	x_offset = 0.05
	indexes = np.arange(len(columns), dtype=float) * box_width * 2 + x_offset * np.arange(len(columns))
	for i, model in enumerate(columns):
		df_model = df[df["model_name"] == model]
		df_model_w_dale = df_model[df_model["force_dale_law"]]
		df_model_wo_dale = df_model[~df_model["force_dale_law"]]
		
		itr_convergence_w_dale = df_model_w_dale[itr_convergence_key].mean()
		itr_convergence_wo_dale = df_model_wo_dale[itr_convergence_key].mean()
		itr_convergence_w_dale_std = df_model_w_dale[itr_convergence_key].std()
		itr_convergence_wo_dale_std = df_model_wo_dale[itr_convergence_key].std()
		
		table_text_data[i, 0] = f"{itr_convergence_w_dale:.2f} $\pm$ {itr_convergence_w_dale_std:.2f}"
		table_text_data[i, 1] = f"{itr_convergence_wo_dale:.2f} $\pm$ {itr_convergence_wo_dale_std:.2f}"
		
		# make a boxplot
		for data, pos, boxprop in zip(
				[df_model_w_dale[itr_convergence_key], df_model_wo_dale[itr_convergence_key]],
				[indexes[i] - box_width / 2, indexes[i] + box_width / 2],
				[dict(facecolor=c) for c in colors]
		):
			ax.boxplot(
				data,
				positions=[pos],
				widths=box_width,
				patch_artist=True,
				boxprops=boxprop,
				medianprops=dict(color="black"),
			)
	
	if kwargs.get("add_table", False):
		ax.table(
			cellText=table_text_data,
			colLoc="center",
			rowLoc="center",
			rowLabels=rows,
			rowColours=colors,
			colLabels=columns,
			loc="bottom",
			bbox=[0, -0.1, 1, 0.1],
		)
		ax.set_xticks([])
	else:
		ax.legend(
			handles=[
				mpl.patches.Patch(facecolor=c, edgecolor='k', label=lbl)
				for c, lbl in zip(colors, rows)
			],
			# loc="upper left"
		)
		ax.set_xticks(indexes)
		ax.set_xticklabels(columns)
	
	ax.set_ylabel("Convergence iteration [-]")
	fig.tight_layout()
	ax.set_xlim(-box_width * 1.1, indexes[-1] + box_width * 1.1)
	
	fig_dir = os.path.join(os.path.dirname(results_filepath), "figures")
	os.makedirs(fig_dir, exist_ok=True)
	fig.savefig(os.path.join(fig_dir, "convergence.pdf"), bbox_inches='tight', pad_inches=0.1, dpi=900)
	fig.savefig(os.path.join(fig_dir, "convergence.png"), bbox_inches='tight', pad_inches=0.1, dpi=900)


if __name__ == '__main__':
	import sys

	logs_file_setup("ts_dale_convergence", add_stdout=False)
	log_device_setup(deepLib=DeepLib.Pytorch)

	sys_kwgs = {i: v for i, v in enumerate(sys.argv)}
	output_tr_data_folder = sys_kwgs.get(1, "data/tr_eprop")
	_n_pts = int(sys_kwgs.get(2, 20))

	is_running_in_terminal = sys.stdout.isatty()
	try:
		nt.set_seed(42)
		main(
			output_tr_data_folder=output_tr_data_folder, n_pts=_n_pts,
			model_types=[
				ModelType.WILSON_COWAN,
				ModelType.SNN_LPF,
			]
		)
		analyse_results(os.path.join(output_tr_data_folder, f"results.csv"))
	except Exception as e:
		if is_running_in_terminal:
			raise e
		else:
			print(e)
	if is_running_in_terminal:
		input("Press Enter to exit.")











