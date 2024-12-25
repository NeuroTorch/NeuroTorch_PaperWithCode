r"""
This script is used to run all the scripts to generate the data for the figures in the paper.
Be aware that this script can take days to run, so it is recommended to run it in a cluster.

Before running this script, make sure to have the virtual environment and all the dependencies installed.
The python version used is 3.10.

The scripts that are run are:
- `eprop_training.py` -> This script trains the models using the e-Prop algorithm.
- `gen_figure_models_comparison.py` -> This script generates the figures that compare the models from the
    `eprop_training.py` script.
- `dale_convergence.py` -> This script run multiple trainings of the models until they converge to a performance
    threshold.
- `_filter_models.py` -> This script filters the models that have a performance above a certain threshold.
- `resilience.py` -> This script runs the models that have been filtered by the `_filter_models.py` script and
    analyses the resilience of the models to ablations.

Note that all the scripts are seeded for reproducibility. However, the results may vary slightly due to the
numerical precision of the machine.
"""
import os
import argparse
from typing import List
import pythonbasictools as pbt
import json


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--venv",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "venv")),
        help="Path to the virtual environment folder.",
    )
    parser.add_argument(
        "--nb_workers",
        type=int,
        default=0,
        help="Number of workers to use for running the scripts.",
    )
    return parser


def run_cmds(cmds: List[str]):
    for cmd in cmds:
        print(f"Running command: {cmd}")
        os.system(cmd)
    return 0



def main():
    parser = get_parser()
    args = parser.parse_args()

    print(f"Running scripts with the following arguments:")
    print(json.dumps(vars(args), indent=4))

    python_path = os.path.join(args.venv, "Scripts", "python.exe")
    if not os.path.exists(python_path):
        python_path = os.path.join(args.venv, "bin", "python")
    if not os.path.exists(python_path):
        raise FileNotFoundError(
            f"Python path not found: {python_path}. "
            f"Please create the virtual environment and install the dependencies."
            f"You can do this by running the following commands:\n"
            f"python -m venv {args.venv}\n"
            f"{python_path} -m pip install -r requirements.txt"
        )
    dir_path = os.path.dirname(os.path.realpath(__file__))

    eprop_scripts = [
        "eprop_training.py",
        "gen_figure_models_comparison.py",
    ]
    resilience_scripts = [
        "dale_convergence.py",
        "_filter_models.py",
        "resilience.py",
    ]

    eprop_cmds = [f"{python_path} {os.path.join(dir_path, script)}" for script in eprop_scripts]
    resilience_cmds = [f"{python_path} {os.path.join(dir_path, script)}" for script in resilience_scripts]

    pbt.multiprocessing_tools.apply_func_multiprocess(
        run_cmds,
        iterable_of_args=[(eprop_cmds, ), (resilience_cmds, )],
        desc="Running scripts",
        nb_workers=args.nb_workers,
    )
    return 0


if __name__ == '__main__':
    exit(main())



