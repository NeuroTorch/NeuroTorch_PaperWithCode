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


if __name__ == '__main__':
    venv = "../venv"
    python_path = os.path.join(venv, "Scripts", "python")
    if not os.path.exists(python_path):
        python_path = os.path.join(venv, "bin", "python")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.system(f"{python_path} {os.path.join(dir_path, 'eprop_training.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, 'gen_figure_models_comparison.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, 'dale_convergence.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, '_filter_models.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, 'resilience.py')}")


