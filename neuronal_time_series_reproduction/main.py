import os


if __name__ == '__main__':
    venv = "../venv"
    python_path = os.path.join(venv, "Scripts", "python")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.system(f"{python_path} {os.path.join(dir_path, 'eprop_training.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, 'dale_convergence.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, '_filter_models.py')}")
    os.system(f"{python_path} {os.path.join(dir_path, 'resilience.py')}")


