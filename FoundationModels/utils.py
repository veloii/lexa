import shutil
import os

def get_conda_cmd():
    """Auto-detect conda implementation with environment variable override"""
    env_conda = os.getenv('CONDA_CMD')
    if env_conda and shutil.which(env_conda):
        return env_conda
        
    for cmd in ['micromamba', 'mamba', 'conda']:
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("No conda implementation found")

def run_in_env(c, env_name, command, conda_cmd=None):
    """Helper to run command in conda environment"""
    if conda_cmd is None:
        conda_cmd = get_conda_cmd()
    c.run(f"{conda_cmd} run -n {env_name} {command}")
