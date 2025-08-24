from invoke import task, Collection
import sys
import importlib.util
from pathlib import Path
from functools import wraps
from utils import get_conda_cmd

# -----------------
# Discovery / loading
# -----------------
def get_toolkits():
    """Discover all toolkit directories (presence of environment.yml)."""
    return sorted([
        d for d in Path('.').iterdir()
        if d.is_dir() and (d / 'environment.yml').exists()
    ])

def load_toolkit_tasks(toolkit_path: Path):
    """Load tasks Collection ('ns') from a toolkit's tasks.py, if present."""
    tasks_file = toolkit_path / 'tasks.py'
    if not tasks_file.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"{toolkit_path.name}.tasks", tasks_file)
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(Path.cwd()))
        spec.loader.exec_module(module)  # type: ignore
        sys.path.pop(0)
        return getattr(module, 'ns', None)
    except Exception as e:
        print(f"Error loading {toolkit_path}: {e}")
        return None

# -----------------
# Task resolution helpers
# -----------------
def _name_variants(base: str):
    snake = base
    kebab = base.replace('_', '-')
    return [snake] if snake == kebab else [snake, kebab]

def _resolve_task(tasks_obj, base_name):
    """
    Find a task/callable inside an Invoke Collection matching base_name
    trying snake_case and kebab-case variants.
    Returns (callable, resolved_name) or (None, None).
    """
    from invoke import Collection  # local import to avoid circulars if any
    variants = _name_variants(base_name)

    # Direct attribute lookup (rare here—Collections usually used)
    for v in variants:
        if '-' in v:
            continue
        if hasattr(tasks_obj, v):
            return getattr(tasks_obj, v), v

    # Collection lookup
    if isinstance(tasks_obj, Collection):
        for v in variants:
            if v in tasks_obj.tasks:
                return tasks_obj.tasks[v], v

    # Fallback swap
    for v in variants:
        alt = v.replace('-', '_') if '-' in v else v.replace('_', '-')
        if hasattr(tasks_obj, alt):
            return getattr(tasks_obj, alt), alt
        if isinstance(tasks_obj, Collection) and alt in tasks_obj.tasks:
            return tasks_obj.tasks[alt], alt

    return None, None

# -----------------
# Decorator for cross-toolkit tasks
# -----------------
def toolkit_command(func):
    """Decorator that runs a task across one or all toolkits, with snake/kebab normalization."""
    @wraps(func)
    def wrapper(c, toolkit=None, **kwargs):
        python_name = func.__name__
        kebab_name = python_name.replace('_', '-')
        variants = _name_variants(python_name)
        display_name = kebab_name if kebab_name != python_name else python_name

        def run_in_toolkit(toolkit_path: Path):
            tasks_obj = load_toolkit_tasks(toolkit_path)
            if not tasks_obj:
                print(f"  ⏭️  {toolkit_path.name}: no tasks object")
                return False

            task_callable, resolved = _resolve_task(tasks_obj, python_name)
            if not task_callable:
                print(f"  ⏭️  {toolkit_path.name}: missing any of {variants}")
                return False

            invoke_body = getattr(task_callable, 'body', None)
            callable_obj = invoke_body or task_callable

            print(f"  → {toolkit_path.name} (using '{resolved}')")
            try:
                with c.cd(toolkit_path):
                    callable_obj(c, **kwargs)
                return True
            except Exception as e:
                print(f"    ❌ Failed in {toolkit_path.name}: {e}")
                return False

        if toolkit:
            toolkit_path = Path(toolkit)
            if not toolkit_path.exists():
                print(f"❌ Toolkit '{toolkit}' not found")
                return
            print(f"🚀 Running '{display_name}' in {toolkit_path.name} (variants tried: {variants})")
            ok = run_in_toolkit(toolkit_path)
            if ok:
                print(f"✅ {display_name} completed in {toolkit_path.name}")
            else:
                print(f"❌ {display_name} not executed in {toolkit_path.name}")
        else:
            print(f"🚀 Running '{display_name}' across all toolkits (variants: {variants})")
            executed = 0
            total = 0
            for toolkit_path in get_toolkits():
                total += 1
                if run_in_toolkit(toolkit_path):
                    executed += 1
            print(f"✅ Executed '{display_name}' in {executed}/{total} toolkits")

    return wrapper

# -----------------
# Utility tasks (still exposed)
# -----------------
@task
def setup_all(c):
    """Set up all toolkit environments."""
    conda_cmd = get_conda_cmd()
    for toolkit in get_toolkits():
        print(f"Setting up {toolkit.name}...")
        with c.cd(toolkit):
            c.run(f"{conda_cmd} env create -f environment.yml")

@task
def setup(c, toolkit):
    """Set up specific toolkit: invoke setup toolkit-name"""
    path = Path(toolkit)
    if not path.exists():
        print(f"Toolkit {toolkit} not found")
        return
    conda_cmd = get_conda_cmd()
    with c.cd(path):
        c.run(f"{conda_cmd} env create -f environment.yml")

@task
def run(c, command, toolkit=None):
    """Run any task in one or all toolkits (snake_case or kebab-case)."""
    def execute(toolkit_path: Path):
        tasks_obj = load_toolkit_tasks(toolkit_path)
        if not tasks_obj:
            print(f"❌ {toolkit_path.name}: no tasks object")
            return
        task_callable, resolved = _resolve_task(tasks_obj, command)
        if not task_callable:
            print(f"❌ {toolkit_path.name}: command '{command}' not found")
            return
        invoke_body = getattr(task_callable, 'body', None)
        callable_obj = invoke_body or task_callable
        print(f"🚀 {toolkit_path.name}: {resolved}")
        with c.cd(toolkit_path):
            callable_obj(c)

    if toolkit:
        tpath = Path(toolkit)
        if not tpath.exists():
            print(f"❌ Toolkit {toolkit} not found")
            return
        execute(tpath)
    else:
        for tk in get_toolkits():
            execute(tk)

@task
def list_toolkits(c):
    """List all toolkits and their available commands."""
    conda_cmd = get_conda_cmd()
    print(f"Using: {conda_cmd}\n")
    for toolkit in get_toolkits():
        tasks = load_toolkit_tasks(toolkit)
        if tasks:
            names = sorted(tasks.tasks.keys())
        else:
            names = []
        print(f"📁 {toolkit.name}")
        if names:
            print(f"   Commands: {', '.join(names)}")
        else:
            print("   No tasks.py or no tasks")
        print()

@task
def status(c):
    """Show environment status (which toolkit envs exist)."""
    conda_cmd = get_conda_cmd()
    result = c.run(f"{conda_cmd} info -e", hide=True)
    existing_envs = {
        line.split()[0] for line in result.stdout.split('\n')
        if line.strip() and not line.startswith('#')
    }
    print("Environment Status:")
    for toolkit in get_toolkits():
        env_exists = toolkit.name in existing_envs
        print(f"📁 {toolkit.name}: {'✅' if env_exists else '❌'}")
