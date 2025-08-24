from invoke import task, Collection
from tasks_core import (
    toolkit_command,
    setup_all, setup, run, list_toolkits, status
)

@task
@toolkit_command
def e2e(c, toolkit=None):
    """Build across toolkits."""
    pass

@task
@toolkit_command
def sample_data(c, toolkit=None):
    """Generate / process sample data across toolkits."""
    pass

ns = Collection(
    setup_all, setup,
    run,
    sample_data, e2e,
    list_toolkits, status
)
