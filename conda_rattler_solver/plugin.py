from conda import plugins
from conda.plugins.types import CondaSolver

from .solver import RattlerSolver


@plugins.hookimpl
def conda_solvers():
    """
    The conda plugin hook implementation to load the solver into conda.
    """
    yield CondaSolver(
        name="rattler",
        backend=RattlerSolver,
    )
