from . import options as opt
from .context import Context
from .errors import AnalysisError
from .registry import (analysis, r_analysis, prepare_long, prepare_long_subject,
                       run_analysis, ANALYSES)
from .results import Results, Table, stars, fmt_p
from .plotstyle import new_figure, COLORS

__all__ = ["opt", "Context", "AnalysisError", "analysis", "r_analysis",
           "prepare_long", "prepare_long_subject", "run_analysis", "ANALYSES",
           "Results", "Table", "stars", "fmt_p", "new_figure", "COLORS"]
