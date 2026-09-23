"""Explicit, auditable graph-to-dynamics prototype; no biological fidelity claim."""

from .graph import Graph
from .lif import LIF, LIFParameters, RateReadout

__all__ = ["Graph", "LIF", "LIFParameters", "RateReadout"]
