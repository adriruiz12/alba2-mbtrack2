"""
replot.py — Regenerates the diagnostic plots from a completed run
without repeating the tracking.

VALIDATION_CASE must be set to match the original run: it determines
which output directory to read from 
(results/soa_base/soa_base_<VALIDATION_CASE>) 
and which target values to overlay on the plots.

diagnostics.py imports mpi4py at module level, so a minimal stub is
injected into sys.modules before the import. FakeMPIComm stands in for
MPI.COMM_WORLD, implementing only the two methods diagnostics.py calls
(Get_size, Get_rank). This avoids requiring an active MPI environment
just to regenerate plots.

Usage:
    VALIDATION_CASE=goal python replot.py
"""

import types
import sys

# Stub mpi4py (diagnostics.py imports it, but replot does not need MPI)
mpi = types.ModuleType("mpi4py")
mpi.MPI = types.ModuleType("mpi4py.MPI")

class FakeMPIComm:
    def Get_size(self): return 1
    def Get_rank(self): return 0

mpi.MPI.COMM_WORLD = FakeMPIComm()
sys.modules["mpi4py"] = mpi
sys.modules["mpi4py.MPI"] = mpi.MPI

from constants import CIRCUMFERENCE_M, C_LIGHT
from diagnostics import load_run_data, make_plots

class Ring:
    """Minimal ring stub: only T0 is needed by make_plots."""
    
    T0 = CIRCUMFERENCE_M / C_LIGHT

history, final_data = load_run_data()
make_plots(Ring(), history, final_data)
print("Done.")