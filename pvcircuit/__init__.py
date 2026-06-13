# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
Model tandem and multijunction solar cells
Written by John Geisz at the National Renewable Energy Laboratory
December, 2021
Based on publications:
    J. F. Geisz, et al., IEEE Journal of Photovoltaics 5, p. 1827 (2015).
    http://dx.doi.org/10.1109/JPHOTOV.2015.2478072

    J. F. Geisz, et al., Cell Reports Physical Science 2, p. 100677 (2021).
    https://doi.org/10.1016/j.xcrp.2021.100677


This module contains the classes:
    pvc.Junction()   # properties and methods for each junction
    pvc.IV3T()       # many forms of operational conditions of 3T tandems
    pvc.Tandem3T()   # properties of a 3T tandem including 2 junctions
    pvc.Multi2T()    # properties of a 2T multijunction with arbitrary junctions
    pvc.TMY()        # properties of a typical meterological year
    pvc.EQE()        # properties of tandem external quantum efficiency measurements
"""

# set global data paths first
from pathlib import Path

pvcpath = Path(__file__).parent
datapath = pvcpath.joinpath(r"data")
notebook_datapath = pvcpath.parent.joinpath(r"notebooks/notebook_data")


# Submodule imports must follow the path constants above because each of these
# modules touches ``pvc.pvcpath`` / ``pvc.datapath`` at import time.
import pvcircuit.EY as EY  # noqa: E402
import pvcircuit.iv3T as iv3T  # noqa: E402
import pvcircuit.junction as junction  # noqa: E402
import pvcircuit.multi2T as multi2T  # noqa: E402
import pvcircuit.qe as qe  # noqa: E402
import pvcircuit.tandem3T as tandem3T  # noqa: E402

# expose constructors to package's top level
Junction = junction.Junction
TK = junction.TK
Vth = junction.Vth
Jdb = junction.Jdb

Multi2T = multi2T.Multi2T
IV3T = iv3T.IV3T
Tandem3T = tandem3T.Tandem3T

JdbFromEg = qe.JdbFromEg
EgFromJdb = qe.EgFromJdb
JdbMD = qe.JdbMD
JintMD = qe.JintMD
PintMD = qe.PintMD
EQE = qe.EQE

VERSION = 0.05

__author__ = "John Geisz"
__email__ = "john.geisz@nrel.gov"
__url__ = "https://github.nrel.gov/jgeisz/PVcircuit"
__version__ = VERSION
__release__ = "development"
__all__ = ["junction", "multi2T", "iv3T", "tandem3T", "qe", "EY"]
