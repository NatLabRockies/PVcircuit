# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
pvcircuit.Junction()
properties and methods for each junction
"""
from __future__ import annotations

import copy
import math  # simple math
import os
from datetime import datetime
from functools import lru_cache
from time import time
from typing import List

import numpy as np  # arrays
import scipy.constants as con  # physical constants
from parse import parse
from scipy.optimize import brentq  # root finder

from pvcircuit.conversions import DB_PREFIX, HC_E, K_Q, TK, Vth

# constants

# Junction defaults
Eg_DEFAULT = 1.1  # [eV]
SIGMA_DEFAULT = 0  # [eV]
TC_REF = 25.0  # [C]
AREA_DEFAULT = 1.0  # [cm2] note: if A=1, then I->J
BETA_DEFAULT = 15.0  # unitless

# numerical calculation parameters
VLIM_REVERSE = 10.0
VLIM_FORWARD = 3.0
VTOL = 0.0001
EPSREL = 1e-15
MAXITER = 1000

GITpath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

@lru_cache(maxsize=100)
def Jdb(TC: float, Eg: float, sigma: float = 0):
    # detailed balance saturation current

    EgkT = Eg / Vth(TC)

    # Jdb from Geisz et al.
    # return DB_PREFIX * TK(TC) ** 3.0 * (EgkT * EgkT + 2.0 * EgkT + 2.0) * np.exp(-EgkT)  # units from DB_PREFIX

    # Jdb from Rau et al.
    return (
        DB_PREFIX
        * TK(TC) ** 3.0
        * (EgkT * EgkT + 2.0 * EgkT + 2.0 - 2 * sigma**2 * Eg / (Vth(TC)) ** 3 - sigma**2 / (Vth(TC)) ** 2 + sigma**4 / (Vth(TC)) ** 4)
        * np.exp(-EgkT + sigma**2 / (2 * (Vth(TC)) ** 2))
    )  # units from DB_PREFIX


def timestamp(fmt="%y%m%d-%H%M%S", tm=None) -> str:
    # return a timestamp string with given format and epoch time
    if tm is None:
        tm = time()
    date_time = datetime.fromtimestamp(tm)
    return date_time.strftime(fmt)


def newoutpath(dname: str = None) -> str:
    # return a new output within pvc_output
    if os.path.exists(GITpath):
        pvcoutpath = os.path.join(GITpath, "pvc_output")
        if not os.path.exists(pvcoutpath):
            os.mkdir(pvcoutpath)

        if dname is None:
            dname = timestamp()
        else:
            dname += timestamp()

        newpath = os.path.join(pvcoutpath, dname)
        if not os.path.exists(newpath):
            os.mkdir(newpath)

        return newpath


class Junction(object):
    """
    Class for PV junctions.
    """

    ATTR = ["Eg", "TC", "Gsh", "Rser", "area", "lightarea", "totalarea", "Jext", "JLC", "beta", "gamma", "pn", "Jphoto", "TK", "Jdb", "RBB"]
    ARY_ATTR = ["n", "J0ratio", "J0"]
    # Internal numerical-stability factor (mA/cm^2). With ``Jdb ~ 1e-26`` and
    # ``J0 ~ 1e-21`` the ratio ``J0/Jdb^(1/n)`` would otherwise span ~20 orders
    # of magnitude. Scaling by 1000 keeps ``J0ratio`` close to O(1) so fits and
    # serialisation stay numerically well-behaved. Do not change.
    J0scale = 1000.0

    def __init__(
        self,
        name: str = "junc",
        Eg: float = Eg_DEFAULT,
        sigma: float = SIGMA_DEFAULT,
        TC: float = TC_REF,
        Gsh: float = 0.0,
        Rser: float = 0.0,
        area: float = AREA_DEFAULT,
        n: List[float] = [1.0, 2.0],
        J0ratio: float = None,
        J0ref: float = None,
        RBB: float = None,
        Jext: float = 0.04,
        JLC: float = 0.0,
        J0default: float = 10.0,
        pn: int = -1,
        beta: float = BETA_DEFAULT,
        gamma: float = 0.0,
    ):

        self.ui = None
        # self.debugout = widgets.Output()  # debug output
        self.RBB_dict = {}

        # user inputs
        self.name = name  # remember my name
        self.Eg = np.float64(Eg)  #: [eV] junction band gap
        self.sigma = np.float64(sigma)  #: [eV] junction band gap sigma (Urbach tail width)
        self.TC = np.float64(TC)  #: [\degC] junction temperature (use the ``TK`` property for Kelvin)
        self.Jext = np.float64(Jext)  #: [A/cm^2] external photocurrent density (printed as mA/cm^2)
        self.Gsh = np.float64(Gsh)  #: [S/cm^2] shunt conductance (= 1/Rsh, area-normalised)
        self.Rser = np.float64(Rser)  #: [\Omega*cm^2] series resistance (area-normalised, so ``Vdrop = Rser * J``)
        self.lightarea = np.float64(area)  #: [cm^2] illuminated junction area
        self.totalarea = np.float64(area)  #: [cm^2] total junction area including shaded regions
        # used for tandems only
        self.pn = int(pn)  #: polarity flag: +1 for p-on-n, -1 for n-on-p (sign convention)
        self.beta = np.float64(beta)  #: [unitless] luminescent coupling efficiency (top -> bottom radiative coupling)
        self.gamma = np.float64(gamma)  #: [unitless] photoluminescent coupling coefficient (Lan et al. PL parameter)
        self.JLC = np.float64(JLC)  #: [A/cm^2] luminescent coupling current density injected from the previous junction (``beta * Jem`` of the neighbour)

        # multiple diodes
        # n=1 bulk, n=m SNS, and n=2/3 Auger mechanisms
        ndiodes = len(n)
        self.n = np.array(n)  # diode ideality list e.g. [n0, n1]
        if J0ref:  # input list of absolute J0
            if len(J0ref) == ndiodes:  # check length
                self._J0init(J0ref)  # calculate self.J0ratio from J0ref at current self.TC
            else:
                print("J0ref mismatch", ndiodes, len(J0ref))
                self.J0ratio = np.full_like(n, J0default)  # default J0ratio
        elif J0ratio:  # input list of relative J0 ratios
            if len(J0ratio) == ndiodes:  # check length
                self.J0ratio = np.array(J0ratio)  # diode J0/Jdb^(1/n) ratio list for T dependence
            else:
                print("J0ratio mismatch", ndiodes, len(J0ratio))
                self.J0ratio = np.full_like(n, J0default)  # default J0ratio
        else:  # create J0ratio
            self.J0ratio = np.full_like(n, J0default)  # default J0ratio

        self.set(RBB=RBB)

    def copy(self) -> Junction:
        """
        create a copy of a Junction
        need deepcopy() to separate lists, dicts, etc but crashes
        """
        tmp = copy.copy(self)
        # manual since deepcopy does not work
        tmp.n = self.n.copy()
        tmp.J0ratio = self.J0ratio.copy()
        tmp.RBB_dict = self.RBB_dict.copy()
        return tmp

    def __str__(self):
        # attr_list = self.__dict__.keys()
        # attr_dict = self.__dict__.items()
        # print(attr_list)

        strout = self.name + ": <pvcircuit.junction.Junction class>"

        strout += "\nEg = {0:.2f} eV, TC = {1:.1f} C".format(self.Eg, self.TC)

        strout += "\nJext = {0:.1f} mA/cm2, JLC = {1:.1f} mA/cm2".format(self.Jext * 1000.0, self.JLC * 1000.0)

        strout += "\nGsh = {0:g} S/cm2, Rser = {1:g} Ωcm2".format(self.Gsh, self.Rser)

        strout += "\nlightA = {0:g} cm2, totalA = {1:g} cm2".format(self.lightarea, self.totalarea)

        strout += "\npn = {0:d}, beta = {1:g}, gamma = {2:g}".format(self.pn, self.beta, self.gamma)

        strout += "\n {0:^5s} {1:^10s} {2:^10s}".format("n", "J0ratio", "J0(A/cm2)")
        strout += "\n {0:^5s} {1:^10.0f} {2:^10.3e}".format("db", 1.0, self.Jdb)

        i = 0
        for i, _ in enumerate(self.n):
            strout += "\n {0:^5.2f} {1:^10.2f} {2:^10.3e}".format(self.n[i], self.J0ratio[i], self.J0[i])
            i += 1

        if self.RBB_dict["method"]:
            strout += " \nRBB_dict: " + str(self.RBB_dict)

        return strout

    def __repr__(self):
        return str(self)

    # """
    # def __setattr__(self, key, value):
    #     # causes problems
    #     super(Junction, self).__setattr__(key, value)
    #     self.set(key = value)
    # """

    # def update(self):
    #     # update Junction self.ui controls

    #     if self.ui:  # junction user interface has been created
    #         if self.RBB_dict:
    #             if self.RBB_dict["method"]:
    #                 RBB_keys = list(self.RBB_dict.keys())
    #             else:
    #                 RBB_keys = []

    #         cntrls = self.ui.children
    #         for cntrl in cntrls:
    #             desc = cntrl.trait_values().get("description", "nodesc")  # control description
    #             cval = cntrl.trait_values().get("value", "noval")  # control value
    #             if desc == "nodesc" or cval == "noval":
    #                 break
    #             elif desc.endswith("]") and desc.find("[") > 0:
    #                 key, ind = parse("{}[{:d}]", desc)
    #             else:
    #                 key = desc
    #                 ind = None

    #             if key in self.ATTR:  # Junction scalar controls to update
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if cval != attrval:
    #                     with self.debugout:
    #                         print("Jupdate: " + desc, attrval)
    #                     cntrl.value = attrval
    #             elif key in self.ARY_ATTR:  # Junction array controls to update
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if isinstance(ind, int):
    #                     if isinstance(attrval, np.ndarray):
    #                         if cval != attrval[ind]:
    #                             with self.debugout:
    #                                 print("Jupdate: " + desc, attrval[ind])
    #                             cntrl.value = attrval[ind]
    #             elif key in RBB_keys:
    #                 attrval = self.RBB_dict[key]
    #                 if cval != attrval:
    #                     with self.debugout:
    #                         print("Jupdate: " + desc, attrval)
    #                     cntrl.value = attrval

    def set(self, **kwargs):
        # controlled update of Junction attributes

        # with self.debugout:
        #     print("Jset(" + self.name + "): ", list(kwargs.keys()))

        for testkey, value in kwargs.items():
            if testkey.endswith("]") and testkey.find("[") > 0:
                key, ind = parse("{}[{:d}]", testkey)  # set one element of array e.g. 'n[0]'
            else:
                key = testkey
                ind = None

            if self.RBB_dict:
                if self.RBB_dict["method"]:
                    RBB_keys = list(self.RBB_dict.keys())
                else:
                    RBB_keys = []

            if key == "RBB" or key == "method":
                # this change requires redrawing self.ui
                if value == "JFG":  # RBB shortcut
                    self.__dict__["RBB_dict"] = {"method": "JFG", "mrb": 10.0, "J0rb": 0.5, "Vrb": 0.0}
                elif value == "bishop":
                    self.__dict__["RBB_dict"] = {"method": "bishop", "mrb": 3.28, "avalanche": 1.0, "Vrb": -5.5}
                else:
                    self.__dict__["RBB_dict"] = {"method": None}  # no RBB
                if self.ui:  # junction user interface has been created
                    # ui = self.controls()    # redraw junction controls
                    pass
            elif key in RBB_keys:  # RBB parameters
                self.RBB_dict[key] = np.float64(value)
            elif key == "area":  # area shortcut
                self.__dict__["lightarea"] = np.float64(value)
                self.__dict__["totalarea"] = np.float64(value)
            elif key == "name":  # strings
                self.__dict__[key] = str(value)
            elif key == "pn":  # integers
                self.__dict__[key] = int(value)
            elif key == "RBB_dict":
                self.__dict__[key] = value
            elif key in ["n", "J0ratio"]:  # diode parameters (array)
                if isinstance(ind, int) and np.isscalar(value):
                    attrval = getattr(self, key)  # current value of attribute
                    localarray = attrval.copy()
                    if isinstance(localarray, np.ndarray):
                        if ind < localarray.size:
                            localarray[ind] = np.float64(value)  # add new value
                            self.__dict__[key] = localarray
                            # with self.debugout:
                            #     print("scalar", key, ind, localarray)
                        else:
                            raise IndexError(f"invalid junction index. Set index is {ind+1} but junction size is {localarray.size}")
                else:
                    # check if both, n and J0ratio, are set if they have the same size
                    if "n" in kwargs.keys() and "J0ratio" in kwargs.keys():
                        if not len(kwargs["n"]) == len(kwargs["J0ratio"]):
                            raise ValueError("n and J0ratio must be same size")

                    # if only n or J0ratio is set, check if it matches current diode configuration
                    elif not len(value) == len(self.n) and not len(value) == len(self.J0ratio):
                        raise ValueError("setting single n or J0ratio value must match previous number of diodes")

                    self.__dict__[key] = np.array(value)
                    # with self.debugout:
                    #     print("array", key, value)
            elif key in self.ATTR:  # scalar float
                self.__dict__[key] = np.float64(value)

            # raise error if the key is not in the class attributes
            elif key not in list(self.__dict__.keys()):
                raise ValueError(f"invalid class attribute {key}")
                # with self.debugout:
                #     print("ATTR", key, value)
            # else:
            #     with self.debugout:
            #         print("no Junckey", key)

    @property
    def Jphoto(self) -> float:
        """[A/cm^2] total photocurrent density on the junction.

        Combines the external photocurrent (scaled by the illuminated
        fraction of the total area) and the luminescent-coupling current
        injected from a neighbouring junction:

            'Jphoto = Jext * (lightarea / totalarea) + JLC'.

        The area scaling assumes 'Jext' is referenced to 'lightarea' and
        spreads it uniformly across 'totalarea' so shaded regions
        contribute zero photocurrent.
        """
        return self.Jext * self.lightarea / self.totalarea + self.JLC

    @property
    def TK(self) -> float:
        """[K] junction temperature in Kelvin (derived from :attr:'TC')."""
        return TK(self.TC)

    @property
    def Vth(self) -> float:
        """[V] thermal voltage kT/q at the current junction temperature."""
        return Vth(self.TC)

    @property
    def Jdb(self) -> float:
        """[A/cm^2] radiative (detailed-balance) saturation current density.

        Computed from :attr:'Eg', :attr:'sigma', and :attr:'TC' via the Rau et
        al. formulation. This is a thermodynamic quantity (not a free
        parameter) and forms the irreducible lower bound on :attr:'J0'.
        """
        return Jdb(self.TC, self.Eg, self.sigma)

    @property
    def J0(self) -> float:
        """[A/cm^2] per-diode saturation current densities '[J0(n0), J0(n1), ...]'.

        Recomputed on every access from :attr:'Jdb', :attr:'n', and
        :attr:'J0ratio' using the formula

            'J0[i] = (Jdb * J0scale)^(1/n[i]) * J0ratio[i] / J0scale'

        where 'J0scale = 1000' is an internal numerical-stability factor
        (see the class-level comment on :attr:'J0scale'). Because
        :attr:'Jdb' depends on temperature, :attr:'J0' automatically tracks
        changes in :attr:'TC'.
        """

        if (isinstance(self.n, np.ndarray)) and (isinstance(self.J0ratio, np.ndarray)):
            if self.n.size == self.J0ratio.size:
                return (self.Jdb * self.J0scale) ** (1.0 / self.n) * self.J0ratio / self.J0scale
            else:
                return np.nan  # different sizes
        else:
            return np.nan  # not numpy.ndarray

    def _J0init(self, J0ref: float):
        """
        initialize self.J0ratio from J0ref
        """
        J0ref = np.array(J0ref)
        if self.n.size == J0ref.size:
            self.J0ratio = self.J0scale * J0ref / (self.Jdb * self.J0scale) ** (1.0 / self.n)
        else:
            raise ValueError("J0ref and n must be same size")

    def Jem(self, Vmid: float) -> float:
        """
        light emitted from junction by reciprocity
        quantified as current density
        """
        # TODO swich to raise instead return
        if Vmid > 0.0:
            Jem = self.Jdb * (np.exp(Vmid / self.Vth) - 1.0)  # EL Rau
            Jem += self.gamma * self.Jphoto  # PL Lan and Green
            return Jem
        else:
            return 0.0

    def notdiode(self) -> bool:
        """
        is this junction really a diode
        or just a resistor
        sum(J0) = 0 -> not diode
        pn = 0 -> not diode
        """
        if self.pn == 0:
            return True

        jsum = np.float64(0.0)
        for saturation_current in self.J0:
            jsum += saturation_current

        return jsum == np.float64(0.0)

    def Jmultidiodes(self, Vdiode: float) -> float:
        """
        calculate recombination current density from
        multiple diodes self.n, self.J0 numpy.ndarray
        two-diodes:
        n  = [1, 2]  #two diodes
        J0 = [10,10]  #poor cell
        detailed balance:
        n  = [1]
        J0 = [1]
        three-diodes
        n = [1, 1.8, (2/3)]
        """
        Jrec = np.float64(0.0)
        for ideality_factor, saturation_current in zip(self.n, self.J0):
            if ideality_factor > 0.0 and math.isfinite(saturation_current):
                # try:
                Jrec += saturation_current * (np.exp(Vdiode / self.Vth / ideality_factor) - 1.0)
                # except ValueError:
                # continue

        return Jrec

    def JshuntRBB(self, Vdiode: float) -> float:
        """
        return shunt + reverse-bias breakdown current

            RBB_dict={'method':None}   #None

            RBB_dict={'method':'JFG', mrb'':10., 'J0rb':1., 'Vrb':0.}

            RBB_dict={'method':'bishop','mrb'':3.28, 'avalanche':1, 'Vrb':-5.5}

            RBB_dict={'method':'pvmismatch','ARBD':arbd,'BRBD':brbd,'VRBD':vrb,'NRBD':nrbd:

        Vdiode without Rs
        Vth = kT
        Gshunt
        """

        RBB_dict = self.RBB_dict
        method = RBB_dict["method"]
        JRBB = np.float64(0.0)

        if method == "JFG":
            Vrb = RBB_dict["Vrb"]
            J0rb = RBB_dict["J0rb"]
            mrb = RBB_dict["mrb"]
            if Vdiode <= Vrb and mrb != 0.0:
                # JRBB = -J0rb * (self.Jdb)**(1./mrb) * (np.exp(-Vdiode / self.Vth / mrb) - 1.0)
                JRBB = -J0rb * (self.Jdb * 1000) ** (1.0 / mrb) / 1000.0 * (np.exp(-Vdiode / self.Vth / mrb) - 1.0)

        elif method == "bishop":
            Vrb = RBB_dict["Vrb"]
            a = RBB_dict["avalanche"]
            mrb = RBB_dict["mrb"]
            if Vdiode <= 0.0 and Vrb != 0.0:
                JRBB = Vdiode * self.Gsh * a * (1.0 + Vdiode / Vrb) ** (-mrb)

        elif method == "pvmismatch":
            raise NotImplementedError(
                "RBB method 'pvmismatch' is documented but not implemented. "
                "Use RBB='JFG', RBB='bishop', or RBB=None."
            )

        # else:
        #     JRBB = self.J0.sum()

        return Vdiode * self.Gsh + JRBB

    def Jparallel(self, Vdiode: float, Jtot: float) -> float:
        """
        circuit equation to be zeroed to solve for Vi
        for voltage across parallel diodes with shunt and reverse breakdown
        """

        if self.notdiode():  # sum(J0)=0 -> no diode
            return Jtot

        JLED = self.Jmultidiodes(Vdiode)
        JRBB = self.JshuntRBB(Vdiode)
        # JRBB = JshuntRBB(Vdiode, self.Vth, self.Gsh, self.RBB_dict)
        return Jtot - JLED - JRBB

    def Vdiode(self, Jdiode: float) -> float:
        """
        Jtot = Jphoto + J
        for junction self of class Junction
        return Vdiode(Jtot)
        no Rseries here
        """

        if self.notdiode():  # sum(J0)=0 -> no diode
            return 0.0

        # Jtot = max(self.Jphoto + Jdiode, 0)
        Jtot = self.Jphoto + Jdiode
        # if self.RBB_dict["method"] is None:
        #     Jtot = max(Jtot, -1 * self.J0.sum())

        try:
            Vdiode = brentq(
                self.Jparallel,
                -VLIM_REVERSE,
                VLIM_FORWARD,
                args=(Jtot),
                xtol=VTOL,
                rtol=EPSREL,
                maxiter=MAXITER,
                full_output=False,
                disp=True,
            )
        except ValueError:
            return np.nan
            # print("Exception:",err)

        return Vdiode

    def _dV(self, Vmid: float, Vtot: float) -> float:
        """
        see singlejunction
        circuit equation to be zeroed (returns voltage difference) to solve for Vmid
        single junction circuit with series resistance and parallel diodes
        """

        J = self.Jparallel(Vmid, self.Jphoto)
        dV = Vtot - Vmid + J * self.Rser
        return dV

    def Vmid(self, Vtot: float) -> float:
        """
        see Vparallel
        find intermediate voltage in a single junction diode with series resistance
        Given Vtot=Vparallel + Rser * Jparallel
        """

        if self.notdiode():  # sum(J0)=0 -> no diode
            return 0.0

        try:
            Vmid = brentq(
                self._dV,
                -VLIM_REVERSE,
                VLIM_FORWARD,
                args=(Vtot),
                xtol=VTOL,
                rtol=EPSREL,
                maxiter=MAXITER,
                full_output=False,
                disp=True,
            )

        except ValueError:
            return np.nan
            # print("Exception:",err)

        return Vmid

    # def controls(self):
    #     """
    #     use interactive_output for GUI in IPython
    #     """

    #     cell_layout = widgets.Layout(display="inline_flex", flex_flow="row", justify_content="flex-end", width="300px")
    #     # controls
    #     in_name = widgets.Text(value=self.name, description="name", layout=cell_layout, continuous_update=False)
    #     in_Eg = widgets.FloatSlider(
    #         value=self.Eg, min=0.1, max=3.0, step=0.01, description="Eg", layout=cell_layout, readout_format=".2f"
    #     )
    #     in_TC = widgets.FloatSlider(
    #         value=self.TC, min=-40, max=200.0, step=2, description="TC", layout=cell_layout, readout_format=".1f"
    #     )
    #     in_Jext = widgets.FloatSlider(
    #         value=self.Jext, min=0.0, max=0.080, step=0.001, description="Jext", layout=cell_layout, readout_format=".4f"
    #     )
    #     in_JLC = widgets.FloatSlider(
    #         value=self.JLC,
    #         min=0.0,
    #         max=0.080,
    #         step=0.001,
    #         description="JLC",
    #         layout=cell_layout,
    #         readout_format=".4f",
    #         disabled=True,
    #     )
    #     in_Gsh = widgets.FloatLogSlider(
    #         value=self.Gsh, base=10, min=-12, max=3, step=0.01, description="Gsh", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_Rser = widgets.FloatLogSlider(
    #         value=self.Rser, base=10, min=-7, max=3, step=0.01, description="Rser", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_lightarea = widgets.FloatLogSlider(
    #         value=self.lightarea, base=10, min=-6, max=3.0, step=0.1, description="lightarea", layout=cell_layout
    #     )
    #     in_totalarea = widgets.FloatSlider(
    #         value=self.totalarea, min=self.lightarea, max=1e3, step=0.1, description="totalarea", layout=cell_layout
    #     )
    #     in_beta = widgets.FloatSlider(
    #         value=self.beta, min=0.0, max=50.0, step=0.1, description="beta", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_gamma = widgets.FloatSlider(
    #         value=self.gamma, min=0.0, max=3.0, step=0.1, description="gamma", layout=cell_layout, readout_format=".2e"
    #     )
    #     in_pn = widgets.IntSlider(value=self.pn, min=-1, max=1, step=1, description="pn", layout=cell_layout)

    #     # linkages
    #     # arealink = widgets.jslink((in_lightarea, "value"), (in_totalarea, "min"))  # also jsdlink works

    #     # attr = ["name"] + self.ATTR.copy()
    #     cntrls = [in_name, in_Eg, in_TC, in_Gsh, in_Rser, in_lightarea, in_totalarea, in_Jext, in_JLC, in_beta, in_gamma, in_pn]
    #     # sing_dict = dict(zip(attr, cntrls))
    #     # singout = widgets.interactive_output(self.set, sing_dict)  #all at once

    #     def on_juncchange(change):
    #         # function for changing values
    #         old = change["old"]  # old value
    #         new = change["new"]  # new value
    #         owner = change["owner"]  # control
    #         value = owner.value
    #         desc = owner.description

    #         if new == old:
    #             with self.debugout:
    #                 print("Jcontrol: " + desc + "=", value)
    #         else:
    #             with self.debugout:
    #                 print("Jcontrol: " + desc + "->", value)
    #             self.set(**{desc: value})

    #         # iout.clear_output()
    #         # with iout: print(self)

    #     # diode array
    #     in_tit = widgets.Label(value="Junction", description="Junction")
    #     in_diodelab = widgets.Label(value="diodes:", description="diodes:")
    #     # diode_layout = widgets.Layout(flex_flow="column", align_items="center")

    #     cntrls.append(in_diodelab)
    #     in_n = []  # empty list of n controls
    #     in_ratio = []  # empty list of Jratio controls
    #     diode_dict = {}
    #     for i in range(len(self.n)):
    #         in_n.append(
    #             widgets.FloatLogSlider(
    #                 value=self.n[i], base=10, min=-1, max=1, step=0.001, description="n[" + str(i) + "]", layout=cell_layout
    #             )
    #         )
    #         in_ratio.append(
    #             widgets.FloatLogSlider(
    #                 value=self.J0ratio[i],
    #                 base=10,
    #                 min=-6,
    #                 max=6,
    #                 step=0.1,
    #                 description="J0ratio[" + str(i) + "]",
    #                 layout=cell_layout,
    #             )
    #         )
    #         cntrls.append(in_n[i])
    #         cntrls.append(in_ratio[i])
    #         diode_dict["n[" + str(i) + "]"] = in_n[i]
    #         diode_dict["J0ratio[" + str(i) + "]"] = in_ratio[i]
    #         # hui.append(widgets.HBox([in_n[i],in_ratio[i]]))
    #         # cntrls.append(hui[i])

    #     # diodeout = widgets.interactive_output(self.set, diode_dict)  #all at once

    #     if self.RBB_dict:
    #         RBB_keys = list(self.RBB_dict.keys())
    #         in_rbblab = widgets.Label(value="RBB:", description="RBB:")
    #         cntrls.append(in_rbblab)
    #         in_rbb = []  # empty list of n controls
    #         for i, key in enumerate(RBB_keys):
    #             with self.debugout:
    #                 print("RBB:", i, key)
    #             if key == "method":
    #                 in_rbb.append(
    #                     widgets.Dropdown(
    #                         options=["", "JFG", "bishop"],
    #                         value=self.RBB_dict[key],
    #                         description=key,
    #                         layout=cell_layout,
    #                         continuous_update=False,
    #                     )
    #                 )
    #             else:
    #                 in_rbb.append(
    #                     widgets.FloatLogSlider(
    #                         value=self.RBB_dict[key], base=10, min=-10, max=5, step=0.1, description=key, layout=cell_layout
    #                     )
    #                 )
    #             cntrls.append(in_rbb[i])

    #     for cntrl in cntrls:
    #         cntrl.observe(on_juncchange, names="value")

    #     # output
    #     iout = widgets.Output()
    #     iout.layout.height = "5px"
    #     # with iout: print(self)
    #     cntrls.append(iout)

    #     # user interface
    #     box_layout = widgets.Layout(
    #         display="flex", flex_flow="column", align_items="center", border="1px solid black", width="320px", height="350px"
    #     )

    #     ui = widgets.VBox([in_tit] + cntrls, layout=box_layout)
    #     self.ui = ui  # make it an attribute

    #     return ui
