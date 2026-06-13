# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.Multi2T()    # properties of a 2T multijunction with arbitrary junctions
"""

from __future__ import annotations

import copy
import math  # simple math
import os
from time import time

import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd
from scipy.optimize import root, root_scalar

from pvcircuit import junction
from pvcircuit.junction import Junction

# from pvcircuit.tandem3T import Tandem3T

VTOL = 1e-15
EPSREL = 1e-15
MAXITER = 1000


class Multi2T(object):
    """
    Multi2T class for optoelectronic model of two terminal multijunction
    """

    junctioncolors = [
        ["black"],  # 0J
        ["red"],  # 1J
        ["blue", "red"],  # 2J
        ["blue", "green", "red"],  # 3J
        ["blue", "green", "orange", "red"],  # 4J
        ["purple", "blue", "green", "orange", "red"],  # 5J
        ["purple", "blue", "green", "black", "orange", "red"],
    ]  # 6J

    def __init__(
        self,
        name: str = "Multi2T",
        TC: float = junction.TC_REF,
        Rs2T: float = 0.0,
        area: float = 1.0,
        Jext: float = 0.014,
        Eg_list: list[float] = None,
        n: list[int] = None,
        J0ratio: float = None,
        J0ref: float = None,
    ):
        # user inputs
        # note n and J0ratio much be same size

        # Avoid mutable default args (PEP 8 footgun).
        if Eg_list is None:
            Eg_list = [1.8, 1.4]
        if n is None:
            n = [1, 2]

        self.Vpoints = None
        self.Ipoints = None

        # self.debugout = widgets.Output()  # debug output
        # self.debugout.layout.height = '400px'

        self.name = name
        self.Rs2T = Rs2T
        self.njuncs = len(Eg_list)
        # CM = 0.03 / self.njuncs #TODO remove
        self.Vmid = np.full(self.njuncs, np.nan, dtype=np.float64)  # subcell voltages
        self.j = list()  # empty list of junctions
        for i, Eg in enumerate(Eg_list):
            jname = "j[" + str(i) + "]"
            self.j.append(Junction(name=jname, Eg=Eg, TC=TC, n=n, J0ratio=J0ratio, J0ref=J0ref, Jext=Jext, area=area))

        self.j[0].set(beta=0.0)

    def copy(self) -> Multi2T:
        """
        Create an independent copy of this Multi2T.

        'copy.deepcopy' crashes on this class, so we build the copy manually:
        the wrapper is shallow-copied, but the contained junction list and
        each junction inside it are duplicated via 'Junction.copy()'.
        Mutating a junction on the returned object will therefore NOT
        affect the original device.
        """
        tmp = copy.copy(self)
        tmp.j = [junc.copy() for junc in self.j]
        tmp.Vmid = self.Vmid.copy()
        return tmp

    @classmethod
    def from_3T(cls, dev3T: object, copy_attributes: bool = True) -> Multi2T:
        """
        create a Multi2T object that contains the values from a Tandem3T object
        input dev3T is Tandem3T object
        output is a Multi2T object
        """
        top = dev3T.top
        bot = dev3T.bot

        dev2T = cls(name=dev3T.name, TC=dev3T.TC, Eg_list=[top.Eg, bot.Eg])
        # Rs2T [\Omega*cm^2] = area-weighted average of the two Rser values
        # (each Rser is itself in \Omega*cm^2, so the formula has dimensions of
        # \Omega*cm^2 * cm^2 / cm^2 -> \Omega*cm^2).
        dev2T.set(Rs2T=(top.Rser * top.totalarea + bot.Rser * bot.totalarea) / dev3T.totalarea)

        if copy_attributes:
            dev2T.j[0] = dev3T.top.copy()  # disconnected
            dev2T.j[1] = dev3T.bot.copy()
        else:
            dev2T.j[0] = dev3T.top  # dynamically connected
            dev2T.j[1] = dev3T.bot

        # dev2T.j[0].Rser = 0.
        # dev2T.j[1].Rser = 0.

        return dev2T

    @classmethod
    def from_single_junction(cls, junc: Junction, copy_attributes: bool = True):
        """
        create a 2T single junction cell from a Junction object
        from this you can calculate Voc, Jsc, MPP, plot, etc.
        """
        dev2T = cls(name=junc.name, TC=junc.TC, Eg_list=[junc.Eg])
        dev2T.set(Rs2T=junc.Rser)

        if copy_attributes:
            dev2T.j[0] = junc.copy()  # disconnected
        else:
            dev2T.j[0] = junc  # dynamically connected

        return dev2T
    
    def append_junction(self, junc: Junction, copy_attributes: bool = True) -> None:
        """
        append a junction to the Multi2T
        """

        self.njuncs += 1
        self.Vmid = np.append(self.Vmid, np.nan)  # add new subcell voltage
        # Keep Rs2T in \Omega*cm^2: parallel-combine the area-normalised resistances
        # (Rs2T/totalarea, Rser/totalarea), then re-normalise by the smaller area.
        self.set(Rs2T=(self.Rs2T / self.totalarea + junc.Rser / junc.totalarea) * np.min([self.totalarea, junc.totalarea]))  # update Rs2T
        
        if copy_attributes:
            self.j.append(junc.copy())  # disconnected
        else:
            self.j.append(junc)  # dynamically connected

    def __str__(self):
        """
        format concise description of Multi2T object
        """

        strout = self.name + ": <pvcircuit.multi2T.Multi2T class>"

        strout += "\nT = {0:.1f} C, Rs2T= {1:g} Ω cm2".format(self.TC, self.Rs2T)

        for i in range(self.njuncs):
            strout += "\n\n" + str(self.j[i])

        return strout

    def __repr__(self):
        return str(self)

    @property
    def TC(self) -> float:
        # largest junction TC
        TCs = self.proplist("TC")
        return max(TCs)

    @property
    def lightarea(self) -> float:
        # largest junction light area
        areas = self.proplist("lightarea")
        return max(areas)

    @property
    def totalarea(self) -> float:
        # largest junction light area
        areas = self.proplist("totalarea")
        return max(areas)

    # def update(self):
    #     # update Multi2T self.ui controls with manually entered values

    #     for junc in self.j:
    #         junc.update()

    #     if self.ui:  # Multi2T user interface has been created
    #         Boxes = self.ui.children
    #         for cntrl in Boxes[2].children:  # Multi2T controls
    #             desc = cntrl.trait_values().get("description", "nodesc")  # does not fail when not present
    #             cval = cntrl.trait_values().get("value", "noval")  # does not fail when not present
    #             if desc in ["name", "Rs2T"]:  # Multi2T controls to update
    #                 key = desc
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if cval != attrval:
    #                     # with self.debugout:
    #                     #     print("Mupdate: " + key, attrval)
    #                     cntrl.value = attrval
    #             if desc == "Recalc":
    #                 cntrl.click()  # click button

    def set(self, **kwargs):
        # controlled update of Multi2T attributes

        # with self.debugout:
        #     print("Mset: ", list(kwargs.keys()))

        # junction kwargs
        jlist = Junction.ATTR + Junction.ARY_ATTR
        jkwargs = {key: kwargs.pop(key) for key in jlist if key in kwargs}  # Filter keys from kwargs that are junction attributes.
        if len(jkwargs) > 0:
            # if called_from: jkwargs['called_from'] = called_from # pass it on to the junction.set()
            for i, junc in enumerate(self.j):
                jikwargs = {}  # empty
                for key, value in jkwargs.items():
                    if key in Junction.ATTR and not np.isscalar(value):
                        # dimension mismatch possibly from self.proplist()
                        jikwargs[key] = value[i]
                    else:
                        jikwargs[key] = value
                # with self.debugout:
                #     print("M2J[" + str(i) + "]: ", jikwargs)
                junc.set(**jikwargs)

        # remaining Multi2T kwargs
        for key, value in kwargs.items():
            if key == "name":
                self.__dict__[key] = str(value)
            elif key == "njuncs":
                self.__dict__[key] = int(value)
            elif key == "Vmid":
                self.__dict__[key] = np.array(value)
            elif key in ["j"]:
                self.__dict__[key] = value
            elif key in ["Rs2T"]:
                self.__dict__[key] = np.float64(value)
            # raise error if the key is not in the class attributes
            elif key not in list(self.__dict__.keys()):
                raise ValueError(f"invalid class attribute {key}")

    def V2T(self, I: float) -> float:  # noqa: E741
        """
        calcuate V(J) of 2T multijunction
        """

        # convert to prevent brent warnings
        if isinstance(I, np.ndarray) and I.size == 1:
            I = I.item()  # noqa: E741
        else:
            I = np.float64(I)  # noqa: E741

        for i in range(self.njuncs):
            if i > 0:  # previous LC
                self.j[i].JLC = self.j[i].beta * self.j[i - 1].Jem(self.Vmid[i - 1])
                if self.j[i - 1].totalarea < self.j[i].totalarea:  # distribute LC over total area
                    self.j[i].JLC *= self.j[i - 1].totalarea / self.j[i].totalarea

            else:
                self.j[i].JLC = 0.0  # no LC in top junction

            self.Vmid[i] = self.j[i].Vdiode(I / self.j[i].totalarea)

        # Rs2T [\Omega*cm^2] * I [A] / totalarea [cm^2] = V (series-resistance drop).
        Vtot = np.sum(self.Vmid) + self.Rs2T * I / self.totalarea

        # TODO raise error
        if not math.isfinite(Vtot):  # if one nan all nan
            # for i in range(self.njuncs): self.Vmid[i]=np.nan
            pass

        return Vtot

    def Imaxrev(self):
        # find max rev-bias current (w/o Gsh or breakdown)
        # Voc = self.Voc()  # this also calculates JLC at Voc TODO Remove
        J0s = self.proplist("J0")
        Jphotos = self.proplist("Jphoto")
        Jmaxs = Jphotos + np.sum(J0s, axis=1)
        areas = self.proplist("totalarea")
        Imax = max([j * a for j, a in zip(Jmaxs, areas)])
        return Imax

    def _I2T_root_target(self, current, target_voltage):
        """
        Helper function for the root finding method in I2T function.
        """
        voltage = self.V2T(current)
        # if np.isnan(voltage):
        # voltage = self.Voc()
        return voltage - target_voltage

    def I2T(self, V):
        """
        calculate J(V) of 2T multijunction
        using Dan's algorithm
        """

        V = np.float64(V)
        Voc = self.Voc()  # this also calculates JLC at Voc
        Imax = self.Imaxrev()
        stepratio = 10
        count = 0
        if V <= Voc:  # Voc toward Jsc
            Idelta = -Imax / stepratio
        else:  # Voc toward forward
            Idelta = Imax

        Idelta_start = Idelta
        Iold = 0.0  # start from Voc
        Itrace = [Iold]
        while abs(Idelta / Idelta_start) > 1e-7:
            Inew = Iold + Idelta
            Vnew = self.V2T(Inew)
            if not math.isfinite(Vnew):
                Idelta /= stepratio
            elif (Vnew < V) and (V <= Voc):
                Idelta /= stepratio
            elif (Vnew > V) and (V > Voc):
                Idelta /= stepratio
            else:
                Iold = Inew
                Itrace.append(Iold)

            count += 1
            if count > junction.MAXITER:
                Iold = np.nan
                break

        return Iold

    def I2Troot(self, V):
        """
        calculate J(V) of 2T multijunction
        using root finding algorithm
        """

        V = np.float64(V)

        # Assuming the current at open circuit is zero and using Imaxrev for the reverse bias limit
        I_min = -1 * self.Imaxrev()
        I_max = 0

        # Use fsolve as an alternative to brentq
        initial_guess = I_max

        try:
            I_solution = root_scalar(self._I2T_root_target, x0=initial_guess, args=(V,), bracket=[I_min, I_max], method="brenth", xtol=1e-15)
            if not I_solution.converged:
                raise ValueError
            return I_solution.root
        except ValueError as err:
            print(f"root_scalar failed {err}. Fall back to root")
            I_solution = root(self._I2T_root_target, x0=initial_guess, args=(V,), method="lm", tol=1e-15, options={"xtol": 1e-15, "ftol": 1e-15, "maxiter": 10000, "factor": 0.1})
            if np.isfinite(I_solution.x):
                return I_solution.x[0]
            else:
                raise RuntimeError("Both root_scalar and root failed.")

    def proplist(self, key):
        # list of junction properties

        out = []  # list NOT ndarray
        for junc in self.j:
            try:
                value = getattr(junc, key)
            except AttributeError:
                value = np.nan

            out.append(value)  # append scalar or array as list item

        return np.array(out)

    def Voc(self):
        return self.V2T(0.0)

    def Isc(self):
        return abs(self.I2T(0.0))

    def MPP(self, pnts=11, bplot=False, timer=False):
        # calculate maximum power point and associated IV, Vmp, Imp, FF
        # res=0.001   #voltage resolution

        ts = time()
        Voc = self.Voc()
        Isc = self.Isc()
        Ilo = -Isc
        Ihi = 0.0  # 1mA forward
        # ndarray functions
        V2Tvect = np.vectorize(self.V2T)

        Jext_list = self.proplist("Jext")  # list external photocurrents at Voc
        if math.isclose(max(Jext_list), 0.0, abs_tol=1e-6):
            Pmp = np.nan
            Vmp = np.nan
            Imp = np.nan
            FF = np.nan

        else:
            if bplot:  # debug plot
                _, ax = plt.subplots()
                ax.axhline(0, color="gray")
                ax.axvline(0, color="gray")
                ax.set_title(self.name + " MPP")
                ax.set_xlabel("Voltage (V)")
                ax.set_ylabel("Current Density (A/cm2)")
                axr = ax.twinx()
                axr.set_ylabel("Power (W)", c="cyan")

            for _ in range(5):
                Itemp = np.linspace(Ilo, Ihi, pnts)
                # Vtemp = np.array([self.V2T(I) for I in Itemp])
                Vtemp = V2Tvect(Itemp)
                Ptemp = np.array([(-v * j) for v, j in zip(Vtemp, Itemp)])
                nmax = np.argmax(Ptemp)
                if bplot:
                    ax.plot(Vtemp, Itemp, marker=".", ls="")
                    axr.plot(Vtemp, Ptemp, marker=".", ls="", c="cyan")
                    print(nmax, Ilo, Ihi, Ptemp[nmax])
                Ilo = Itemp[max(0, (nmax - 1))]
                Ihi = Itemp[min((nmax + 1), (pnts - 1))]

            Pmp = Ptemp[nmax]
            Vmp = Vtemp[nmax]
            Imp = abs(Itemp[nmax])
            FF = abs((Vmp * Imp) / (Voc * Isc))

            self.Vpoints = np.array([0.0, Vmp, Voc])
            self.Ipoints = np.array([-Isc, -Imp, 0.0])
            if bplot:
                ax.plot(self.Vpoints, self.Ipoints, marker="x", ls="", ms=12, c="black")  # special points
                axr.plot(Vmp, Pmp, marker="o", fillstyle="none", ms=12, c="black")

        mpp_dict = {"Voc": Voc, "Isc": Isc, "Vmp": Vmp, "Imp": Imp, "Pmp": Pmp, "FF": FF}

        te = time()
        ds = te - ts
        if timer:
            print(f"MPP {ds:2.4f} s")

        return mpp_dict

    # def controls(self):
    #     """
    #     use interactive_output for GUI in IPython
    #     """
    #     tand_layout = widgets.Layout(width="300px", height="40px")
    #     vout_layout = widgets.Layout(width="180px", height="40px")
    #     junc_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")
    #     multi_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")

    #     replot_types = [
    #         widgets.widgets.widget_float.BoundedFloatText,
    #         widgets.widgets.widget_int.BoundedIntText,
    #         widgets.widgets.widget_int.IntSlider,
    #         widgets.widgets.widget_float.FloatSlider,
    #         widgets.widgets.widget_float.FloatLogSlider,
    #     ]

    #     def on_2Tchange(change):
    #         # function for changing values
    #         old = change["old"]  # old value
    #         new = change["new"]  # new value
    #         owner = change["owner"]  # control
    #         value = owner.value
    #         desc = owner.description
    #         with self.debugout:
    #             print("Mcontrol: " + desc + "->", value)
    #         self.set(**{desc: value})

    #     def on_2Treplot(change):
    #         # change info
    #         fast = True
    #         if type(change) is widgets.widgets.widget_button.Button:
    #             owner = change
    #             desc = owner.description
    #         else:  # other controls
    #             owner = change["owner"]  # control
    #         desc = owner.description
    #         if desc == "Recalc":
    #             fast = False
    #         elif desc == "savefig":
    #             fast = False

    #         # recalculate
    #         ts = time()
    #         Idark, Vdark, Vdarkmid = self.calcDark()
    #         Vlight, Ilight, Plight, Vlightmid, MPP = self.calcLight(fast=fast)
    #         Voc = MPP["Voc"]
    #         Imax = self.Imaxrev()
    #         Eg_list = self.proplist("Eg")  # list of Eg
    #         Egmax = sum(Eg_list)
    #         scale = 1000.0
    #         fmtstr = "Fit:  Voc = {0:>5.3f} V, Isc = {1:>6.2f} mA, FF = {2:>4.1f}%, "
    #         fmtstr += "Pmp = {3:>5.1f} mW, Vmp = {4:>5.3f} V, Imp = {5:>6.2f} mA"
    #         fmtstr += ", Eff = {6:>5.2f} %"
    #         outstr = fmtstr.format(
    #             MPP["Voc"],
    #             MPP["Isc"] * scale,
    #             MPP["FF"] * 100,
    #             MPP["Pmp"] * scale,
    #             MPP["Vmp"],
    #             MPP["Imp"] * scale,
    #             (MPP["Pmp"] * scale / self.lightarea),
    #         )

    #         # out_Voc.value = ('{0:>7.3f} V'.format(MPP['Voc']))
    #         # out_Isc.value = ('{0:>7.2f} mA'.format(MPP['Isc']*scale))
    #         # out_FF.value = ('{0:>7.1f}%'.format(MPP['FF']*100))
    #         # out_Pmp.value = ('{0:>7.1f} mW'.format(MPP['Pmp']*scale))
    #         # out_Vmp.value = ('{0:>7.3f} V'.format(MPP['Vmp']))
    #         # out_Imp.value = ('{0:>7.2f} mA'.format(MPP['Imp']*scale))

    #         VoutBox.clear_output()
    #         with VoutBox:
    #             print(outstr)

    #         with Lout:  # left output device -> dark
    #             # replot
    #             if desc == "Eg":
    #                 dax.set_xlim(right=Egmax * 1.1)
    #                 dax.set_ylim(np.nanmin(abs(Idark)), np.nanmax(abs(Idark)))

    #             lines = dax.get_lines()
    #             for line in lines:
    #                 linelabel = line.get_label()
    #                 if linelabel in ["pdark", "ndark"]:
    #                     if linelabel.startswith("p"):
    #                         line.set_data(Vdark, Idark)
    #                     elif linelabel.startswith("n"):
    #                         line.set_data(Vdark, -Idark)
    #                 elif linelabel.find("junction") >= 0:  # pjunction0, njunction0, etc
    #                     for junc in range(self.njuncs):
    #                         if linelabel.endswith("junction" + str(junc)):
    #                             if linelabel.startswith("p"):
    #                                 line.set_data(Vdarkmid[:, junc], Idark)
    #                             elif linelabel.startswith("n"):
    #                                 line.set_data(Vdarkmid[:, junc], -Idark)

    #         with Rout:  # right output device -> light
    #             # replot
    #             if desc == "Eg":
    #                 lax.set_xlim(right=max(min(Egmax, Voc * 1.1), 0.1))
    #                 lax.set_ylim(-Imax * 1.5 * scale, Imax * 1.5 * scale)
    #             lines = lax.get_lines()
    #             for line in lines:
    #                 linelabel = line.get_label()
    #                 if linelabel.find("dark") >= 0:
    #                     line.set_data(Vdark, Idark * scale)
    #                 elif linelabel.find("light") >= 0:
    #                     line.set_data(Vlight, Ilight * scale)
    #                 elif linelabel.find("points") >= 0:
    #                     line.set_data(self.Vpoints, self.Ipoints * scale)
    #                 elif linelabel.find("junction") >= 0:  # ljunction0, djunction0, etc
    #                     for junc in range(self.njuncs):
    #                         if linelabel.endswith("junction" + str(junc)):
    #                             if linelabel.startswith("d"):
    #                                 line.set_data(Vdarkmid[:, junc], Idark * scale)
    #                             elif linelabel.startswith("l"):
    #                                 line.set_data(Vlightmid[:, junc], Ilight * scale)

    #                 elif linelabel.find("light") >= 0:
    #                     line.set_data(Vlight, Ilight * scale)
    #                 elif linelabel.find("points") >= 0:
    #                     line.set_data(self.Vpoints, self.Ipoints * scale)

    #             if False:
    #                 Jext_list = self.proplist("Jext")  # remember list external photocurrents
    #                 snote = "T = {0:.1f} C, Rs2T = {1:g} \Omega cm2, A = {2:g} cm2".format(self.TC, self.Rs2T, self.lightarea)
    #                 snote += "\nEg = " + str(Eg_list) + " eV"
    #                 snote += "\nJext = " + str(Jext_list * 1000) + " mA/cm2"
    #                 snote += "\nVoc = {0:.3f} V, Isc = {1:.2f} mA/cm2\nFF = {2:.1f}%, Pmp = {3:.1f} mW".format(
    #                     Voc, MPP["Isc"] * 1000, MPP["FF"] * 100, MPP["Pmp"] * 1000
    #                 )
    #                 kids = lax.get_children()
    #                 for kid in kids:
    #                     if kid.get_label() == "mpptext":
    #                         kid.set(text=snote)

    #         te = time()
    #         dt = te - ts

    #         if desc == "savefig":
    #             outpath = junction.newoutpath(self.name)

    #             strout = str(self)
    #             strout += "\n\nMPP:" + str(MPP)
    #             with open(os.path.join(outpath, self.name + ".txt"), "wt") as fout:
    #                 fout.write(strout)

    #             # save mathplotlib graphs
    #             dax.get_figure().savefig(os.path.join(outpath, "dax.png"))
    #             lax.get_figure().savefig(os.path.join(outpath, "lax.png"))

    #             # assemble data into 2D arrays then dataframes for saving
    #             npdark = np.column_stack((Idark, Vdark, Vdarkmid))
    #             dfdark = pd.DataFrame(data=npdark, index=None, columns=["Idark", "Vdark", "Vdarkmid"])
    #             dfdark.to_csv(os.path.join(outpath, "darkfit.csv"), index=False)

    #             nplight = np.column_stack((Vlight, Ilight, Plight, Vlightmid))
    #             dflight = pd.DataFrame(data=nplight, index=None, columns=["Vlight", "Ilight", "Plight", "Vlightmid"])
    #             dflight.to_csv(os.path.join(outpath, "lightfit.csv"), index=False)

    #             # controls output
    #             with VoutBox:
    #                 print("Calc Time: {0:>6.2f} s".format(dt), "saved: " + outpath)
    #         else:
    #             with VoutBox:
    #                 print("Calc Time: {0:>6.2f} s".format(dt))

    #     # summary line
    #     VoutBox = widgets.Output()
    #     VoutBox.layout.height = "40px"
    #     with VoutBox:
    #         print("Summary")

    #     # Left output -> dark
    #     Lout = widgets.Output()
    #     with Lout:  # output device
    #         # print(desc, old, new)
    #         if plt.isinteractive:
    #             plt.ioff()
    #             restart = True
    #         else:
    #             restart = False

    #         dfig, dax = self.plot(dark=True)
    #         dfig.set_figheight(4)
    #         dfig.show()
    #         if restart:
    #             plt.ion()

    #     # Right output -> light
    #     Rout = widgets.Output()
    #     with Rout:  # output device
    #         # print(desc, old, new)
    #         if plt.isinteractive:
    #             plt.ioff()
    #             restart = True
    #         else:
    #             restart = False

    #         lfig, lax = self.plot(dark=False)
    #         lfig.set_figheight(4)
    #         lfig.show()
    #         if restart:
    #             plt.ion()

    #     ToutBox = widgets.HBox([Lout, Rout], layout=junc_layout)

    #     # numerical outputs
    #     out_Voc = widgets.Text(value="Voc", description="Voc", disabled=True, layout=vout_layout)
    #     out_Isc = widgets.Text(value="Isc", description="Isc", disabled=True, layout=vout_layout)
    #     out_FF = widgets.Text(value="FF", description="FF", disabled=True, layout=vout_layout)
    #     out_Pmp = widgets.Text(value="Pmp", description="Pmp", disabled=True, layout=vout_layout)
    #     out_Vmp = widgets.Text(value="Vmp", description="Vmp", disabled=True, layout=vout_layout)
    #     out_Imp = widgets.Text(value="Imp", description="Imp", disabled=True, layout=vout_layout)
    #     # VoutBox = widgets.HBox([in_tit,out_Voc,out_Isc,out_FF,out_Pmp,out_Vmp,out_Imp])

    #     # tandem3T controls
    #     in_tit = widgets.Label(value="Multi2T: ", description="title")
    #     in_name = widgets.Text(value=self.name, description="name", layout=tand_layout, continuous_update=False)
    #     in_Rs2T = widgets.FloatLogSlider(
    #         value=self.Rs2T, base=10, min=-6, max=3, step=0.01, description="Rs2T", layout=tand_layout, readout_format=".2e"
    #     )
    #     in_2Tbut = widgets.Button(description="Recalc", button_style="success", tooltip="slow calculations")
    #     in_savefig = widgets.Button(description="savefig", button_style="success", tooltip="save figures")
    #     tand_dict = {"name": in_name, "Rs2T": in_Rs2T}
    #     # tandout = widgets.interactive_output(self.set, tand_dict)
    #     tand_ui = widgets.HBox([in_tit, in_name, in_Rs2T, in_2Tbut, in_savefig])

    #     in_name.observe(on_2Tchange, names="value")  # update values
    #     in_Rs2T.observe(on_2Tchange, names="value")  # update values

    #     jui = []
    #     # list of junction controls
    #     for i in range(self.njuncs):
    #         jui.append(self.j[i].controls())
    #         kids = jui[i].children
    #         for cntrl in kids:
    #             if type(cntrl) in replot_types:
    #                 cntrl.observe(on_2Treplot, names="value")  # replot
    #     in_Rs2T.observe(on_2Treplot, names="value")  # replot
    #     in_2Tbut.on_click(on_2Treplot)  # replot
    #     in_savefig.on_click(on_2Treplot)  # replot

    #     junc_ui = widgets.HBox(jui)

    #     ui = widgets.VBox([ToutBox, VoutBox, tand_ui, junc_ui])
    #     self.ui = ui
    #     in_2Tbut.click()  # fill in MPP values

    #     # return entire user interface, dark and light graph axes for tweaking
    #     return ui, dax, lax

    # def calcDark(self, hilog=3, pdec=5, timer=False):
    #     # calc dark IV
    #     ts = time()
    #     Jext_list = self.proplist("Jext")  # remember list external photocurrents
    #     self.set(Jext=0.0, JLC=0.0)  # turn lights off but don't update controls
    #     Imax = self.Imaxrev()  # in dark
    #     lolog = math.floor(np.log10(Imax)) - 5
    #     dpnts = (hilog - lolog) * pdec + 1
    #     Ifor = np.logspace(hilog, lolog, num=dpnts)
    #     Irev = np.logspace(lolog, hilog, num=dpnts) * (-1)
    #     Idark = np.concatenate((Ifor, Irev), axis=None)
    #     dpnts = Idark.size  # redefine
    #     Vdark = np.full(dpnts, np.nan, dtype=np.float64)  # Vtotal
    #     Vdarkmid = np.full((dpnts, self.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
    #     for ii, I in enumerate(Idark):
    #         Vdark[ii] = self.V2T(I)  # also sets self.Vmid[i]
    #         for junc in range(self.njuncs):
    #             Vdarkmid[ii, junc] = self.Vmid[junc]
    #     self.set(Jext=Jext_list, JLC=0.0)  # turn lights back on but don't update controls
    #     te = time()
    #     ds = te - ts
    #     if timer:
    #         print(f"dark {ds:2.4f} s")

    #     return Idark, Vdark, Vdarkmid

    # def calcLight(self, pnts=21, Vmin=-0.5, timer=False, fast=False):
    #     # calc light IV
    #     Jext_list = self.proplist("Jext")  # remember list external photocurrents
    #     areas = self.proplist("lightarea")  # list of junction areas
    #     # Imax = max([j*a for j,a in zip(Jext_list,areas)])
    #     Imax = self.Imaxrev()
    #     Eg_list = self.proplist("Eg")  # list of Eg
    #     Egmax = sum(Eg_list)

    #     # ndarray functions
    #     V2Tvect = np.vectorize(self.V2T)
    #     I2Tvect = np.vectorize(self.I2T)

    #     MPP = self.MPP()  # calculate all just once
    #     Voc = MPP["Voc"]

    #     # vertical portion
    #     ts = time()
    #     IxI = np.linspace(-Imax, Imax * 2, pnts)
    #     # VxI = V2Tvect(IxI)
    #     VxI = np.full(pnts, np.nan, dtype=np.float64)  # Vtotal
    #     VmidxI = np.full((pnts, self.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
    #     for ii, I in enumerate(IxI):
    #         VxI[ii] = self.V2T(I)  # also sets self.Vmid[i]
    #         for junc in range(self.njuncs):
    #             VmidxI[ii, junc] = self.Vmid[junc]
    #     te = time()
    #     dsI = te - ts
    #     if timer:
    #         print(f"lightI {dsI:2.4f} s")

    #     if fast:
    #         Vlight = VxI
    #         Ilight = IxI
    #         Vlightmid = VmidxI
    #     else:
    #         # horizonal portion slow part
    #         ts = time()
    #         VxV = np.linspace(Vmin, Voc, pnts)
    #         # IxV = I2Tvect(VxV)
    #         IxV = np.full(pnts, np.nan, dtype=np.float64)  # Vtotal
    #         VmidxV = np.full((pnts, self.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
    #         for ii, V in enumerate(VxV):
    #             IxV[ii] = self.I2T(V)  # also sets self.Vmid[i]
    #             for junc in range(self.njuncs):
    #                 VmidxV[ii, junc] = self.Vmid[junc]
    #         te = time()
    #         dsV = te - ts
    #         if timer:
    #             print(f"lightV {dsV:2.4f} s")
    #         # combine
    #         Vboth = np.concatenate((VxV, VxI), axis=None)
    #         Iboth = np.concatenate((IxV, IxI), axis=None)
    #         Vbothmid = np.concatenate((VmidxV, VmidxI), axis=0)
    #         # sort
    #         p = np.argsort(Vboth)
    #         Vlight = Vboth[p]
    #         Ilight = Iboth[p]
    #         Vlightmid = []
    #         for junc in range(self.njuncs):
    #             Vlightmid.append(np.take_along_axis(Vbothmid[:, junc], p, axis=0))
    #         Vlightmid = np.transpose(np.array(Vlightmid))

    #     Plight = np.array([(-v * j) for v, j in zip(Vlight, Ilight)])
    #     Vlight = np.array(Vlight)
    #     Ilight = np.array(Ilight)

    #     return Vlight, Ilight, Plight, Vlightmid, MPP

    # def plot(self, title="", pplot=False, dark=None, pnts=21, Vmin=-0.5, lolog=-8, hilog=7, pdec=5, size="x-large"):
    #     # plot a light IV of Multi2T

    #     Jext_list = self.proplist("Jext")  # remember list external photocurrents
    #     areas = self.proplist("lightarea")  # list of junction areas
    #     # Imax = max([j*a for j,a in zip(Jext_list,areas)])
    #     Imax = self.Imaxrev()
    #     Eg_list = self.proplist("Eg")  # list of Eg
    #     Egmax = sum(Eg_list)
    #     scale = 1000.0

    #     # ndarray functions
    #     V2Tvect = np.vectorize(self.V2T)
    #     I2Tvect = np.vectorize(self.I2T)

    #     if self.name:
    #         title += self.name

    #     if dark == None:
    #         if math.isclose(self.Isc(), 0.0, abs_tol=1e-6):
    #             dark = True
    #         else:
    #             dark = False

    #     # calc dark IV
    #     Idark, Vdark, Vdarkmid = self.calcDark()

    #     if not dark:
    #         # calc light IV
    #         Vlight, Ilight, Plight, Vlightmid, MPP = self.calcLight()
    #         Voc = MPP["Voc"]

    #     if dark:
    #         # dark plot
    #         dfig, dax = plt.subplots()
    #         dax.set_prop_cycle(color=self.junctioncolors[self.njuncs])
    #         for junc in range(Vdarkmid.shape[1]):  # plot Vdiode of each junction
    #             plns = dax.plot(Vdarkmid[:, junc], Idark, lw=2, label="pjunction" + str(junc))
    #             dax.plot(Vdarkmid[:, junc], -Idark, lw=2, c=plns[0].get_color(), label="njunction" + str(junc))

    #         dax.plot(Vdark, Idark, lw=2, c="black", label="pdark")  # IV curve
    #         dax.plot(Vdark, -Idark, lw=2, c="black", label="ndark")  # IV curve

    #         dax.set_yscale("log")  # logscale
    #         dax.set_autoscaley_on(True)
    #         dax.set_xlim(Vmin, Egmax * 1.1)
    #         dax.grid(color="gray")
    #         dax.set_title(self.name + " Dark", size=size)  # Add a title to the axes.
    #         dax.set_xlabel("Voltage (V)", size=size)  # Add an x-label to the axes.
    #         dax.set_ylabel("Current (A)", size=size)  # Add a y-label to the axes.
    #         # dax.legend()
    #         return dfig, dax

    #     else:
    #         # light plot
    #         lfig, lax = plt.subplots()
    #         lax.set_prop_cycle(color=self.junctioncolors[self.njuncs])
    #         if self.njuncs > 1:
    #             for junc in range(self.njuncs):  # plot Vdiode of each junction
    #                 dlns = lax.plot(Vdarkmid[:, junc], Idark * scale, marker="", ls="--", label="djunction" + str(junc))
    #                 lax.plot(
    #                     Vlightmid[:, junc], Ilight * scale, marker="", ls="-", c=dlns[0].get_color(), label="ljunction" + str(junc)
    #                 )

    #         lax.plot(Vdark, Idark * scale, lw=2, ls="--", c="black", label="dark")  # dark IV curve
    #         lax.plot(Vlight, Ilight * scale, lw=2, c="black", label="light")  # IV curve
    #         lax.plot(self.Vpoints, self.Ipoints * scale, marker="x", ls="", ms=12, c="black", label="points")  # special points
    #         if pplot:  # power curve
    #             laxr = lax.twinx()
    #             laxr.plot(Vlight, Plight * scale, ls="--", c="cyan", zorder=0, label="power")
    #             laxr.set_ylabel("Power (mW)", c="cyan")
    #         lax.set_xlim((Vmin - 0.1), max(min(Egmax, Voc * 1.1), 0.1))
    #         lax.set_ylim(-Imax * 1.5 * scale, Imax * 1.5 * scale)
    #         lax.set_title(self.name + " Light", size=size)  # Add a title to the axes.
    #         lax.set_xlabel("Voltage (V)", size=size)  # Add an x-label to the axes.
    #         lax.set_ylabel("Current (mA)", size=size)  # Add a y-label to the axes.
    #         lax.axvline(0, ls="--", c="gray", label="_vline")
    #         lax.axhline(0, ls="--", c="gray", label="_hline")
    #         # lax.legend()

    #         if False:
    #             # annotate
    #             snote = "T = {0:.1f} C, Rs2T = {1:g} \Omega cm2, A = {2:g} cm2".format(self.TC, self.Rs2T, self.lightarea)
    #             snote += "\nEg = " + str(Eg_list) + " eV"
    #             snote += "\nJext = " + str(Jext_list * 1000) + " mA/cm2"
    #             snote += "\nVoc = {0:.3f} V, Isc = {1:.2f} mA/cm2\nFF = {2:.1f}%, Pmp = {3:.1f} mW".format(
    #                 Voc, MPP["Isc"] * 1000, MPP["FF"] * 100, MPP["Pmp"] * 1000
    #             )

    #             # lax.text(Vmin+0.1,Imax/2,snote,zorder=5,bbox=dict(facecolor='white'))
    #             lax.text(
    #                 0.05,
    #                 0.95,
    #                 snote,
    #                 verticalalignment="top",
    #                 label="mpptext",
    #                 bbox=dict(facecolor="white"),
    #                 transform=lax.transAxes,
    #             )
    #         return lfig, lax
