# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.Tandem3T()   # properties of a 3T tandem including 2 junctions
"""

import copy
import math  # simple math
from time import time
from typing import List

import matplotlib as mpl  # plotting
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
from scipy.interpolate import interp1d
from scipy.optimize import brentq  # root finder

from pvcircuit import junction
from pvcircuit.iv3T import IV3T
from pvcircuit.junction import Junction
from pvcircuit.multi2T import Multi2T


class Tandem3T(object):
    """
    Tandem3T class for optoelectronic model
    """

    update_now = True

    def __init__(self, name="Tandem3T", TC: float = junction.TC_REF, Rz: float = 1, Eg_list: List[float] = [1.8, 1.4], pn: List[float] = [-1, 1], Jext: float = 0.014):
        # user inputs
        # default s-type n-on-p

        # update_now = False #TODO remove

        self.ui = None
        self.Vax = None
        self.Iax = None
        self.Rax = None
        self.Lax = None
        # self.debugout = widgets.Output()  # debug output

        # set attributes
        self.name = name
        self.Rz = Rz
        self.top = Junction(name="top", Eg=Eg_list[0], TC=TC, Jext=Jext, pn=pn[0], beta=0.0)
        self.bot = Junction(name="bot", Eg=Eg_list[1], TC=TC, Jext=Jext, pn=pn[1])

    def copy(self):
        """
        create a copy of a Tandem3T
        need deepcopy() to separate lists, dicts, etc but crashes
        """
        return copy.copy(self)

    def __str__(self):
        """
        format concise description of Tandem3T object
        """

        # attr_list = self.__dict__.keys()
        # attr_dict = self.__dict__.items()
        # print(attr_list)

        strout = self.name + ": <pvcircuit.tandem3T.Tandem3T class>"
        strout += "\nT = {0:.1f} C, Rz= {1:g} Ω cm2, Rt= {2:g} Ω cm2, Rr = {3:g} Ω cm2".format(self.TC, self.Rz, self.top.Rser, self.bot.Rser)
        strout += "\n\n" + str(self.top)
        strout += "\n\n" + str(self.bot)
        return strout

    def __repr__(self):
        return str(self)

    # def update(self):
    #     # update Tandem3T self.ui controls

    #     # for junc in self.j:
    #     for junc in [self.top, self.bot]:  # two junctions
    #         junc.update()

    #     if self.ui:  # Tandem3T user interface has been created
    #         Boxes = self.ui.children
    #         for cntrl in Boxes[2].children:  # Multi2T controls
    #             desc = cntrl.trait_values().get("description", "nodesc")  # does not fail when not present
    #             cval = cntrl.trait_values().get("value", "noval")  # does not fail when not present
    #             if desc in ["name", "Rz"]:  # Multi2T controls to update
    #                 key = desc
    #                 attrval = getattr(self, key)  # current value of attribute
    #                 if cval != attrval:
    #                     # with self.debugout:
    #                     #     print("Tupdate: " + key, attrval)
    #                     cntrl.value = attrval
    #             if desc == "Recalc":
    #                 cntrl.click()  # click button

    def set(self, **kwargs):
        # controlled update of Tandem3T attributes

        # with self.debugout:
        #     print("Tset: ", list(kwargs.keys()))

        # junction kwargs
        jlist = Junction.ATTR.copy() + Junction.ARY_ATTR.copy()
        jkwargs = {key: kwargs.pop(key) for key in jlist if key in kwargs}
        if len(jkwargs) > 0:
            for i, junc in enumerate(["top", "bot"]):
                jikwargs = {}  # empty
                for key, value in jkwargs.items():
                    if key in Junction.ATTR and not np.isscalar(value):
                        # dimension mismatch possibly from self.proplist()
                        jikwargs[key] = value[i]
                    else:
                        jikwargs[key] = value
                # with self.debugout:
                #     print("T2J[" + str(i) + "]: ", jikwargs)
                getattr(self, junc).set(**jikwargs)

        # remaining Multi2T kwargs
        for key, value in kwargs.items():
            if key == "name":
                self.__dict__[key] = str(value)
            # elif key == 'njunc':
            #    self.__dict__[key] = int(value)
            # elif key == 'Vmid':
            #    self.__dict__[key] = np.array(value)
            elif key in ["top", "bot", "update_now"]:
                self.__dict__[key] = value
            elif key in ["Rz"]:
                self.__dict__[key] = np.float64(value)
            # raise error if the key is not in the class attributes
            elif key not in list(self.__dict__.keys()):
                raise ValueError(f"invalid class attribute {key}")

    @property
    def TC(self):
        # largest junction TC
        return max(self.top.TC, self.bot.TC)

    @property
    def totalarea(self):
        # largest junction total area
        return max(self.top.totalarea, self.bot.totalarea)

    @property
    def lightarea(self):
        # largest junction light area
        return max(self.top.lightarea, self.bot.lightarea)

    def V3T(self, iv3T):
        """
        calcuate iv3T.(Vzt,Vrz,Vtr) from iv3T.(Iro,Izo,Ito)
        input class tandem.IV3T object 'iv3T'
        """

        top = self.top  # top Junction
        bot = self.bot  # bot Junction

        inlist = iv3T.Idevlist.copy()
        outlist = iv3T.Vdevlist.copy()
        err = iv3T.init(inlist, outlist)  # initialize output
        if err:
            return err

        # i = 0
        # for index in np.ndindex(iv3T.shape):
        for i, _ in enumerate(np.ndindex(iv3T.shape)):
            # for Ito, Iro, Izo in zip(iv3T.Ito.flat, iv3T.Iro.flat, iv3T.Izo.flat):
            # loop through points

            Ito = iv3T.Ito.flat[i]
            Iro = iv3T.Iro.flat[i]
            Izo = iv3T.Izo.flat[i]
            # loop through points
            kzero = Ito + Iro + Izo
            if not math.isclose(kzero, 0.0, abs_tol=1e-5):
                # print(i, 'Kirchhoff violation', kzero)
                iv3T.Vzt[i] = np.nan
                iv3T.Vrz[i] = np.nan
                iv3T.Vtr[i] = np.nan
                # i += 1
                continue

            # input current densities
            Jt = Ito / top.totalarea
            Jr = Iro / bot.totalarea
            Jz = Izo / self.totalarea

            # top Junction
            top.JLC = 0.0
            Vtmid = top.Vdiode(Jt * top.pn) * top.pn
            Vt = Vtmid + Jt * top.Rser

            # bot Junction
            bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)  # top to bot LC
            if top.totalarea < bot.totalarea:  # distribute LC over total area
                bot.JLC *= top.totalarea / bot.totalarea

            Vrmid = bot.Vdiode(Jr * bot.pn) * bot.pn
            Vr = Vrmid + Jr * bot.Rser

            if top.beta > 0.0:  # repeat if backwards LC
                # top Junction
                top.JLC = top.beta * bot.Jem(Vrmid * bot.pn)  # bot to top LC
                if bot.totalarea < top.totalarea:  # distribute LC over total area
                    top.JLC *= bot.totalarea / top.totalarea
                Vtmid = top.Vdiode(Jt * top.pn) * top.pn
                Vt = Vtmid + Jt * top.Rser

                # bot Junction
                bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)  # top to bot LC
                if top.totalarea < bot.totalarea:  # distribute LC over total area
                    bot.JLC *= top.totalarea / bot.totalarea
                Vrmid = bot.Vdiode(Jr * bot.pn) * bot.pn
                Vr = Vrmid + Jr * bot.Rser

            # extra Z contact
            Vz = Jz * self.Rz

            # items in array = difference of local variable
            iv3T.Vzt.flat[i] = Vz - Vt
            iv3T.Vrz.flat[i] = Vr - Vz
            iv3T.Vtr.flat[i] = Vt - Vr
            # i += 1

        iv3T.Pcalc()  # dev2load defaults

        return 0

    def J3Tabs(self, iv3T):
        """
        calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
        from ABSOLUTE (Vz,Vr,Vt) mapped <- iv3T.(Vzt,Vrz,Vtr)
        input class tandem.IV3T object 'iv3T'
        Operates on absolute voltages, directly using the actual voltages Vz, Vr, and Vt to calculate the currents.
        Calculates the currents directly by applying the voltage directly into the junction models.

        """

        top = self.top  # top Junction
        bot = self.bot  # bot Junction

        inlist = iv3T.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
        outlist = iv3T.Idevlist.copy()
        err = iv3T.init(inlist, outlist)  # initialize output
        if err:
            return err

        if top.notdiode() and top.Rser == 0:
            return 1
        if bot.notdiode() and bot.Rser == 0:
            return 1

        # i = 0
        # for index in np.ndindex(iv3T.shape):
        for i, _ in enumerate(np.ndindex(iv3T.shape)):
            # for Vz, Vr, Vt in zip(iv3T.Vzt.flat, iv3T.Vrz.flat, iv3T.Vtr.flat):
            # loop through points
            # ABSOLUTE (Vz,Vr,Vt) mapped <- iv3T.(Vzt,Vrz,Vtr)

            Vz = iv3T.Vzt.flat[i]
            Vr = iv3T.Vrz.flat[i]
            Vt = iv3T.Vtr.flat[i]
            # top Junction
            top.JLC = 0.0
            if top.notdiode():  # top resistor only
                Vtmid = 0.0
                Jt = Vt / top.Rser
            else:  # top diode
                Vtmid = top.Vmid(Vt * top.pn) * top.pn
                Jt = -top.Jparallel(Vtmid * top.pn, top.Jphoto) * top.pn

            # bot Junction
            bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)  # top to bot LC
            if top.totalarea < bot.totalarea:  # distribute LC over total area
                bot.JLC *= top.totalarea / bot.totalarea

            if bot.notdiode():  # bot resistor only
                Vrmid = 0.0
                Jr = Vr / bot.Rser
            else:  # bot diode
                Vrmid = bot.Vmid(Vr * bot.pn) * bot.pn
                Jr = -bot.Jparallel(Vrmid * bot.pn, bot.Jphoto) * bot.pn

            if top.beta > 0.0:  # repeat if backwards LC
                # top Junction
                top.JLC = top.beta * bot.Jem(Vrmid * bot.pn)  # bot to top LC
                if bot.totalarea < top.totalarea:  # distribute LC over total area
                    top.JLC *= bot.totalarea / top.totalarea

                if top.notdiode():  # top resistor only
                    Vtmid = 0.0
                    Jt = Vt / top.Rser
                else:  # top diode
                    Vtmid = top.Vmid(Vt * top.pn) * top.pn
                    Jt = -top.Jparallel(Vtmid * top.pn, top.Jphoto) * top.pn

                # bot Junction
                bot.JLC = bot.beta * top.Jem(Vtmid * top.pn)  # top to bot LC
                if top.totalarea < bot.totalarea:  # distribute LC over total area
                    bot.JLC *= top.totalarea / bot.totalarea

                if bot.notdiode():  # bot resistor only
                    Vrmid = 0.0
                    Jr = Vr / bot.Rser
                else:  # bot diode
                    Vrmid = bot.Vmid(Vr * bot.pn) * bot.pn
                    Jr = -bot.Jparallel(Vrmid * bot.pn, bot.Jphoto) * bot.pn

            # extra Z contact
            if self.Rz == 0.0:
                Jz = (-Jt * top.totalarea - Jr * bot.totalarea) / self.totalarea  # determine from kirchhoff
            else:
                Jz = Vz / self.Rz  # determine from Rz

            # output (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
            iv3T.Iro.flat[i] = Jr
            iv3T.Izo.flat[i] = Jz
            iv3T.Ito.flat[i] = Jt
            # i += 1

        return 0

    def _dI(self, Vz, Vzt, Vrz, temp3T):
        """
        return dI = Iro + Izo + Ito
        function solved for dI(Vz)=0 in I3rel
        input Vzt, Vrz, temp3T <IV3T class> container for calculation
        """
        top = self.top  # top Junction
        bot = self.bot  # bot Junction

        Vt = Vz - Vzt
        Vr = Vrz + Vz

        temp3T.set(Vzt=Vz, Vrz=Vr, Vtr=Vt)

        self.J3Tabs(temp3T)  # calcuate (Jro,Jzo,Jto) from (Vz,Vr,Vt)

        # (Jro,Jzo,Jto)  -> (Iro,Izo,Ito)
        Jro = temp3T.Iro[0]
        Jzo = temp3T.Izo[0]
        Jto = temp3T.Ito[0]

        Iro = Jro * bot.totalarea
        Izo = Jzo * self.totalarea
        Ito = Jto * top.totalarea

        return Iro + Izo + Ito

    def I3Trel(self, iv3T):
        """
        calcuate (Jro,Jzo,Jto) mapped -> iv3T.(Iro,Izo,Ito)
        from RELATIVE iv3T.(Vzt,Vrz,Vtr) ignoring Vtr
        input class tandem.IV3T object 'iv3T'

        operates on relative voltages, taking the input voltages Vzt and Vrz, and calculates currents while ignoring Vtr.
        Iteratively adjusts these voltages using resistance models to calculate the current densities. Uses iterative method to find the
        correct value of Vz that satisfies Kirchhoff’s current law across the junctions, adjusting the voltage drop for the z-terminal and checking if the current balances (via _dI function).
        """

        top = self.top  # top Junction
        bot = self.bot  # bot Junction

        inlist = iv3T.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
        outlist = iv3T.Idevlist.copy()
        err = iv3T.init(inlist, outlist)  # initialize output
        if err:
            return err

        temp3T = IV3T(name="temp3T", shape=1, meastype=iv3T.meastype, area=iv3T.area)

        # remember resistances
        Rz = self.Rz
        Rt = top.Rser
        Rr = bot.Rser

        # i = 0
        # for index in np.ndindex(iv3T.shape):
        for i, _ in enumerate(np.ndindex(iv3T.shape)):
            # for Vzt, Vrz in zip(iv3T.Vzt.flat, iv3T.Vrz.flat):
            # loop through points

            Vzt = iv3T.Vzt.flat[i]
            Vrz = iv3T.Vrz.flat[i]
            # initial guess Rz=0 -> Vz=0
            self.Rz = 0.0
            top.Rser = Rt + Rz
            bot.Rser = Rr + Rz
            Vz = 0.0
            Vt = Vz - Vzt
            Vr = Vrz + Vz
            temp3T.set(Vzt=Vz, Vrz=Vr, Vtr=Vt)
            self.J3Tabs(temp3T)

            Jro = temp3T.Iro[0]
            Jzo = temp3T.Izo[0]
            Jto = temp3T.Ito[0]
            Vzmax = Jzo * Rz

            # reset resistance
            self.Rz = Rz
            top.Rser = Rt
            bot.Rser = Rr

            # iterate with varying Vz
            # bracket
            Vmin = 0.5
            if math.isfinite(Vzmax) and (abs(Vzmax) > junction.VTOL):
                Vlo = min(0, 1.2 * Vzmax)
                Vhi = max(0, 1.2 * Vzmax)
            else:
                Vlo = -Vmin
                Vhi = Vmin
            dIlo = self._dI(Vlo, Vzt, Vrz, temp3T)
            dIhi = self._dI(Vhi, Vzt, Vrz, temp3T)

            count = 0
            while dIlo * dIhi > 0.0:
                # print('I3Trel not bracket', i, count, Vlo, Vhi, dIlo, dIhi)
                Vlo -= 0.1
                Vhi += 0.1
                dIlo = self._dI(Vlo, Vzt, Vrz, temp3T)
                dIhi = self._dI(Vhi, Vzt, Vrz, temp3T)
                if count > 2:
                    break
                count += 1

            if dIlo * dIhi > 0.0:
                # print('I3Trel not bracket', i, count, Vlo, Vhi, dIlo, dIhi)
                # i += 1
                continue

            if Rz > 0.0:
                try:  # find Vz that satisfies dJ(Vz) = Jro+Jzo+Jto = 0
                    Vz = brentq(
                        self._dI,
                        Vlo,
                        Vhi,
                        args=(Vzt, Vrz, temp3T),
                        xtol=junction.VTOL,
                        rtol=junction.EPSREL,
                        maxiter=junction.MAXITER,
                        full_output=False,
                        disp=True,
                    )
                    # dI0 = self._dI(Vz,Vzt,Vrz,temp3T)

                    Jro = temp3T.Iro[0]
                    Jzo = temp3T.Izo[0]
                    Jto = temp3T.Ito[0]

                except ValueError:
                    Jro = np.nan
                    Jzo = np.nan
                    Jto = np.nan

            # output
            iv3T.Iro.flat[i] = Jro * bot.totalarea
            iv3T.Izo.flat[i] = Jzo * self.totalarea
            iv3T.Ito.flat[i] = Jto * top.totalarea
            # i += 1

        iv3T.kirchhoff(["Vzt", "Vrz"])  # Vtr not used so make sure consistent
        iv3T.kirchhoff(iv3T.Idevlist.copy())  # check for bad results
        iv3T.Pcalc()  # dev2load defaults

        return 0

    def Voc3(self, meastype="CZ"):
        """
        triple Voc of 3T tandem
        (Vzt, Vrz, Vtr) of (Iro = 0, Izo = 0, Ito = 0)
        """

        temp3T = IV3T(name="Voc3", shape=1, meastype=meastype, area=self.lightarea)
        temp3T.set(Iro=0.0, Izo=0.0, Ito=0.0)
        self.V3T(temp3T)

        # return (temp3T.Vzt[0], temp3T.Vrz[0], temp3T.Vtr[0])
        return temp3T

    def Isc3(self, meastype="CZ"):
        """
        triple Isc of 3T tandem
        (Iro, Izo, Ito ) of (Vzt = 0, Vrz = 0, Vtr = 0)
        """

        temp3T = IV3T(name="Isc3", shape=1, meastype=meastype, area=self.lightarea)
        temp3T.set(Vzt=0.0, Vrz=0.0, Vtr=0.0)
        self.I3Trel(temp3T)

        # return (temp3T.Iro[0], temp3T.Izo[0], temp3T.Ito[0])
        return temp3T

    def VM(self, bot, top, pnts=11):
        """
        create VM constrained line for tandem3T
        """
        if bot == 0 or top == 0:
            return self.CM(pnts=pnts)

        sbot = str(bot)
        stop = str(top)
        name = "VM" + sbot + stop
        meastype = "CZ"
        Voc3 = self.Voc3(meastype)  # find triple Voc point
        ln = IV3T(name=name, meastype=meastype, area=self.lightarea)
        lnout = IV3T(name=name, shape=0, meastype=meastype, area=self.lightarea)
        sign = np.sign(Voc3.Vzt[0] / Voc3.Vrz[0])
        x0 = 0
        if abs(Voc3.Vzt[0]) * top > abs(Voc3.Vrz[0]) * bot:
            yconstraint = "x * " + stop + " / " + sbot + " * (" + str(sign) + ")"
            xkey = "Vzt"
            x1 = Voc3.Vzt[0]
            ykey = "Vrz"
        else:
            yconstraint = "x * " + sbot + " / " + stop + "* (" + str(sign) + ")"
            xkey = "Vrz"
            x1 = Voc3.Vrz[0]
            ykey = "Vzt"

        for i in range(4):  # focus on MPP
            ln.line(xkey, x0, x1, pnts, ykey, yconstraint)
            ln.name = name + "_" + str(i)
            self.I3Trel(ln)
            lnout.append(ln)
            xln = getattr(ln, xkey)
            nmax = np.argmax(ln.Ptot)
            x0 = xln[max(0, (nmax - 1))]
            x1 = xln[min((nmax + 1), (pnts - 1))]

        lnout.sort(xkey)  # sort for nice line
        # find useless values
        ind = []
        for i, Ptot in enumerate(lnout.Ptot.flat):
            if math.isfinite(Ptot):
                if Ptot / lnout.area < 0.0 or np.isnan(Ptot / lnout.area):
                    ind.append(i)  # negative
            else:  # not finite
                ind.append(i)
        lnout.delete(ind)  # delete extraneous points from lnout
        if lnout.shape[0] > 0:
            MPP = lnout.MPP(name)  # single MPP point in IV3T space
        else:
            lnout = IV3T(name="bogus", meastype=meastype, shape=1, area=self.lightarea)
            lnout.Ptot[0] = 0
            MPP = lnout.MPP(name)
            return lnout, MPP

        # TODO needs externalization in PlottingWithControls
        # plot if possible
        pltargs = {"lw": 0, "ms": 7, "mew": 1, "mec": "black", "marker": "o", "zorder": 5}
        pltargs["label"] = name
        if self.Vax:
            ln = lnout.addpoints(self.Vax, "VA", "VB")  # let cycler choose color
            c = ln.get_color()
            MPP.addpoints(self.Vax, "VA", "VB", c=c, **pltargs)
            # self.Vax.legend()
        if self.Iax:
            lnout.addpoints(self.Iax, "IA", "IB", c=c)
            MPP.addpoints(self.Iax, "IA", "IB", c=c, **pltargs)
            # self.Iax.legend()

        return lnout, MPP

    def CM(self, pnts=11):
        """
        create CM constrained line for tandem3T
        """
        name = "CM"
        meastype = "CZ"
        Isc3 = self.Isc3(meastype)  # find triple Voc point
        ln = IV3T(name=name, meastype=meastype, area=self.lightarea)
        lnout = IV3T(name=name, shape=0, meastype=meastype, area=self.lightarea)
        x0 = 0
        # sign = -1   #np.sign(Isc3.Iro[0]/Isc3.Ito[0])
        # yconstraint = 'x'+ ' * (' + str(sign) + ')'
        yconstraint = "-x"
        if abs(Isc3.Iro[0]) < abs(Isc3.Ito[0]):
            xkey = "Ito"
            x1 = Isc3.Ito[0]
            ykey = "Iro"
        else:
            xkey = "Iro"
            x1 = Isc3.Iro[0]
            ykey = "Ito"

        for i in range(4):  # focus on MPP
            ln.line(xkey, x0, x1, pnts, ykey, yconstraint)
            ln.name = name + "_" + str(i)
            self.V3T(ln)
            lnout.append(ln)
            xln = getattr(ln, xkey)
            nmax = np.argmax(ln.Ptot)
            x0 = xln[max(0, (nmax - 1))]
            x1 = xln[min((nmax + 1), (pnts - 1))]

        lnout.sort(xkey)  # sort for nice line
        # find useless values
        ind = []
        for i, Ptot in enumerate(lnout.Ptot.flat):
            if math.isfinite(Ptot):
                if Ptot / lnout.area < 0.0 or np.isnan(Ptot / lnout.area):
                    ind.append(i)  # negative
            else:  # not finite
                ind.append(i)
        lnout.delete(ind)  # delete extraneous points from lnout
        if lnout.shape[0] > 0:
            MPP = lnout.MPP(name)  # single MPP point in IV3T space
        else:
            lnout = IV3T(name="bogus", meastype=meastype, shape=1, area=self.lightarea)
            lnout.Ptot[0] = 0
            MPP = lnout.MPP(name)
            return lnout, MPP

        # TODO needs externalization in PlottingWithControls
        # plot if possible
        pltargs = {"lw": 0, "ms": 7, "mew": 1, "mec": "black", "marker": "o", "zorder": 5}
        pltargs["label"] = name
        if self.Vax:
            ln = lnout.addpoints(self.Vax, "VA", "VB", label="ln_" + name)  # let cycler choose color
            c = ln.get_color()
            MPP.addpoints(self.Vax, "VA", "VB", c=c, **pltargs)
        if self.Iax:
            lnout.addpoints(self.Iax, "IA", "IB", c=c, label="ln_" + name)
            MPP.addpoints(self.Iax, "IA", "IB", c=c, **pltargs)

        return lnout, MPP

    def MPP(self, pnts=31, VorI="I", less=2.0, bplot=False):
        """
        iteratively find MPP from lines
        as experimentally done
        varying I is faster than varying V
        but initial guess is not as good
        'less' must be > 1.0
        if FF is really bad, may need larger 'less'
        bplot for debugging information
        """

        ts = time()
        meastype = "CZ"
        Voc3 = self.Voc3(meastype)
        Isc3 = self.Isc3(meastype)
        Isct = Isc3.Ito[0]
        Iscr = Isc3.Iro[0]
        if any(np.isnan([Isct, Iscr])):
            VorI = "V"
        pnt = self.top.pn
        pnr = self.bot.pn
        tol = 1e-5
        # initial guesses
        Vmpr = 0.0
        Impr = Iscr
        Impt = Isct
        # create IV3T classes for top and rear
        lnt = IV3T(name="lnMPPt", meastype=meastype, area=self.lightarea)
        lnr = IV3T(name="lnMPPr", meastype=meastype, area=self.lightarea)

        if bplot:
            _, ax = plt.subplots()
            ax.axhline(0, color="gray")
            ax.axvline(0, color="gray")
            ax.set_title(self.name + " MPP calc.")
            ax.plot(Voc3.Vrz, 0, marker="o")
            ax.plot(Voc3.Vzt, 0, marker="o")
            ax.plot(0, Isc3.Iro * pnr, marker="o")
            ax.plot(0, Isc3.Ito * pnt, marker="o")

        Pmpo = 0.0
        for i in range(5):
            # iterate
            if VorI == "V":
                lnt.line("Vzt", 0, Voc3.Vzt[0], pnts, "Vrz", str(Vmpr))
                self.I3Trel(lnt)
            else:
                lnt.line("Ito", Isct, Isct / less, pnts, "Iro", str(Impr))
                self.V3T(lnt)
            aPt = getattr(lnt, "Ptot")
            aVzt = getattr(lnt, "Vzt")
            aIto = getattr(lnt, "Ito")
            nt = np.argmax(aPt)
            Vmpt = aVzt[nt]
            Impt = aIto[nt]
            Pmpt = aPt[nt]
            if bplot:
                ax.plot(aVzt, aIto * pnt, marker=".")
                ax.plot(Vmpt, Impt * pnt, marker="o")
                print(i, "T", Vmpt, Impt, Pmpt)

            if VorI == "V":
                lnr.line("Vrz", 0, Voc3.Vrz[0], pnts, "Vzt", str(Vmpt))
                self.I3Trel(lnr)
            else:
                lnr.line("Iro", Iscr, Iscr / less, pnts, "Ito", str(Impt))
                self.V3T(lnr)
            aPr = getattr(lnr, "Ptot")
            aVrz = getattr(lnr, "Vrz")
            aIro = getattr(lnr, "Iro")
            nr = np.argmax(aPr)
            Vmpr = aVrz[nr]
            Impr = aIro[nr]
            Pmpr = aPr[nr]
            if bplot:
                ax.plot(aVrz, aIro * pnr, marker=".")
                ax.plot(Vmpr, Impr * pnr, marker="o")
                print(i, "R", Vmpr, Impr, Pmpr)

            if (Pmpr - Pmpo) / Pmpr < tol:
                break
            Pmpo = Pmpr
            # VorI = "I"  # switch to 'I' after first iteration

        # create one line solution
        pt = IV3T(name="MPP", meastype=meastype, shape=1, area=self.lightarea)
        pt.Vzt[0] = Vmpt
        pt.Vrz[0] = Vmpr
        pt.kirchhoff(["Vzt", "Vrz"])
        self.I3Trel(pt)

        te = time()
        dt = te - ts
        if bplot:
            print("MPP: {0:d}pnts , {1:2.4f} s".format(pnts, dt))

        return pt

    def VI0(self, VIname, meastype="CZ"):
        """
        solve for mixed (V=0, I=0) zero power points
        separate diodes for quick single point solution
        """
        ts = time()
        pt = IV3T(name=VIname, meastype=meastype, shape=1, area=self.lightarea)
        top = self.top  # pointer
        bot = self.bot  # pointer

        if VIname == "VztIro":
            pt.Iro[0] = 0.0
            pt.Vzt[0] = 0.0
            # top
            tmptop = self.top.copy()  # copy for temporary calculations
            tmptop.set(name="tmptop", Rser=top.Rser + self.Rz / self.totalarea * top.totalarea, JLC=0.0)  # correct Rz by area ratio
            Vtmid = tmptop.Vmid(pt.Vzt[0] * top.pn) * top.pn
            pt.Ito[0] = -tmptop.Jparallel(Vtmid * top.pn, tmptop.Jphoto) * top.pn * top.totalarea
            pt.Izo[0] = -pt.Ito[0]  # from Kirchhoff
            self.V3T(pt)  # calc Vs from Is
            # bot to top LC?

        elif VIname == "VrzIto":
            pt.Ito[0] = 0.0
            pt.Vrz[0] = 0.0
            # top
            tmptop = self.top.copy()  # copy for temporary calculations
            tmptop.set(name="tmptop")
            tmptop.JLC = 0.0
            Vtmid = tmptop.Vdiode(pt.Ito[0] / top.totalarea * top.pn) * top.pn
            # bot
            tmpbot = self.bot.copy()
            tmpbot.set(name="tmpbot", Rser=bot.Rser + self.Rz / self.totalarea * bot.totalarea)  # correct Rz by area ratio
            tmpbot.JLC = bot.beta * tmptop.Jem(Vtmid * top.pn)  # top to bot LC
            if top.totalarea < bot.totalarea:  # distribute LC over total area
                tmpbot.JLC *= top.totalarea / bot.totalarea

            Vrmid = tmpbot.Vmid(pt.Vrz[0] * bot.pn) * bot.pn
            pt.Iro[0] = -tmpbot.Jparallel(Vrmid * bot.pn, tmpbot.Jphoto) * bot.pn * bot.totalarea
            pt.Izo[0] = -pt.Iro[0]  # Kirchhoff
            self.V3T(pt)  # calc Vs from Is
            # bot to top LC?

        elif VIname == "VtrIzo":
            ### does not work for r-type since Multi2T ignore pn #####
            pt.Izo[0] = 0.0  # series-connected 2T
            pt.Vtr[0] = 0.0
            if self.top.pn == self.bot.pn:  # r-type
                # pt=self.VIpoint('Izo','Ito','Vtr',meastype=meastype, bplot=True)
                # pt=self.VIpoint('Vtr','Vzt','Izo',meastype=meastype, bplot=True)
                # pt=self.VIpoint('Vtr','Vrz','Izo',meastype=meastype)
                pt.nanpnt(0)
            else:  # s-type
                dev2T = Multi2T.from_3T(self)
                pt.Iro[0] = dev2T.I2T(pt.Vtr[0])
                pt.Ito[0] = -pt.Iro[0]
            self.V3T(pt)  # calc Vs from Is

        else:
            pt.nanpnt(0)

        te = time()
        # dt = te - ts
        # print('VI0: ' + pt.name + ' {0:2.4f} s'.format(dt))

        return pt

    def VIpoint(self, zerokey, varykey, crosskey, meastype="CZ", pnts=11, bplot=False):
        """
        absolete! use VI0 for point only
        solve for mixed (V=0, I=0) zero power points
        using a constrained line
        """

        ts = time()
        if varykey[0] == "V":
            Voc3 = self.Voc3(meastype)
            x0 = getattr(Voc3, varykey)[0]
            dx = 1
            # abs_tol = 1e-5  # .01 mA #TODO remove
        else:
            Isc3 = self.Isc3(meastype)
            x0 = getattr(Isc3, varykey)[0]
            # dx = abs(getattr(Isc3, 'Izo')[0])
            dx = 0.1
            # abs_tol = 1e-3  # 1 mV #TODO remove

        ln = IV3T(name="ln" + zerokey + "_0", meastype=meastype, area=self.lightarea)

        growth = pnts - 1.0

        if bplot:
            _, ax = plt.subplots()
            ax.axhline(0, color="gray")
            ax.set_title("VIpoint: " + zerokey + crosskey + "  " + zerokey + "=0")
            ax.set_xlabel(varykey)
            ax.set_ylabel(crosskey)
            ax.plot(x0, 0, marker="o", fillstyle="none", ms=8, color="black")

        for _ in range(4):
            if bplot:
                print("x0 =", x0)
            ln.line(varykey, x0 - dx, x0 + dx, pnts, zerokey, "0")
            if varykey[0] == "V":
                self.I3Trel(ln)
            else:
                self.V3T(ln)
            xx = getattr(ln, varykey)
            yy = getattr(ln, crosskey)

            try:
                fn_interp = interp1d(yy, xx)  # scipy interpolate function
                xguess = fn_interp(0.0)
            except ValueError:
                xguess = np.interp(0.0, yy, xx)  # y=zero crossing
            if not np.isnan(xguess):
                x0 = xguess
            dx /= growth
            if bplot:
                ax.plot(xx, yy, marker=".")

        # create one line solution
        pt = IV3T(name=zerokey + crosskey, meastype=meastype, shape=1, area=self.lightarea)
        xp = getattr(pt, varykey)
        xp[0] = x0
        zp = getattr(pt, zerokey)
        zp[0] = 0.0
        pt.kirchhoff([zerokey, varykey])
        if varykey[0] == "V":
            self.I3Trel(pt)
        else:
            self.V3T(pt)

        yp = getattr(pt, crosskey)
        if not math.isclose(yp[0], 0.0, abs_tol=1e-3):  # test if it worked
            # pt.nanpnt(0)
            pass

        te = time()
        dt = te - ts
        print("VIpoint: " + pt.name + " {0:d}pnts , {1:2.4f} s".format(pnts, dt))

        return pt

    def specialpoints(self, meastype="CZ", bplot=False, fast=False):
        """
        compile all the special zero power points
        and fast MPP estimate
        """

        sp = self.Voc3(meastype=meastype)  # Voc3 = 0
        sp.names[0] = "Voc3"
        sp.name = self.name + " SpecialPoints"
        sp.append(self.Isc3(meastype=meastype))  # Isc3 = 1

        # (Ito = 0, Vrz = 0)
        # sp.append(self.VIpoint('Ito','Iro','Vrz', meastype=meastype, bplot=bplot)) # Ito = 0, x = Iro, y = Vrz
        # sp.append(self.VIpoint('Vrz','Vtr','Ito', meastype=meastype, bplot=bplot)) # Vrz = 0, x = Vzt, y = Ito
        # sp.append(self.VIpoint('Vrz','Vzt','Ito', meastype=meastype, bplot=bplot)) # Vrz = 0, x = Vzt, y = Ito
        sp.append(self.VI0("VrzIto", meastype=meastype))

        # (Iro = 0, Vzt = 0)
        # sp.append(self.VIpoint('Iro','Ito','Vzt', meastype=meastype)) # Iro = 0, x = Ito, y = Vzt
        # sp.append(self.VIpoint('Vzt','Vrz','Iro', meastype=meastype, bplot=bplot)) # Vzt = 0, x = Vrz , y = Iro
        # sp.append(self.VIpoint('Vzt','Vtr','Iro', meastype=meastype, bplot=bplot)) # Vzt = 0, x = Vrz , y = Iro
        sp.append(self.VI0("VztIro", meastype=meastype))

        # (Izo = 0, Vtr =0)
        # sp.append(self.VIpoint('Izo','Ito','Vtr', meastype=meastype, bplot=bplot)) # Izo = 0, x = Ito, y = Vtr
        # sp.append(self.VIpoint('Vtr','Vzt','Izo', meastype=meastype, bplot=bplot)) # Vtr = 0, x = Vzt, y = Izo
        # sp.append(self.VIpoint('Vtr','Vrz','Izo', meastype=meastype, bplot=bplot)) # Vtr = 0, x = Vzt, y = Izo
        sp.append(self.VI0("VtrIzo", meastype=meastype))

        # MPP
        if not fast:
            sp.append(self.MPP(bplot=bplot))

        return sp

    # def controls(
    #     self,
    #     Vdata3T=None,
    #     Idata3T=None,
    #     darkData3T=None,
    #     hex=False,
    #     meastype="CZ",
    #     size="x-large",
    #     Iargs={"xkey": "IA", "ykey": "IB", "zkey": "Ptot", "density": True},
    #     Vargs={"xkey": "VA", "ykey": "VB", "zkey": "Ptot", "density": True},
    # ):
    #     """
    #     use interactive_output for GUI in IPython
    #     Iargs = {'xkey':'VA', 'ykey':'VB', 'zkey':'IB', 'log':True}
    #     Vargs = {'xkey':'VA', 'ykey':'VB', 'zkey':'IA', 'log':True}

    #     """
    #     tand_layout = widgets.Layout(width="300px", height="40px")
    #     vout_layout = widgets.Layout(width="180px", height="40px")
    #     grid_layout = widgets.Layout(grid_template_columns="repeat(2, 180px)", grid_template_rows="repeat(3, 30px)", height="100px")
    #     junc_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")
    #     replot_types = [
    #         widgets.widgets.widget_float.BoundedFloatText,
    #         widgets.widgets.widget_int.BoundedIntText,
    #         widgets.widgets.widget_int.IntSlider,
    #         widgets.widgets.widget_float.FloatSlider,
    #         widgets.widgets.widget_float.FloatLogSlider,
    #     ]
    #     scale = 1000.0
    #     pnts = 71
    #     pltargs = {
    #         "lw": 0,
    #         "ms": 7,
    #         "mew": 1,
    #         "mec": "black",
    #         "mfc": "white",
    #         "marker": "o",
    #         "c": "red",
    #         "label": "fitsp",
    #         "zorder": 5,
    #     }
    #     dsun = [1e-9, 1e-10]  # dark suns
    #     dpnts = 21
    #     dlo = -14
    #     dhi = 2

    #     def on_3Tchange(change):
    #         # function for changing values
    #         old = change["old"]  # old value
    #         new = change["new"]  # new value
    #         owner = change["owner"]  # control
    #         value = owner.value
    #         desc = owner.description
    #         with self.debugout:
    #             print("Tcontrol: " + desc + "->", value)
    #         self.set(**{desc: value})

    #     def on_3Treplot(change):
    #         # change info
    #         fast = False
    #         Vcalc = False
    #         Icalc = False
    #         Dcalc = False
    #         if type(change) is widgets.widgets.widget_button.Button:
    #             owner = change
    #             desc = owner.description
    #         else:  # other controls
    #             owner = change["owner"]  # control
    #         desc = owner.description
    #         if desc == "All":
    #             Vcalc = True
    #             Icalc = True
    #             Dcalc = True
    #         elif desc == "P(I)":
    #             Vcalc = True
    #         elif desc == "P(V)":
    #             Icalc = True
    #         elif desc == "dark":
    #             Dcalc = True
    #         elif desc == "MPPcalc":
    #             fast = False
    #         elif desc == "savefig":
    #             fast = True
    #         else:
    #             fast = True

    #         with self.debugout:
    #             print(desc, self)
    #         # recalculate
    #         ts = time()
    #         fitsp = self.specialpoints(meastype=meastype, fast=fast)
    #         if Iargs["density"] == True:
    #             fscale = 1000.0 / fitsp.area
    #         else:
    #             fscale = 1000.0
    #         # summary line
    #         fmtstr = "Fit:  (Vzt = {0:>5.3f}, Vrz = {1:>5.3f}, Vtr = {2:>5.3f} V),   "
    #         if Iargs["density"] == True:
    #             fmtstr += "(Jro = {3:>5.2f}, Jzo = {4:>5.2f}, Jto = {5:>5.2f} mA/mA)"
    #         else:
    #             fmtstr += "(Iro = {3:>5.2f}, Izo = {4:>5.2f}, Ito = {5:>5.2f} mA)"

    #         VoutBox.clear_output()
    #         if VdataMPP:
    #             outstr = (fmtstr + ",   Pmp = {6:>5.2f} mW/cm2").format(
    #                 VdataMPP.Vzt[0],
    #                 VdataMPP.Vrz[0],
    #                 VdataMPP.Vtr[0],
    #                 VdataMPP.Iro[0] * fscale,
    #                 VdataMPP.Izo[0] * fscale,
    #                 VdataMPP.Ito[0] * fscale,
    #                 VdataMPP.Ptot[0] * fscale,
    #             )
    #             with VoutBox:
    #                 print(outstr.replace("Fit:", "VData:"))

    #         elif IdataMPP:
    #             outstr = (fmtstr + ",   Pmp = {6:>5.2f} mW/cm2").format(
    #                 IdataMPP.Vzt[0],
    #                 IdataMPP.Vrz[0],
    #                 IdataMPP.Vtr[0],
    #                 IdataMPP.Iro[0] * fscale,
    #                 IdataMPP.Izo[0] * fscale,
    #                 IdataMPP.Ito[0] * fscale,
    #                 IdataMPP.Ptot[0] * fscale,
    #             )
    #             with VoutBox:
    #                 print(outstr.replace("Fit:", "IData:"))

    #         if "MPP" in fitsp.names:  # not fast
    #             ii = fitsp.names.index("MPP")  # index of MPP from sp
    #             fmtstr += ",   Pmp = {6:>5.2f} mW/cm2"
    #             outstr = fmtstr.format(
    #                 fitsp.Vzt[0],
    #                 fitsp.Vrz[0],
    #                 fitsp.Vtr[0],
    #                 fitsp.Iro[1] * fscale,
    #                 fitsp.Izo[1] * fscale,
    #                 fitsp.Ito[1] * fscale,
    #                 fitsp.Ptot[ii] * fscale,
    #             )
    #             with VoutBox:
    #                 print(outstr.replace("fit", "data"))
    #         else:
    #             outstr = fmtstr.format(
    #                 fitsp.Vzt[0], fitsp.Vrz[0], fitsp.Vtr[0], fitsp.Iro[1] * fscale, fitsp.Izo[1] * fscale, fitsp.Ito[1] * fscale
    #             )
    #             with VoutBox:
    #                 print(outstr.replace("fit", "data"))

    #         tmp = time()

    #         # with Rout: # right output device -> P(I) typically
    #         # replot: Iax with Rout
    #         lines = Iax.get_lines()
    #         for line in lines:
    #             linelabel = line.get_label()
    #             if linelabel == "fitsp":
    #                 xp = getattr(fitsp, Iargs["xkey"]) * Iscale
    #                 yp = getattr(fitsp, Iargs["ykey"]) * Iscale
    #                 line.set_data(xp, yp)
    #         if Vcalc:
    #             if RVorI == "I":
    #                 self.V3T(Ifit3T)
    #             else:
    #                 self.I3Trel(Ifit3T)  # slow
    #             # self.V3T(Ifit3T)
    #             for i, obj in enumerate(Iobjs):
    #                 if type(obj) is mpl.contour.QuadContourSet:  # contours
    #                     if obj.colors == "red":  # identify fit contour
    #                         fitcont = Iobjs.pop(i)  # remove it
    #                         for coll in fitcont.collections:
    #                             if coll in Iax.collections:
    #                                 Iax.collections.remove(coll)  # remove old contour lines from plot
    #                         for text in fitcont.labelTexts:
    #                             if text in Iax.texts:
    #                                 Iax.texts.remove(text)  # remove old contour labels from plot
    #                         break

    #             with Rout:
    #                 Ifit3T.plot(inplot=(Iax, Iobjs), cmap=None, ccont="red", **Iargs)  # replot fit contour

    #         tI = time()

    #         # with Lout: # left output device -> P(V) typically
    #         # replot: Vax with Lout
    #         lines = Vax.get_lines()
    #         for line in lines:
    #             linelabel = line.get_label()
    #             if linelabel == "fitsp":
    #                 xp = getattr(fitsp, Vargs["xkey"]) * Vscale
    #                 yp = getattr(fitsp, Vargs["ykey"]) * Vscale
    #                 line.set_data(xp, yp)
    #         if Icalc:
    #             if LVorI == "I":
    #                 self.V3T(Vfit3T)
    #             else:
    #                 self.I3Trel(Vfit3T)  # slow
    #             # self.I3Trel(Vfit3T)    #slow
    #             for i, obj in enumerate(Vobjs):
    #                 if type(obj) is mpl.contour.QuadContourSet:  # contours
    #                     if obj.colors == "red":  # identify fit contour
    #                         fitcont = Vobjs.pop(i)  # remove it
    #                         for coll in fitcont.collections:
    #                             if coll in Vax.collections:
    #                                 Vax.collections.remove(coll)  # remove old contour lines from plot
    #                         for text in fitcont.labelTexts:
    #                             if text in Vax.texts:
    #                                 Vax.texts.remove(text)  # remove old contour labels from plot
    #                         break

    #             with Lout:
    #                 Vfit3T.plot(inplot=(Vax, Vobjs), cmap=None, ccont="red", **Vargs)  # replot fit contour

    #         tV = time()

    #         # replot: dark Lax, Rax
    #         if darkFit3T:  # add new dark fit
    #             # dark plots
    #             Jtop = self.top.Jext  # remember light
    #             Jbot = self.bot.Jext
    #             self.top.Jext = Jtop * dsun[0]  # make dark
    #             self.bot.Jext = Jbot * dsun[1]

    #             lines = Lax.get_lines() + Rax.get_lines()
    #             for line in lines:
    #                 linelabel = line.get_label()
    #                 if linelabel == "_dlntop":
    #                     self.V3T(dlntop)  # fast
    #                     line.set_data(dlntop.VB, abs(dlntop.IB) * scale)
    #                     # Lax.plot(dlntop.VB, abs(dlntop.IB)*scale, c='black',marker='.',label='_dlntop')
    #                 elif linelabel == "_dlnbot":
    #                     self.V3T(dlnbot)  # fast
    #                     line.set_data(dlnbot.VA, abs(dlnbot.IA) * scale)
    #                     # Rax.plot(dlnbot.VA, abs(dlnbot.IA)*scale, c='black',marker='.',label='_dlnbot')
    #                 elif linelabel.startswith("_fit"):
    #                     if Dcalc:
    #                         line.remove()  # remove all the fit lines
    #             if Dcalc:  # replace all coupled dark IV lines
    #                 self.I3Trel(darkFit3T)
    #                 darkFit3T.plotIVslice(step=2, log=True, inplots=(Lax, Rax), labelplus="_fit")

    #             self.top.Jext = Jtop  # make light again
    #             self.bot.Jext = Jbot

    #         tD = time()

    #         if desc == "savefig":
    #             outpath = junction.newoutpath(self.name)
    #             strout = str(self)
    #             with open(os.path.join(outpath, self.name + ".txt"), "wt") as fout:
    #                 fout.write(strout)

    #             # save mathplotlib graphs
    #             Vax.get_figure().savefig(os.path.join(outpath, "Vax.png"))
    #             Iax.get_figure().savefig(os.path.join(outpath, "Iax.png"))
    #             Lax.get_figure().savefig(os.path.join(outpath, "Lax.png"))
    #             Rax.get_figure().savefig(os.path.join(outpath, "Rax.png"))
    #             with VoutBox:
    #                 print(
    #                     "points{0:>6.2f}; P(I){1:>6.2f}; P(V){2:>6.2f}; dark{3:>6.2f} s".format(
    #                         (tmp - ts), (tI - tmp), (tV - tI), (tD - tV)
    #                     ),
    #                     "saved: " + outpath,
    #                 )
    #         else:
    #             with VoutBox:
    #                 print(
    #                     "points{0:>6.2f}; P(I){1:>6.2f}; P(V){2:>6.2f}; dark{3:>6.2f} s".format(
    #                         (tmp - ts), (tI - tmp), (tV - tI), (tD - tV)
    #                     )
    #                 )

    #     # Tandem 3T controls
    #     in_tit = widgets.Label(value="Tandem3T: ", description="title")
    #     in_name = widgets.Text(value=self.name, description="name", tooltip="name of Tandem3T model", layout=tand_layout)
    #     in_Rz = widgets.FloatLogSlider(
    #         value=self.Rz,
    #         base=10,
    #         min=-6,
    #         max=3,
    #         step=0.01,
    #         description="Rz",
    #         layout=tand_layout,
    #         readout_format=".2e",
    #         tooltip="resistance of Z contact",
    #     )
    #     in_savefig = widgets.Button(description="savefig", button_style="success", tooltip="save figures")
    #     in_3Tbut = widgets.Button(description="All", button_style="success", tooltip="slowest calculations")
    #     in_Dbut = widgets.Button(description="dark", button_style="success", tooltip="slow calculations")
    #     in_Ibut = widgets.Button(description="P(V)", button_style="success", tooltip="slow calculations")
    #     in_Vbut = widgets.Button(description="P(I)", button_style="success", tooltip="moderately fast calculations")
    #     in_Mbut = widgets.Button(description="MPPcalc", button_style="success", tooltip="fairly quick calculations")
    #     tand_dict = {"name": in_name, "Rz": in_Rz}
    #     # tandout = widgets.interactive_output(self.set, tand_dict)
    #     tand_ui = widgets.HBox([in_tit, in_name, in_Rz, in_Mbut, in_Vbut, in_Ibut, in_Dbut, in_3Tbut, in_savefig])

    #     if Vdata3T:
    #         meastype = Vdata3T.meastype
    #     elif Idata3T:
    #         meastype = Idata3T.meastype

    #     fitsp = self.specialpoints(meastype=meastype)
    #     Vmax = max(abs(fitsp.Vzt[0]), abs(fitsp.Vrz[0]), abs(fitsp.Vtr[0])) * 2.0
    #     Imax = max(abs(fitsp.Iro[1]), abs(fitsp.Izo[1]), abs(fitsp.Ito[1])) * 2.0

    #     # summary line
    #     VoutBox = widgets.Output()
    #     VoutBox.layout.height = "60px"
    #     with VoutBox:
    #         print("Summary")

    #     if plt.isinteractive:
    #         plt.ioff()
    #         restart = True
    #     else:
    #         restart = False

    #     # graphical outputs
    #     Rout = widgets.Output()
    #     Lout = widgets.Output()
    #     ToutBox = widgets.HBox([Lout, Rout], layout=junc_layout)

    #     with self.debugout:
    #         print(self)

    #     ######## initial plots: Idata and Ifit ##########
    #     RVorI = Iargs["xkey"][0]  #'V' or 'I'

    #     if Idata3T:
    #         Ifit3T = Idata3T.copy()
    #         Ifit3T.set(name=self.name + "_Ifit")
    #         xs, ys = Ifit3T.shape
    #         if xs * ys > pnts * pnts:  # too big
    #             Ifit3T.box(Ifit3T.xkey, min(Ifit3T.x), max(Ifit3T.x), pnts, Ifit3T.ykey, min(Ifit3T.y), max(Ifit3T.y), pnts)
    #             Ifit3T.convert(RVorI, "load2dev")
    #         # self.V3T(Ifit3T)  #fast enough
    #     else:
    #         Ifit3T = IV3T(name=self.name + "_Ifit", meastype=meastype, area=self.lightarea)
    #         Ifit3T.box(Iargs["xkey"], -Imax, Imax, pnts, Iargs["ykey"], -Imax, Imax, pnts)
    #         # Ifit3T.box('IA',-Imax, Imax, pnts, 'IB', -Imax, Imax, pnts)
    #         Ifit3T.convert(RVorI, "load2dev")
    #         # self.V3T(Ifit3T)  #fast enough

    #     if RVorI == "I":
    #         self.V3T(Ifit3T)
    #         if Iargs["density"]:
    #             Iscale = 1000.0 / Ifit3T.area
    #         else:
    #             Iscale = 1000.0
    #     else:  #'V'
    #         # self.I3Trel(Ifit3T)
    #         Iscale = 1.0

    #     if hex:
    #         Iargs["xkey"] = RVorI + "xhex"
    #         Iargs["ykey"] = RVorI + "yhex"

    #     if Idata3T:
    #         Iax, Iobjs = Idata3T.plot(**Iargs)  # plot data
    #         Ifit3T.plot(inplot=(Iax, Iobjs), cmap=None, ccont="red", **Iargs)  # append fit
    #         IdataMPP = Idata3T.MPP()
    #     else:
    #         Iax, Iobjs = Ifit3T.plot(cmap=None, ccont="red", **Iargs)

    #     self.Iax = Iax
    #     Iax.set_title("Light P(I)", size=size)
    #     # Iax.set(title='P(I)')
    #     fitsp.addpoints(Iax, Iargs["xkey"], Iargs["ykey"], density=Iargs["density"], **pltargs)
    #     Ifig = Iax.get_figure()
    #     Ifig.set_figheight(4)
    #     with Rout:
    #         Ifig.show()

    #     ######## initial plots: Vdata and Vfit ##########
    #     LVorI = Vargs["xkey"][0]  #'V' or 'I'

    #     if Vdata3T:
    #         Vfit3T = Vdata3T.copy()
    #         Vfit3T.set(name=self.name + "_Vfit")
    #         xs, ys = Vfit3T.shape
    #         if xs * ys > pnts * pnts:  # too big
    #             Vfit3T.box(Vfit3T.xkey, min(Vfit3T.x), max(Vfit3T.x), pnts, Vfit3T.ykey, min(Vfit3T.y), max(Vfit3T.y), pnts)
    #             Vfit3T.convert(LVorI, "load2dev")
    #         if LVorI == "I":
    #             self.V3T(Vfit3T)
    #         else:
    #             # self.I3Trel(Vfit3T)    #too slow
    #             pass
    #     else:
    #         Vfit3T = IV3T(name=self.name + "_Vfit", meastype=meastype, area=self.lightarea)
    #         Vfit3T.box(Vargs["xkey"], -Vmax, Vmax, pnts, Vargs["ykey"], -Vmax, Vmax, pnts)
    #         # Vfit3T.box('VA',-Vmax, Vmax, pnts, 'VB', -Vmax, Vmax, pnts)
    #         Vfit3T.convert(LVorI, "load2dev")
    #         # self.I3Trel(Vfit3T)    #necessary
    #         if LVorI == "I":
    #             self.V3T(Vfit3T)
    #         else:
    #             self.I3Trel(Vfit3T)  # necessary

    #     if LVorI == "I":
    #         if Vargs["density"]:
    #             Vscale = 1000.0 / Vfit3T.area
    #         else:
    #             Vscale = 1000.0
    #     else:  #'V'
    #         Vscale = 1.0

    #     if hex:
    #         Vargs["xkey"] = LVorI + "xhex"
    #         Vargs["ykey"] = LVorI + "yhex"

    #     if Vdata3T:
    #         Vax, Vobjs = Vdata3T.plot(**Vargs)  # plot data
    #         # Vfit3T.plot(inplot = (Vax, Vobjs), cmap=None, ccont='red', **Vargs) #append fit
    #         VdataMPP = Vdata3T.MPP()
    #     else:
    #         Vax, Vobjs = Vfit3T.plot(cmap=None, ccont="red", **Vargs)

    #     self.Vax = Vax
    #     Vax.set_title("Light P(V)", size=size)
    #     # Vax.set(title='P(V)')
    #     fitsp.addpoints(Vax, Vargs["xkey"], Vargs["ykey"], density=Vargs["density"], **pltargs)
    #     Vfig = Vax.get_figure()
    #     Vfig.set_figheight(4)
    #     with Lout:
    #         Vfig.show()

    #     ######## initial plots: darkData and darkFit ##########
    #     if darkData3T:
    #         Lax, Rax = darkData3T.plotIVslice(step=2, log=True)  # plot dark data
    #         Lax.set_xlim(np.min(darkData3T.y) - 0.1, np.max(darkData3T.y) + 0.1)
    #         Rax.set_xlim(np.min(darkData3T.x) - 0.1, np.max(darkData3T.x) + 0.1)
    #         self.Lax = Lax
    #         self.Rax = Rax
    #         Lax.set_title("Top coupled dark I(V)", size=size)
    #         Rax.set_title("Bottom coupled dark I(V)", size=size)
    #         # create dark fit model
    #         darkFit3T = darkData3T.copy()  # same 2D span as data
    #         darkFit3T.set(name=self.name + "_darkfit")
    #         # create top dark IV with Ibot=0
    #         dlntop = IV3T(name="dlntop", meastype="CZ", area=self.lightarea)
    #         dlntop.line("Ito", dlo, dhi, dpnts, "Iro", "0", log=True)
    #         # create bot dark IV with Itop=0
    #         dlnbot = IV3T(name="dlnbot", meastype="CZ", area=self.lightarea)
    #         dlnbot.line("Iro", dlo, dhi, dpnts, "Ito", "0", log=True)

    #         Jtop = self.top.Jext  # remember light
    #         Jbot = self.bot.Jext
    #         self.top.Jext = Jtop * dsun[0]  # make dark (almost)
    #         self.bot.Jext = Jbot * dsun[1]

    #         # calculate dark fit
    #         self.V3T(dlntop)  # fast
    #         # with self.debugout: print(dlntop)
    #         Lax.plot(dlntop.VB, abs(dlntop.IB) * scale, c="black", label="_dlntop", zorder=5)
    #         self.V3T(dlnbot)  # fast
    #         # with self.debugout: print(dlnbot)
    #         Rax.plot(dlnbot.VA, abs(dlnbot.IA) * scale, c="black", label="_dlnbot", zorder=5)
    #         if False:  # slow
    #             self.I3Trel(darkFit3T)
    #             darkFit3T.plotIVslice(step=2, log=True, inplots=(Lax, Rax), labelplus="_fit")

    #         self.top.Jext = Jtop  # make light again
    #         self.bot.Jext = Jbot
    #         Lax.get_figure().set_figheight(4)
    #         Rax.get_figure().set_figheight(4)
    #         with Lout:
    #             Lax.get_figure().show()
    #         with Rout:
    #             Rax.get_figure().show()
    #     else:
    #         darkFit3T = None

    #     if restart:
    #         plt.ion()

    #     in_name.observe(on_3Tchange, names="value")  # update values
    #     in_Rz.observe(on_3Tchange, names="value")  # update values

    #     # junction ui
    #     uit = self.top.controls()
    #     uib = self.bot.controls()
    #     junc_ui = widgets.HBox([uit, uib])
    #     for jui in [uit, uib]:
    #         kids = jui.children
    #         for cntrl in kids:
    #             if type(cntrl) in replot_types:
    #                 cntrl.observe(on_3Treplot, names="value")  # replot fast
    #     in_Rz.observe(on_3Treplot, names="value")  # replot fast
    #     in_savefig.on_click(on_3Treplot)  # replot all
    #     in_3Tbut.on_click(on_3Treplot)  # replot all
    #     in_Dbut.on_click(on_3Treplot)  # replot some
    #     in_Ibut.on_click(on_3Treplot)  # replot some
    #     in_Vbut.on_click(on_3Treplot)  # replot some
    #     in_Mbut.on_click(on_3Treplot)  # replot some

    #     in_Mbut.click()  # calculate MPP

    #     ui = widgets.VBox([ToutBox, VoutBox, tand_ui, junc_ui])
    #     self.ui = ui

    #     return ui, Vax, Iax

    # def plot(self, pnts=31, meastype="CZ", oper="load2dev", cmap="terrain"):
    #     """
    #     calculate and plot Tandem3T devices 'self'

    #     """

    #     # bounding points
    #     factor = 1.2
    #     pltargs = {
    #         "lw": 0,
    #         "ms": 7,
    #         "mew": 1,
    #         "mec": "black",
    #         "mfc": "white",
    #         "marker": "o",
    #         "c": "red",
    #         "label": "fitsp",
    #         "zorder": 5,
    #     }
    #     sp = self.specialpoints(meastype)
    #     colors = ["white", "lightgreen", "lightgray", "pink", "orange", "cyan", "cyan", "cyan", "cyan", "cyan"]
    #     Vmax = max(abs(sp.Vzt[0]), abs(sp.Vrz[0]), abs(sp.Vtr[0]))
    #     Imax = max(abs(sp.Iro[1]), abs(sp.Izo[1]), abs(sp.Ito[1]))
    #     ii = sp.names.index("MPP")  # index of MPP from sp

    #     iv = list()  # empty list to contain IV3T structures
    #     axs = list()  # empty list to contain axis of each figure
    #     # figs = list()  #empty list to contain each figure

    #     for i, VorI in enumerate(["V", "I"]):

    #         ts = time()
    #         # create box IV3T instance
    #         name = VorI + "plot"
    #         common = meastype[1].lower()
    #         if VorI == "V":
    #             devlist = IV3T.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
    #             factor = 1.1
    #             xmax = Vmax * factor
    #             ymax = Vmax * factor
    #         elif VorI == "I":
    #             devlist = IV3T.Idevlist.copy()  # ['Iro','Izo','Ito']
    #             factor = 3.0
    #             xmax = Imax * factor
    #             ymax = Imax * factor

    #         if oper == "load2dev":
    #             xkey = VorI + "A"
    #             ykey = VorI + "B"
    #             # ax.set_title(self.name + ' P-'+VorI+'-'+VorI + ' ' + meastype + '-mode ' , loc='center')
    #         elif oper == "dev2load":
    #             xkey = devlist[0]
    #             ykey = devlist[1]
    #             # ax.set_title(self.name + ' P-'+VorI+'-'+VorI , loc='center')
    #         elif oper == "dev2hex":
    #             xkey = VorI + "xhex"
    #             ykey = VorI + "yhex"
    #         elif oper == "hex2dev":
    #             xkey = VorI + "xhex"
    #             ykey = VorI + "yhex"

    #         iv.append(IV3T(name=name, meastype=meastype, area=self.lightarea))  # add another IV3T class to iv list
    #         iv[i].box(xkey, -xmax, xmax, pnts, ykey, -ymax, ymax, pnts)
    #         if oper:
    #             iv[i].convert(VorI, oper)

    #         if VorI == "V":
    #             self.I3Trel(iv[i])  # calculate I(V)
    #         else:
    #             self.V3T(iv[i])  # calculate V(I)

    #         sp.append(iv[i].MPP(VorI))  # append MPP of current iv[i] to special points

    #         ax, objs = iv[i].plot(xkey=xkey, ykey=ykey, cmap=cmap)
    #         sp.addpoints(ax, xkey, ykey, **pltargs)
    #         axs.append(ax)
    #         # figs.append(fig)

    #         xkey = VorI + "xhex"
    #         ykey = VorI + "yhex"
    #         ax, objs = iv[i].plot(xkey=xkey, ykey=ykey, cmap=cmap)
    #         sp.addpoints(ax, xkey, ykey, **pltargs)
    #         axs.append(ax)
    #         # figs.append(fig)

    #         te = time()
    #         dt = te - ts
    #         print("axs[{0:g}]: {1:d}pnts , {2:2.4f} s".format(i, pnts, dt))

    #     return axs, iv, sp
