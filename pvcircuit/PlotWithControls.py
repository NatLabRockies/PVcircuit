# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.EY     use Ripalda's Tandem proxy spectra for energy yield
"""

import math
import os
from time import time

import ipywidgets as widgets
import matplotlib as mpl
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd
from parse import parse

import pvcircuit as pvc
from pvcircuit import junction
from pvcircuit.iv3T import IV3T
from pvcircuit.junction import Junction
from pvcircuit.multi2T import Multi2T
from pvcircuit.qe import EQE
from pvcircuit.tandem3T import Tandem3T


class PlotsWithControls:
    def __init__(self, class_to_plot, *args, **kwargs) -> None:

        self.debugout = widgets.Output()  # debug output

        self.VdataMPP = None
        self.IdataMPP = None
        self.class_to_plot = class_to_plot

        if isinstance(class_to_plot, Junction):
            self.controls_junction(class_to_plot)
        elif isinstance(class_to_plot, Tandem3T):
            self.controls_3T(class_to_plot, *args, **kwargs)
        elif isinstance(class_to_plot, EQE):
            self.controls_qe(class_to_plot, *args, **kwargs)
        elif isinstance(class_to_plot, Multi2T):
            self.controls_2T(class_to_plot, *args, **kwargs)
        else:
            raise ValueError(f"cannot plot class {type(class_to_plot)}")

    def controls_junction(self, class_to_plot):
        """
        use interactive_output for GUI in IPython
        """

        """
        use interactive_output for GUI in IPython
        """

        cell_layout = widgets.Layout(display="inline_flex", flex_flow="row", justify_content="flex-end", width="300px")
        # controls
        in_name = widgets.Text(value=class_to_plot.name, description="name", layout=cell_layout, continuous_update=False)
        in_Eg = widgets.FloatSlider(value=class_to_plot.Eg, min=0.1, max=3.0, step=0.01, description="Eg", layout=cell_layout, readout_format=".2f")
        in_TC = widgets.FloatSlider(value=class_to_plot.TC, min=-40, max=200.0, step=2, description="TC", layout=cell_layout, readout_format=".1f")
        in_Jext = widgets.FloatSlider(value=class_to_plot.Jext, min=0.0, max=0.080, step=0.001, description="Jext", layout=cell_layout, readout_format=".4f")
        in_JLC = widgets.FloatSlider(
            value=class_to_plot.JLC,
            min=0.0,
            max=0.080,
            step=0.001,
            description="JLC",
            layout=cell_layout,
            readout_format=".4f",
            disabled=True,
        )
        in_Gsh = widgets.FloatLogSlider(value=class_to_plot.Gsh, base=10, min=-12, max=3, step=0.01, description="Gsh", layout=cell_layout, readout_format=".2e")
        in_Rser = widgets.FloatLogSlider(value=class_to_plot.Rser, base=10, min=-7, max=3, step=0.01, description="Rser", layout=cell_layout, readout_format=".2e")
        in_lightarea = widgets.FloatLogSlider(value=class_to_plot.lightarea, base=10, min=-6, max=3.0, step=0.1, description="lightarea", layout=cell_layout)
        in_totalarea = widgets.FloatSlider(
            value=class_to_plot.totalarea,
            min=class_to_plot.lightarea,
            max=1e3,
            step=0.1,
            description="totalarea",
            layout=cell_layout,
        )
        in_beta = widgets.FloatSlider(value=class_to_plot.beta, min=0.0, max=50.0, step=0.1, description="beta", layout=cell_layout, readout_format=".2e")
        in_gamma = widgets.FloatSlider(value=class_to_plot.gamma, min=0.0, max=3.0, step=0.1, description="gamma", layout=cell_layout, readout_format=".2e")
        in_pn = widgets.IntSlider(value=class_to_plot.pn, min=-1, max=1, step=1, description="pn", layout=cell_layout)

        # linkages
        # arealink = widgets.jslink((in_lightarea, "value"), (in_totalarea, "min"))  # also jsdlink works

        # attr = ["name"] + class_to_plot.ATTR.copy()
        cntrls = [in_name, in_Eg, in_TC, in_Gsh, in_Rser, in_lightarea, in_totalarea, in_Jext, in_JLC, in_beta, in_gamma, in_pn]
        # sing_dict = dict(zip(attr, cntrls))
        # singout = widgets.interactive_output(class_to_plot.set, sing_dict)  #all at once

        def on_juncchange(change):
            # function for changing values
            old = change["old"]  # old value
            new = change["new"]  # new value
            owner = change["owner"]  # control
            value = owner.value
            desc = owner.description

            if new == old:
                with self.debugout:
                    print("Jcontrol: " + desc + "=", value)
            else:
                with self.debugout:
                    print("Jcontrol: " + desc + "->", value)
                class_to_plot.set(**{desc: value})

            # iout.clear_output()
            # with iout: print(self)

        # diode array
        in_tit = widgets.Label(value="Junction", description="Junction")
        in_diodelab = widgets.Label(value="diodes:", description="diodes:")
        # diode_layout = widgets.Layout(flex_flow="column", align_items="center")

        cntrls.append(in_diodelab)
        in_n = []  # empty list of n controls
        in_ratio = []  # empty list of Jratio controls
        diode_dict = {}
        for i in range(len(class_to_plot.n)):
            in_n.append(widgets.FloatLogSlider(value=class_to_plot.n[i], base=10, min=-1, max=1, step=0.001, description="n[" + str(i) + "]", layout=cell_layout))
            in_ratio.append(
                widgets.FloatLogSlider(
                    value=class_to_plot.J0ratio[i],
                    base=10,
                    min=-6,
                    max=6,
                    step=0.1,
                    description="J0ratio[" + str(i) + "]",
                    layout=cell_layout,
                )
            )
            cntrls.append(in_n[i])
            cntrls.append(in_ratio[i])
            diode_dict["n[" + str(i) + "]"] = in_n[i]
            diode_dict["J0ratio[" + str(i) + "]"] = in_ratio[i]
            # hui.append(widgets.HBox([in_n[i],in_ratio[i]]))
            # cntrls.append(hui[i])

        # diodeout = widgets.interactive_output(class_to_plot.set, diode_dict)  #all at once

        if class_to_plot.RBB_dict:
            RBB_keys = list(class_to_plot.RBB_dict.keys())
            in_rbblab = widgets.Label(value="RBB:", description="RBB:")
            cntrls.append(in_rbblab)
            in_rbb = []  # empty list of n controls
            for i, key in enumerate(RBB_keys):
                with self.debugout:
                    print("RBB:", i, key)
                if key == "method":
                    in_rbb.append(
                        widgets.Dropdown(
                            options=["", "JFG", "bishop"],
                            value=class_to_plot.RBB_dict[key],
                            description=key,
                            layout=cell_layout,
                            continuous_update=False,
                        )
                    )
                else:
                    in_rbb.append(widgets.FloatLogSlider(value=class_to_plot.RBB_dict[key], base=10, min=-10, max=5, step=0.1, description=key, layout=cell_layout))
                cntrls.append(in_rbb[i])

        for cntrl in cntrls:
            cntrl.observe(on_juncchange, names="value")

        # output
        iout = widgets.Output()
        iout.layout.height = "5px"
        # with iout: print(class_to_plot)
        cntrls.append(iout)

        # user interface
        box_layout = widgets.Layout(display="flex", flex_flow="column", align_items="center", border="1px solid black", width="320px", height="350px")

        ui = widgets.VBox([in_tit] + cntrls, layout=box_layout)
        # self.ui = ui  # make it an attribute

        return ui

    def controls_3T(
        self,
        class_to_plot,
        Vdata3T=None,
        Idata3T=None,
        darkData3T=None,
        hex=False,
        meastype="CZ",
        size="x-large",
        Iargs={"xkey": "IA", "ykey": "IB", "zkey": "Ptot", "density": True},
        Vargs={"xkey": "VA", "ykey": "VB", "zkey": "Ptot", "density": True},
    ):
        """
        use interactive_output for GUI in IPython
        Iargs = {'xkey':'VA', 'ykey':'VB', 'zkey':'IB', 'log':True}
        Vargs = {'xkey':'VA', 'ykey':'VB', 'zkey':'IA', 'log':True}

        """
        tand_layout = widgets.Layout(width="300px", height="40px")
        _vout_layout = widgets.Layout(width="180px", height="40px")
        _grid_layout = widgets.Layout(grid_template_columns="repeat(2, 180px)", grid_template_rows="repeat(3, 30px)", height="100px")
        junc_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")
        replot_types = [
            widgets.widgets.widget_float.BoundedFloatText,
            widgets.widgets.widget_int.BoundedIntText,
            widgets.widgets.widget_int.IntSlider,
            widgets.widgets.widget_float.FloatSlider,
            widgets.widgets.widget_float.FloatLogSlider,
        ]
        scale = 1000.0
        pnts = 71
        pltargs = {
            "lw": 0,
            "ms": 7,
            "mew": 1,
            "mec": "black",
            "mfc": "white",
            "marker": "o",
            "c": "red",
            "label": "fitsp",
            "zorder": 5,
        }
        dsun = [1e-9, 1e-10]  # dark suns
        dpnts = 21
        dlo = -14
        dhi = 2

        def on_3Tchange(change):
            # function for changing values
            _old = change["old"]  # old value (unused -- kept for callback signature documentation)
            _new = change["new"]  # new value (unused)
            owner = change["owner"]  # control
            value = owner.value
            desc = owner.description
            with self.debugout:
                print("Tcontrol: " + desc + "->", value)
            class_to_plot.set(**{desc: value})

        def on_3Treplot(change):
            # change info
            fast = False
            Vcalc = False
            Icalc = False
            Dcalc = False
            if type(change) is widgets.widgets.widget_button.Button:
                owner = change
                desc = owner.description
            else:  # other controls
                owner = change["owner"]  # control
            desc = owner.description
            if desc == "All":
                Vcalc = True
                Icalc = True
                Dcalc = True
            elif desc == "P(I)":
                Vcalc = True
            elif desc == "P(V)":
                Icalc = True
            elif desc == "dark":
                Dcalc = True
            elif desc == "MPPcalc":
                fast = False
            elif desc == "savefig":
                fast = True
            else:
                fast = True

            with self.debugout:
                print(desc, class_to_plot)
            # recalculate
            ts = time()
            fitsp = class_to_plot.specialpoints(meastype=meastype, fast=fast)
            if Iargs["density"]:
                fscale = 1000.0 / fitsp.area
            else:
                fscale = 1000.0
            # summary line
            fmtstr = "Fit:  (Vzt = {0:>5.3f}, Vrz = {1:>5.3f}, Vtr = {2:>5.3f} V),   "
            if Iargs["density"]:
                fmtstr += "(Jro = {3:>5.2f}, Jzo = {4:>5.2f}, Jto = {5:>5.2f} mA/mA)"
            else:
                fmtstr += "(Iro = {3:>5.2f}, Izo = {4:>5.2f}, Ito = {5:>5.2f} mA)"

            VoutBox.clear_output()
            if self.VdataMPP:
                outstr = (fmtstr + ",   Pmp = {6:>5.2f} mW/cm^2").format(
                    self.VdataMPP.Vzt[0],
                    self.VdataMPP.Vrz[0],
                    self.VdataMPP.Vtr[0],
                    self.VdataMPP.Iro[0] * fscale,
                    self.VdataMPP.Izo[0] * fscale,
                    self.VdataMPP.Ito[0] * fscale,
                    self.VdataMPP.Ptot[0] * fscale,
                )
                with VoutBox:
                    print(outstr.replace("Fit:", "VData:"))

            elif self.IdataMPP:
                outstr = (fmtstr + ",   Pmp = {6:>5.2f} mW/cm^2").format(
                    self.IdataMPP.Vzt[0],
                    self.IdataMPP.Vrz[0],
                    self.IdataMPP.Vtr[0],
                    self.IdataMPP.Iro[0] * fscale,
                    self.IdataMPP.Izo[0] * fscale,
                    self.IdataMPP.Ito[0] * fscale,
                    self.IdataMPP.Ptot[0] * fscale,
                )
                with VoutBox:
                    print(outstr.replace("Fit:", "IData:"))

            if "MPP" in fitsp.names:  # not fast
                ii = fitsp.names.index("MPP")  # index of MPP from sp
                fmtstr += ",   Pmp = {6:>5.2f} mW/cm^2"
                outstr = fmtstr.format(
                    fitsp.Vzt[0],
                    fitsp.Vrz[0],
                    fitsp.Vtr[0],
                    fitsp.Iro[1] * fscale,
                    fitsp.Izo[1] * fscale,
                    fitsp.Ito[1] * fscale,
                    fitsp.Ptot[ii] * fscale,
                )
                with VoutBox:
                    print(outstr.replace("fit", "data"))
            else:
                outstr = fmtstr.format(fitsp.Vzt[0], fitsp.Vrz[0], fitsp.Vtr[0], fitsp.Iro[1] * fscale, fitsp.Izo[1] * fscale, fitsp.Ito[1] * fscale)
                with VoutBox:
                    print(outstr.replace("fit", "data"))

            tmp = time()

            # with Rout: # right output device -> P(I) typically
            # replot: Iax with Rout
            lines = Iax.get_lines()
            for line in lines:
                linelabel = line.get_label()
                if linelabel == "fitsp":
                    xp = getattr(fitsp, Iargs["xkey"]) * Iscale
                    yp = getattr(fitsp, Iargs["ykey"]) * Iscale
                    line.set_data(xp, yp)
            if Vcalc:
                if RVorI == "I":
                    class_to_plot.V3T(Ifit3T)
                else:
                    class_to_plot.I3Trel(Ifit3T)  # slow
                # self.V3T(Ifit3T)
                for i, obj in enumerate(Iobjs):
                    if type(obj) is mpl.contour.QuadContourSet:  # contours
                        if obj.colors == "red":  # identify fit contour
                            fitcont = Iobjs.pop(i)  # remove it
                            for coll in fitcont.collections:
                                if coll in Iax.collections:
                                    Iax.collections.remove(coll)  # remove old contour lines from plot
                            for text in fitcont.labelTexts:
                                if text in Iax.texts:
                                    Iax.texts.remove(text)  # remove old contour labels from plot
                            break

                with Rout:
                    Ifit3T.plot(inplot=(Iax, Iobjs), cmap=None, ccont="red", **Iargs)  # replot fit contour

            tI = time()

            # with Lout: # left output device -> P(V) typically
            # replot: Vax with Lout
            lines = Vax.get_lines()
            for line in lines:
                linelabel = line.get_label()
                if linelabel == "fitsp":
                    xp = getattr(fitsp, Vargs["xkey"]) * Vscale
                    yp = getattr(fitsp, Vargs["ykey"]) * Vscale
                    line.set_data(xp, yp)
            if Icalc:
                if LVorI == "I":
                    class_to_plot.V3T(Vfit3T)
                else:
                    class_to_plot.I3Trel(Vfit3T)  # slow
                # class_to_plot.I3Trel(Vfit3T)    #slow
                for i, obj in enumerate(Vobjs):
                    if type(obj) is mpl.contour.QuadContourSet:  # contours
                        if obj.colors == "red":  # identify fit contour
                            fitcont = Vobjs.pop(i)  # remove it
                            for coll in fitcont.collections:
                                if coll in Vax.collections:
                                    Vax.collections.remove(coll)  # remove old contour lines from plot
                            for text in fitcont.labelTexts:
                                if text in Vax.texts:
                                    Vax.texts.remove(text)  # remove old contour labels from plot
                            break

                with Lout:
                    Vfit3T.plot(inplot=(Vax, Vobjs), cmap=None, ccont="red", **Vargs)  # replot fit contour

            tV = time()

            # replot: dark Lax, Rax
            if darkFit3T:  # add new dark fit
                # dark plots
                Jtop = class_to_plot.top.Jext  # remember light
                Jbot = class_to_plot.bot.Jext
                class_to_plot.top.Jext = Jtop * dsun[0]  # make dark
                class_to_plot.bot.Jext = Jbot * dsun[1]

                lines = Lax.get_lines() + Rax.get_lines()
                for line in lines:
                    linelabel = line.get_label()
                    if linelabel == "_dlntop":
                        class_to_plot.V3T(dlntop)  # fast
                        line.set_data(dlntop.VB, abs(dlntop.IB) * scale)
                        # Lax.plot(dlntop.VB, abs(dlntop.IB)*scale, c='black',marker='.',label='_dlntop')
                    elif linelabel == "_dlnbot":
                        class_to_plot.V3T(dlnbot)  # fast
                        line.set_data(dlnbot.VA, abs(dlnbot.IA) * scale)
                        # Rax.plot(dlnbot.VA, abs(dlnbot.IA)*scale, c='black',marker='.',label='_dlnbot')
                    elif linelabel.startswith("_fit"):
                        if Dcalc:
                            line.remove()  # remove all the fit lines
                if Dcalc:  # replace all coupled dark IV lines
                    class_to_plot.I3Trel(darkFit3T)
                    darkFit3T.plotIVslice(step=2, log=True, inplots=(Lax, Rax), labelplus="_fit")

                class_to_plot.top.Jext = Jtop  # make light again
                class_to_plot.bot.Jext = Jbot

            tD = time()

            if desc == "savefig":
                outpath = junction.newoutpath(class_to_plot.name)
                strout = str(class_to_plot)
                with open(os.path.join(outpath, class_to_plot.name + ".txt"), "wt") as fout:
                    fout.write(strout)

                # save mathplotlib graphs
                Vax.get_figure().savefig(os.path.join(outpath, "Vax.png"))
                Iax.get_figure().savefig(os.path.join(outpath, "Iax.png"))
                Lax.get_figure().savefig(os.path.join(outpath, "Lax.png"))
                Rax.get_figure().savefig(os.path.join(outpath, "Rax.png"))
                with VoutBox:
                    print(
                        "points{0:>6.2f}; P(I){1:>6.2f}; P(V){2:>6.2f}; dark{3:>6.2f} s".format((tmp - ts), (tI - tmp), (tV - tI), (tD - tV)),
                        "saved: " + outpath,
                    )
            else:
                with VoutBox:
                    print("points{0:>6.2f}; P(I){1:>6.2f}; P(V){2:>6.2f}; dark{3:>6.2f} s".format((tmp - ts), (tI - tmp), (tV - tI), (tD - tV)))

        # Tandem 3T controls
        in_tit = widgets.Label(value="Tandem3T: ", description="title")
        in_name = widgets.Text(value=class_to_plot.name, description="name", tooltip="name of Tandem3T model", layout=tand_layout)
        in_Rz = widgets.FloatLogSlider(
            value=class_to_plot.Rz,
            base=10,
            min=-6,
            max=3,
            step=0.01,
            description="Rz",
            layout=tand_layout,
            readout_format=".2e",
            tooltip="resistance of Z contact",
        )
        in_savefig = widgets.Button(description="savefig", button_style="success", tooltip="save figures")
        in_3Tbut = widgets.Button(description="All", button_style="success", tooltip="slowest calculations")
        in_Dbut = widgets.Button(description="dark", button_style="success", tooltip="slow calculations")
        in_Ibut = widgets.Button(description="P(V)", button_style="success", tooltip="slow calculations")
        in_Vbut = widgets.Button(description="P(I)", button_style="success", tooltip="moderately fast calculations")
        in_Mbut = widgets.Button(description="MPPcalc", button_style="success", tooltip="fairly quick calculations")
        _tand_dict = {"name": in_name, "Rz": in_Rz}
        # tandout = widgets.interactive_output(class_to_plot.set, tand_dict)
        tand_ui = widgets.HBox([in_tit, in_name, in_Rz, in_Mbut, in_Vbut, in_Ibut, in_Dbut, in_3Tbut, in_savefig])

        if Vdata3T:
            meastype = Vdata3T.meastype
        elif Idata3T:
            meastype = Idata3T.meastype

        fitsp = class_to_plot.specialpoints(meastype=meastype)
        Vmax = max(abs(fitsp.Vzt[0]), abs(fitsp.Vrz[0]), abs(fitsp.Vtr[0])) * 2.0
        Imax = max(abs(fitsp.Iro[1]), abs(fitsp.Izo[1]), abs(fitsp.Ito[1])) * 2.0

        # summary line
        VoutBox = widgets.Output()
        VoutBox.layout.height = "60px"
        with VoutBox:
            print("Summary")

        if plt.isinteractive:
            plt.ioff()
            restart = True
        else:
            restart = False

        # graphical outputs
        Rout = widgets.Output()
        Lout = widgets.Output()
        ToutBox = widgets.HBox([Lout, Rout], layout=junc_layout)

        with self.debugout:
            print(self)

        ######## initial plots: Idata and Ifit ##########
        RVorI = Iargs["xkey"][0]  #'V' or 'I'

        if Idata3T:
            Ifit3T = Idata3T.copy()
            Ifit3T.set(name=class_to_plot.name + "_Ifit")
            xs, ys = Ifit3T.shape
            if xs * ys > pnts * pnts:  # too big
                Ifit3T.box(Ifit3T.xkey, min(Ifit3T.x), max(Ifit3T.x), pnts, Ifit3T.ykey, min(Ifit3T.y), max(Ifit3T.y), pnts)
                Ifit3T.convert(RVorI, "load2dev")
            # self.V3T(Ifit3T)  #fast enough
        else:
            Ifit3T = IV3T(name=class_to_plot.name + "_Ifit", meastype=meastype, area=class_to_plot.lightarea)
            Ifit3T.box(Iargs["xkey"], -Imax, Imax, pnts, Iargs["ykey"], -Imax, Imax, pnts)
            # Ifit3T.box('IA',-Imax, Imax, pnts, 'IB', -Imax, Imax, pnts)
            Ifit3T.convert(RVorI, "load2dev")
            # self.V3T(Ifit3T)  #fast enough

        if RVorI == "I":
            class_to_plot.V3T(Ifit3T)
            if Iargs["density"]:
                Iscale = 1000.0 / Ifit3T.area
            else:
                Iscale = 1000.0
        else:  #'V'
            # class_to_plot.I3Trel(Ifit3T)
            Iscale = 1.0

        if hex:
            Iargs["xkey"] = RVorI + "xhex"
            Iargs["ykey"] = RVorI + "yhex"

        if Idata3T:
            Iax, Iobjs = Idata3T.plot(**Iargs)  # plot data
            Ifit3T.plot(inplot=(Iax, Iobjs), cmap=None, ccont="red", **Iargs)  # append fit
            self.IdataMPP = Idata3T.MPP()
        else:
            Iax, Iobjs = Ifit3T.plot(cmap=None, ccont="red", **Iargs)

        self.Iax = Iax
        Iax.set_title("Light P(I)", size=size)
        # Iax.set(title='P(I)')
        fitsp.addpoints(Iax, Iargs["xkey"], Iargs["ykey"], density=Iargs["density"], **pltargs)
        Ifig = Iax.get_figure()
        Ifig.set_figheight(4)
        with Rout:
            Ifig.show()

        ######## initial plots: Vdata and Vfit ##########
        LVorI = Vargs["xkey"][0]  #'V' or 'I'

        if Vdata3T:
            Vfit3T = Vdata3T.copy()
            Vfit3T.set(name=class_to_plot.name + "_Vfit")
            xs, ys = Vfit3T.shape
            if xs * ys > pnts * pnts:  # too big
                Vfit3T.box(Vfit3T.xkey, min(Vfit3T.x), max(Vfit3T.x), pnts, Vfit3T.ykey, min(Vfit3T.y), max(Vfit3T.y), pnts)
                Vfit3T.convert(LVorI, "load2dev")
            if LVorI == "I":
                class_to_plot.V3T(Vfit3T)
            else:
                # class_to_plot.I3Trel(Vfit3T)    #too slow
                pass
        else:
            Vfit3T = IV3T(name=class_to_plot.name + "_Vfit", meastype=meastype, area=class_to_plot.lightarea)
            Vfit3T.box(Vargs["xkey"], -Vmax, Vmax, pnts, Vargs["ykey"], -Vmax, Vmax, pnts)
            # Vfit3T.box('VA',-Vmax, Vmax, pnts, 'VB', -Vmax, Vmax, pnts)
            Vfit3T.convert(LVorI, "load2dev")
            # class_to_plot.I3Trel(Vfit3T)    #necessary
            if LVorI == "I":
                class_to_plot.V3T(Vfit3T)
            else:
                class_to_plot.I3Trel(Vfit3T)  # necessary

        if LVorI == "I":
            if Vargs["density"]:
                Vscale = 1000.0 / Vfit3T.area
            else:
                Vscale = 1000.0
        else:  #'V'
            Vscale = 1.0

        if hex:
            Vargs["xkey"] = LVorI + "xhex"
            Vargs["ykey"] = LVorI + "yhex"

        if Vdata3T:
            Vax, Vobjs = Vdata3T.plot(**Vargs)  # plot data
            # Vfit3T.plot(inplot = (Vax, Vobjs), cmap=None, ccont='red', **Vargs) #append fit
            self.VdataMPP = Vdata3T.MPP()
        else:
            Vax, Vobjs = Vfit3T.plot(cmap=None, ccont="red", **Vargs)

        self.Vax = Vax
        Vax.set_title("Light P(V)", size=size)
        # Vax.set(title='P(V)')
        fitsp.addpoints(Vax, Vargs["xkey"], Vargs["ykey"], density=Vargs["density"], **pltargs)
        Vfig = Vax.get_figure()
        Vfig.set_figheight(4)
        with Lout:
            Vfig.show()

        ######## initial plots: darkData and darkFit ##########
        if darkData3T:
            Lax, Rax = darkData3T.plotIVslice(step=2, log=True)  # plot dark data
            Lax.set_xlim(np.min(darkData3T.y) - 0.1, np.max(darkData3T.y) + 0.1)
            Rax.set_xlim(np.min(darkData3T.x) - 0.1, np.max(darkData3T.x) + 0.1)
            self.Lax = Lax
            self.Rax = Rax
            Lax.set_title("Top coupled dark I(V)", size=size)
            Rax.set_title("Bottom coupled dark I(V)", size=size)
            # create dark fit model
            darkFit3T = darkData3T.copy()  # same 2D span as data
            darkFit3T.set(name=class_to_plot.name + "_darkfit")
            # create top dark IV with Ibot=0
            dlntop = IV3T(name="dlntop", meastype="CZ", area=class_to_plot.lightarea)
            dlntop.line("Ito", dlo, dhi, dpnts, "Iro", "0", log=True)
            # create bot dark IV with Itop=0
            dlnbot = IV3T(name="dlnbot", meastype="CZ", area=class_to_plot.lightarea)
            dlnbot.line("Iro", dlo, dhi, dpnts, "Ito", "0", log=True)

            Jtop = class_to_plot.top.Jext  # remember light
            Jbot = class_to_plot.bot.Jext
            class_to_plot.top.Jext = Jtop * dsun[0]  # make dark (almost)
            class_to_plot.bot.Jext = Jbot * dsun[1]

            # calculate dark fit
            class_to_plot.V3T(dlntop)  # fast
            # with self.debugout: print(dlntop)
            Lax.plot(dlntop.VB, abs(dlntop.IB) * scale, c="black", label="_dlntop", zorder=5)
            class_to_plot.V3T(dlnbot)  # fast
            # with self.debugout: print(dlnbot)
            Rax.plot(dlnbot.VA, abs(dlnbot.IA) * scale, c="black", label="_dlnbot", zorder=5)
            if False:  # slow
                class_to_plot.I3Trel(darkFit3T)
                darkFit3T.plotIVslice(step=2, log=True, inplots=(Lax, Rax), labelplus="_fit")

            class_to_plot.top.Jext = Jtop  # make light again
            class_to_plot.bot.Jext = Jbot
            Lax.get_figure().set_figheight(4)
            Rax.get_figure().set_figheight(4)
            with Lout:
                Lax.get_figure().show()
            with Rout:
                Rax.get_figure().show()
        else:
            darkFit3T = None

        if restart:
            plt.ion()

        in_name.observe(on_3Tchange, names="value")  # update values
        in_Rz.observe(on_3Tchange, names="value")  # update values

        # junction ui
        # uit = class_to_plot.top.controls()
        uit = self.controls_junction(class_to_plot.top)
        # uib = class_to_plot.bot.controls()
        uib = self.controls_junction(class_to_plot.bot)
        junc_ui = widgets.HBox([uit, uib])
        for jui in [uit, uib]:
            kids = jui.children
            for cntrl in kids:
                if type(cntrl) in replot_types:
                    cntrl.observe(on_3Treplot, names="value")  # replot fast
        in_Rz.observe(on_3Treplot, names="value")  # replot fast
        in_savefig.on_click(on_3Treplot)  # replot all
        in_3Tbut.on_click(on_3Treplot)  # replot all
        in_Dbut.on_click(on_3Treplot)  # replot some
        in_Ibut.on_click(on_3Treplot)  # replot some
        in_Vbut.on_click(on_3Treplot)  # replot some
        in_Mbut.on_click(on_3Treplot)  # replot some

        in_Mbut.click()  # calculate MPP

        ui = widgets.VBox([ToutBox, VoutBox, tand_ui, junc_ui])
        self.ui = ui
        self.Vax = Vax
        self.Iax = Iax
        self.uit = uit
        self.uib = uib

        return ui, Vax, Iax

    def controls_qe(self, qe_class, Pspec="global", ispec=0, specname=None, xspec=pvc.qe.wvl):
        """
        use interactive_output for GUI in IPython
        """
        tand_layout = widgets.Layout(width="300px", height="40px")
        _vout_layout = widgets.Layout(width="180px", height="40px")
        junc_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")
        _multi_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")

        _replot_types = [
            widgets.widgets.widget_float.BoundedFloatText,
            widgets.widgets.widget_int.BoundedIntText,
            widgets.widgets.widget_int.IntSlider,
            widgets.widgets.widget_float.FloatSlider,
            widgets.widgets.widget_float.FloatLogSlider,
        ]

        def on_EQEchange(change):
            # function for changing values
            _old = change["old"]  # old value (handler body commented out)
            _new = change["new"]  # new value
            owner = change["owner"]  # control
            _value = owner.value
            _desc = owner.description
            # with self.debugout: print('Mcontrol: ' + desc + '->', value)
            # class_to_plot.set(**{desc:value})

        def on_EQEreplot(change):
            # change info
            _fast = True
            if type(change) is widgets.widgets.widget_button.Button:
                owner = change
            else:  # other controls
                owner = change["owner"]  # control
                value = owner.value
            desc = owner.description
            if desc == "Recalc":
                _fast = False  # underscore prefix marks intentional unused reassignment

            # recalculate
            ts = time()
            if desc[:3] == "eta":
                junc, dist = parse("eta{:1d}{:1d}", desc)
                qe_class.LCcorr(junc, dist, value)  # replace one value and recalculate LC
                specname = None
            elif desc == "spec":
                if value in pvc.qe.dfrefspec.columns:
                    specname = value
                    Pspec = pvc.qe.dfrefspec[specname].to_numpy(dtype=np.float64, copy=True)
            else:
                VoutBox.clear_output()
                with VoutBox:
                    print(desc)
                return 0

            with Rout:  # right output device -> light
                # replot
                lines = ax.get_lines()
                for line in lines:
                    linelabel = line.get_label()
                    # with self.debugout: print(linelabel)
                    for i in range(qe_class.njuncs):
                        if linelabel == qe_class.sjuncs[i]:
                            line.set_data(qe_class.wavelength, qe_class.corrEQE[:, i])  # replot

                rlines = rax.get_lines()
                for line in rlines:
                    linelabel = line.get_label()
                    # with self.debugout: print(linelabel)
                    if linelabel in pvc.qe.refnames:
                        if specname is None:  # desc == 'spec'
                            specname = linelabel
                            Pspec = specname
                        else:
                            line.set_data(xspec, Pspec)  # replot spectrum
                            for obj in rax.get_children():
                                if type(obj) is mpl.collections.PolyCollection:  # contours
                                    if obj.get_label() == "fill":
                                        obj.remove()  # remove old fill
                            rax.fill_between(xspec, Pspec, step="mid", alpha=0.2, color="grey", label="fill")
                            line.set(label=specname)  # relabel spectrum

            Jscs = pvc.qe.JintMD(qe_class.corrEQE, qe_class.wavelength.flatten(), Pspec, xspec)
            Jdbs, Egs = qe_class.Jdb(25)
            OP = pvc.qe.PintMD(Pspec, xspec)

            VoutBox.clear_output()
            with VoutBox:
                stext = (specname + " {0:6.2f} W/m2").format(OP)
                print("Eg = ", Egs, " eV")
                print(stext)
                print("Jsc = ", Jscs[0], " mA/cm^2")

            te = time()
            dt = te - ts
            with VoutBox:
                print("Calc Time: {0:>6.2f} s".format(dt))

        # summary line
        VoutBox = widgets.Output()
        VoutBox.layout.height = "70px"
        # with VoutBox: print('Summary')

        # Right output -> EQE plot
        Rout = widgets.Output()
        with Rout:  # output device
            if plt.isinteractive:
                plt.ioff()
                restart = True
            else:
                restart = False
            ax, rax = qe_class.plot(Pspec, ispec, specname, xspec)
            fig = ax.get_figure()
            fig.show()
            rlines = rax.get_lines()
            for line in rlines:
                linelabel = line.get_label()
                if linelabel in pvc.qe.refnames:
                    specname = linelabel
            if restart:
                plt.ion()

        # tandem3T controls
        in_tit = widgets.Label(value="EQE: ", description="title")
        in_name = widgets.Text(value=qe_class.name, description="name", layout=tand_layout, continuous_update=False)
        in_name.observe(on_EQEchange, names="value")  # update values

        in_spec = widgets.Dropdown(value=specname, description="spec", layout=tand_layout, options=pvc.qe.refnames)
        in_spec.observe(on_EQEreplot, names="value")  # update values

        Hui = widgets.HBox([in_tit, in_name, in_spec])
        # in_Rs2T.observe(on_2Tchange,names='value') #update values

        in_eta = []
        elist0 = []
        elist1 = []
        elist2 = []
        # list of eta controls
        for i in range(qe_class.njuncs):
            if i > 0:
                in_eta.append(
                    widgets.FloatSlider(
                        value=qe_class.etas[i, 0],
                        min=-0.2,
                        max=1.5,
                        step=0.001,
                        description="eta" + str(i) + "0",
                        layout=junc_layout,
                        readout_format=".4f",
                    )
                )
                j = len(in_eta) - 1
                elist0.append(in_eta[j])
                in_eta[j].observe(on_EQEreplot, names="value")  # replot
                # if i > 1:
                in_eta.append(
                    widgets.FloatSlider(
                        value=qe_class.etas[i, 1],
                        min=-0.2,
                        max=1.5,
                        step=0.001,
                        description="eta" + str(i) + "1",
                        layout=junc_layout,
                        readout_format=".4f",
                    )
                )
                j = len(in_eta) - 1
                elist1.append(in_eta[j])
                in_eta[j].observe(on_EQEreplot, names="value")  # replot
                if i > 1:
                    in_eta[j].observe(on_EQEreplot, names="value")  # replot
                else:
                    in_eta[j].disabled = True
                # if i > 2:
                in_eta.append(
                    widgets.FloatSlider(
                        value=qe_class.etas[i, 2],
                        min=-0.2,
                        max=1.5,
                        step=0.001,
                        description="eta" + str(i) + "2",
                        layout=junc_layout,
                        readout_format=".4f",
                    )
                )
                j = len(in_eta) - 1
                elist2.append(in_eta[j])
                if i > 2:
                    in_eta[j].observe(on_EQEreplot, names="value")  # replot
                else:
                    in_eta[j].disabled = True
        etaui0 = widgets.HBox(elist0)
        etaui1 = widgets.HBox(elist1)
        etaui2 = widgets.HBox(elist2)

        # in_Rs2T.observe(on_2Treplot,names='value')  #replot
        # in_2Tbut.on_click(on_2Treplot)  #replot

        # EQE_ui = widgets.HBox(clist)
        # eta_ui = widgets.HBox(jui)

        ui = widgets.VBox([Rout, VoutBox, Hui, etaui0, etaui1, etaui2])
        self.ui = ui
        # in_2Tbut.click() #fill in MPP values

        # return entire user interface, dark and light graph axes for tweaking
        return ui, ax, rax

    def controls_2T(self, class_to_plot):
        """
        use interactive_output for GUI in IPython
        """
        tand_layout = widgets.Layout(width="300px", height="40px")
        vout_layout = widgets.Layout(width="180px", height="40px")
        junc_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")
        _multi_layout = widgets.Layout(display="flex", flex_flow="row", justify_content="space-around")

        replot_types = [
            widgets.widgets.widget_float.BoundedFloatText,
            widgets.widgets.widget_int.BoundedIntText,
            widgets.widgets.widget_int.IntSlider,
            widgets.widgets.widget_float.FloatSlider,
            widgets.widgets.widget_float.FloatLogSlider,
        ]

        def on_2Tchange(change):
            # function for changing values
            _old = change["old"]  # old value (unused -- kept for callback signature documentation)
            _new = change["new"]  # new value (unused)
            owner = change["owner"]  # control
            value = owner.value
            desc = owner.description
            with self.debugout:
                print("Mcontrol: " + desc + "->", value)
            class_to_plot.set(**{desc: value})

        def on_2Treplot(change):
            # change info
            fast = True
            if type(change) is widgets.widgets.widget_button.Button:
                owner = change
                desc = owner.description
            else:  # other controls
                owner = change["owner"]  # control
            desc = owner.description
            if desc == "Recalc":
                fast = False
            elif desc == "savefig":
                fast = False

            # recalculate
            ts = time()
            Idark, Vdark, Vdarkmid = self.calcDark(class_to_plot)
            Vlight, Ilight, Plight, Vlightmid, MPP = self.calcLight(class_to_plot, fast=fast)
            Voc = MPP["Voc"]
            Imax = class_to_plot.Imaxrev()
            Eg_list = class_to_plot.proplist("Eg")  # list of Eg
            Egmax = sum(Eg_list)
            scale = 1000.0
            fmtstr = "Fit:  Voc = {0:>5.3f} V, Isc = {1:>6.2f} mA, FF = {2:>4.1f}%, "
            fmtstr += "Pmp = {3:>5.1f} mW, Vmp = {4:>5.3f} V, Imp = {5:>6.2f} mA"
            fmtstr += ", Eff = {6:>5.2f} %"
            outstr = fmtstr.format(
                MPP["Voc"],
                MPP["Isc"] * scale,
                MPP["FF"] * 100,
                MPP["Pmp"] * scale,
                MPP["Vmp"],
                MPP["Imp"] * scale,
                (MPP["Pmp"] * scale / class_to_plot.lightarea),
            )

            # out_Voc.value = ('{0:>7.3f} V'.format(MPP['Voc']))
            # out_Isc.value = ('{0:>7.2f} mA'.format(MPP['Isc']*scale))
            # out_FF.value = ('{0:>7.1f}%'.format(MPP['FF']*100))
            # out_Pmp.value = ('{0:>7.1f} mW'.format(MPP['Pmp']*scale))
            # out_Vmp.value = ('{0:>7.3f} V'.format(MPP['Vmp']))
            # out_Imp.value = ('{0:>7.2f} mA'.format(MPP['Imp']*scale))

            VoutBox.clear_output()
            with VoutBox:
                print(outstr)

            with Lout:  # left output device -> dark
                # replot
                if desc == "Eg":
                    dax.set_xlim(right=Egmax * 1.1)
                    dax.set_ylim(np.nanmin(abs(Idark)), np.nanmax(abs(Idark)))

                lines = dax.get_lines()
                for line in lines:
                    linelabel = line.get_label()
                    if linelabel in ["pdark", "ndark"]:
                        if linelabel.startswith("p"):
                            line.set_data(Vdark, Idark)
                        elif linelabel.startswith("n"):
                            line.set_data(Vdark, -Idark)
                    elif linelabel.find("junction") >= 0:  # pjunction0, njunction0, etc
                        for junc in range(class_to_plot.njuncs):
                            if linelabel.endswith("junction" + str(junc)):
                                if linelabel.startswith("p"):
                                    line.set_data(Vdarkmid[:, junc], Idark)
                                elif linelabel.startswith("n"):
                                    line.set_data(Vdarkmid[:, junc], -Idark)

            with Rout:  # right output device -> light
                # replot
                if desc == "Eg":
                    lax.set_xlim(right=max(min(Egmax, Voc * 1.1), 0.1))
                    lax.set_ylim(-Imax * 1.5 * scale, Imax * 1.5 * scale)
                lines = lax.get_lines()
                for line in lines:
                    linelabel = line.get_label()
                    if linelabel.find("dark") >= 0:
                        line.set_data(Vdark, Idark * scale)
                    elif linelabel.find("light") >= 0:
                        line.set_data(Vlight, Ilight * scale)
                    elif linelabel.find("points") >= 0:
                        line.set_data(class_to_plot.Vpoints, class_to_plot.Ipoints * scale)
                    elif linelabel.find("junction") >= 0:  # ljunction0, djunction0, etc
                        for junc in range(class_to_plot.njuncs):
                            if linelabel.endswith("junction" + str(junc)):
                                if linelabel.startswith("d"):
                                    line.set_data(Vdarkmid[:, junc], Idark * scale)
                                elif linelabel.startswith("l"):
                                    line.set_data(Vlightmid[:, junc], Ilight * scale)

                    elif linelabel.find("light") >= 0:
                        line.set_data(Vlight, Ilight * scale)
                    elif linelabel.find("points") >= 0:
                        line.set_data(class_to_plot.Vpoints, class_to_plot.Ipoints * scale)

                if False:
                    Jext_list = class_to_plot.proplist("Jext")  # remember list external photocurrents
                    snote = "T = {0:.1f} C, Rs2T = {1:g} Ohm*cm^2, A = {2:g} cm^2".format(class_to_plot.TC, class_to_plot.Rs2T, class_to_plot.lightarea)
                    snote += "\nEg = " + str(Eg_list) + " eV"
                    snote += "\nJext = " + str(Jext_list * 1000) + " mA/cm^2"
                    snote += "\nVoc = {0:.3f} V, Isc = {1:.2f} mA/cm^2\nFF = {2:.1f}%, Pmp = {3:.1f} mW".format(Voc, MPP["Isc"] * 1000, MPP["FF"] * 100, MPP["Pmp"] * 1000)
                    kids = lax.get_children()
                    for kid in kids:
                        if kid.get_label() == "mpptext":
                            kid.set(text=snote)

            te = time()
            dt = te - ts

            if desc == "savefig":
                outpath = junction.newoutpath(class_to_plot.name)

                strout = str(class_to_plot)
                # strout += "\n\nMPP:" + str(MPP)
                with open(os.path.join(outpath, class_to_plot.name + ".txt"), "wt", encoding="utf8") as fout:
                    fout.write(strout)

                # save mathplotlib graphs
                dax.get_figure().savefig(os.path.join(outpath, "dax.png"))
                lax.get_figure().savefig(os.path.join(outpath, "lax.png"))

                # TODO check for dimensions - export currently fails
                # assemble data into 2D arrays then dataframes for saving
                # npdark = np.column_stack((Idark, Vdark, Vdarkmid))
                # dfdark = pd.DataFrame(data=npdark, index=None, columns=["Idark", "Vdark", "Vdarkmid"])
                # dfdark.to_csv(os.path.join(outpath, "darkfit.csv"), index=False)

                # nplight = np.column_stack((Vlight, Ilight, Plight, Vlightmid))
                # dflight = pd.DataFrame(data=nplight, index=None, columns=["Vlight", "Ilight", "Plight", "Vlightmid"])
                # dflight.to_csv(os.path.join(outpath, "lightfit.csv"), index=False)

                # controls output
                with VoutBox:
                    print("Calc Time: {0:>6.2f} s".format(dt), "saved: " + outpath)
            else:
                with VoutBox:
                    print("Calc Time: {0:>6.2f} s".format(dt))

        # summary line
        VoutBox = widgets.Output()
        VoutBox.layout.height = "40px"
        with VoutBox:
            print("Summary")

        # Left output -> dark
        Lout = widgets.Output()
        with Lout:  # output device
            # print(desc, old, new)
            if plt.isinteractive:
                plt.ioff()
                restart = True
            else:
                restart = False

            dfig, dax = self.plot(class_to_plot, dark=True)
            dfig.set_figheight(4)
            dfig.show()
            if restart:
                plt.ion()

        # Right output -> light
        Rout = widgets.Output()
        with Rout:  # output device
            # print(desc, old, new)
            if plt.isinteractive:
                plt.ioff()
                restart = True
            else:
                restart = False

            lfig, lax = self.plot(class_to_plot, dark=False)
            lfig.set_figheight(4)
            lfig.show()
            if restart:
                plt.ion()

        ToutBox = widgets.HBox([Lout, Rout], layout=junc_layout)

        # numerical outputs (kept for future VoutBox.HBox composition below; the
        # HBox line is currently commented out so prefix names with `_` to
        # satisfy lint without losing the widget definitions).
        _out_Voc = widgets.Text(value="Voc", description="Voc", disabled=True, layout=vout_layout)
        _out_Isc = widgets.Text(value="Isc", description="Isc", disabled=True, layout=vout_layout)
        _out_FF = widgets.Text(value="FF", description="FF", disabled=True, layout=vout_layout)
        _out_Pmp = widgets.Text(value="Pmp", description="Pmp", disabled=True, layout=vout_layout)
        _out_Vmp = widgets.Text(value="Vmp", description="Vmp", disabled=True, layout=vout_layout)
        _out_Imp = widgets.Text(value="Imp", description="Imp", disabled=True, layout=vout_layout)
        # VoutBox = widgets.HBox([in_tit,out_Voc,out_Isc,out_FF,out_Pmp,out_Vmp,out_Imp])

        # tandem3T controls
        in_tit = widgets.Label(value="Multi2T: ", description="title")
        in_name = widgets.Text(value=class_to_plot.name, description="name", layout=tand_layout, continuous_update=False)
        in_Rs2T = widgets.FloatLogSlider(value=class_to_plot.Rs2T, base=10, min=-6, max=3, step=0.01, description="Rs2T", layout=tand_layout, readout_format=".2e")
        in_2Tbut = widgets.Button(description="Recalc", button_style="success", tooltip="slow calculations")
        in_savefig = widgets.Button(description="savefig", button_style="success", tooltip="save figures")
        _tand_dict = {"name": in_name, "Rs2T": in_Rs2T}
        # tandout = widgets.interactive_output(class_to_plot.set, tand_dict)
        tand_ui = widgets.HBox([in_tit, in_name, in_Rs2T, in_2Tbut, in_savefig])

        in_name.observe(on_2Tchange, names="value")  # update values
        in_Rs2T.observe(on_2Tchange, names="value")  # update values

        jui = []
        # list of junction controls
        for i in range(class_to_plot.njuncs):
            # jui.append(class_to_plot.j[i].controls())
            jui.append(self.controls_junction(class_to_plot.j[i]))
            kids = jui[i].children
            for cntrl in kids:
                if type(cntrl) in replot_types:
                    cntrl.observe(on_2Treplot, names="value")  # replot
        in_Rs2T.observe(on_2Treplot, names="value")  # replot
        in_2Tbut.on_click(on_2Treplot)  # replot
        in_savefig.on_click(on_2Treplot)  # replot

        junc_ui = widgets.HBox(jui)

        ui = widgets.VBox([ToutBox, VoutBox, tand_ui, junc_ui])
        self.ui = ui
        self.dax = dax
        self.lax = lax
        in_2Tbut.click()  # fill in MPP values

        # return entire user interface, dark and light graph axes for tweaking
        return ui, dax, lax

    def calcDark(self, class_to_plot, hilog=3, pdec=5, timer=False):
        # calc dark IV
        ts = time()
        Jext_list = class_to_plot.proplist("Jext")  # remember list external photocurrents
        class_to_plot.set(Jext=0.0, JLC=0.0)  # turn lights off but don't update controls
        Imax = class_to_plot.Imaxrev()  # in dark
        lolog = math.floor(np.log10(Imax)) - 5
        dpnts = (hilog - lolog) * pdec + 1
        Ifor = np.logspace(hilog, lolog, num=dpnts)
        Irev = np.logspace(lolog, hilog, num=dpnts) * (-1)
        Idark = np.concatenate((Ifor, Irev), axis=None)
        dpnts = Idark.size  # redefine
        Vdark = np.full(dpnts, np.nan, dtype=np.float64)  # Vtotal
        Vdarkmid = np.full((dpnts, class_to_plot.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
        for ii, I in enumerate(Idark):  # noqa: E741 (I is the PV convention for current)
            Vdark[ii] = class_to_plot.V2T(I)  # also sets class_to_plot.Vmid[i]
            for junc in range(class_to_plot.njuncs):
                Vdarkmid[ii, junc] = class_to_plot.Vmid[junc]
        class_to_plot.set(Jext=Jext_list, JLC=0.0)  # turn lights back on but don't update controls
        te = time()
        ds = te - ts
        if timer:
            print(f"dark {ds:2.4f} s")

        return Idark, Vdark, Vdarkmid

    def calcLight(self, class_to_plot, pnts=21, Vmin=-0.5, timer=False, fast=False):
        # calc light IV
        _Jext_list = class_to_plot.proplist("Jext")  # remember list external photocurrents (unused; kept for parity with calcDark/plot)
        _areas = class_to_plot.proplist("lightarea")  # list of junction areas (unused; see commented-out Imax computation below)
        # Imax = max([j*a for j,a in zip(Jext_list,areas)])
        Imax = class_to_plot.Imaxrev()
        Eg_list = class_to_plot.proplist("Eg")  # list of Eg
        _Egmax = sum(Eg_list)

        # ndarray functions
        _V2Tvect = np.vectorize(class_to_plot.V2T)  # unused (vectorised paths are inlined below)
        _I2Tvect = np.vectorize(class_to_plot.I2T)  # unused

        MPP = class_to_plot.MPP()  # calculate all just once
        Voc = MPP["Voc"]

        # vertical portion
        ts = time()
        IxI = np.linspace(-Imax, Imax * 2, pnts)
        # VxI = V2Tvect(IxI)
        VxI = np.full(pnts, np.nan, dtype=np.float64)  # Vtotal
        VmidxI = np.full((pnts, class_to_plot.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
        for ii, I in enumerate(IxI):  # noqa: E741 (I is the PV convention for current)
            VxI[ii] = class_to_plot.V2T(I)  # also sets class_to_plot.Vmid[i]
            for junc in range(class_to_plot.njuncs):
                VmidxI[ii, junc] = class_to_plot.Vmid[junc]
        te = time()
        dsI = te - ts
        if timer:
            print(f"lightI {dsI:2.4f} s")

        if fast:
            Vlight = VxI
            Ilight = IxI
            Vlightmid = VmidxI
        else:
            # horizonal portion slow part
            ts = time()
            VxV = np.linspace(Vmin, Voc, pnts)
            # IxV = I2Tvect(VxV)
            IxV = np.full(pnts, np.nan, dtype=np.float64)  # Vtotal
            VmidxV = np.full((pnts, class_to_plot.njuncs), np.nan, dtype=np.float64)  # Vmid[pnt, junc]
            for ii, V in enumerate(VxV):
                IxV[ii] = class_to_plot.I2T(V)  # also sets class_to_plot.Vmid[i]
                for junc in range(class_to_plot.njuncs):
                    VmidxV[ii, junc] = class_to_plot.Vmid[junc]
            te = time()
            dsV = te - ts
            if timer:
                print(f"lightV {dsV:2.4f} s")
            # combine
            Vboth = np.concatenate((VxV, VxI), axis=None)
            Iboth = np.concatenate((IxV, IxI), axis=None)
            Vbothmid = np.concatenate((VmidxV, VmidxI), axis=0)
            # sort
            p = np.argsort(Vboth)
            Vlight = Vboth[p]
            Ilight = Iboth[p]
            Vlightmid = []
            for junc in range(class_to_plot.njuncs):
                Vlightmid.append(np.take_along_axis(Vbothmid[:, junc], p, axis=0))
            Vlightmid = np.transpose(np.array(Vlightmid))

        Plight = np.array([(-v * j) for v, j in zip(Vlight, Ilight)])
        Vlight = np.array(Vlight)
        Ilight = np.array(Ilight)

        return Vlight, Ilight, Plight, Vlightmid, MPP

    def plot(self, class_to_plot, title="", pplot=False, dark=None, pnts=21, Vmin=-0.5, lolog=-8, hilog=7, pdec=5, size="x-large"):
        # plot a light IV of Multi2T

        Jext_list = class_to_plot.proplist("Jext")  # remember list external photocurrents
        _areas = class_to_plot.proplist("lightarea")  # list of junction areas (unused; kept for parity)
        # Imax = max([j*a for j,a in zip(Jext_list,areas)])
        Imax = class_to_plot.Imaxrev()
        Eg_list = class_to_plot.proplist("Eg")  # list of Eg
        Egmax = sum(Eg_list)
        scale = 1000.0

        # ndarray functions
        _V2Tvect = np.vectorize(class_to_plot.V2T)  # unused (vectorised paths inlined inside calcDark/calcLight)
        _I2Tvect = np.vectorize(class_to_plot.I2T)  # unused

        if class_to_plot.name:
            title += class_to_plot.name

        if dark is None:
            if math.isclose(class_to_plot.Isc(), 0.0, abs_tol=1e-6):
                dark = True
            else:
                dark = False

        # calc dark IV
        Idark, Vdark, Vdarkmid = self.calcDark(class_to_plot)

        if not dark:
            # calc light IV
            Vlight, Ilight, Plight, Vlightmid, MPP = self.calcLight(class_to_plot)
            Voc = MPP["Voc"]

        if dark:
            # dark plot
            dfig, dax = plt.subplots()
            dax.set_prop_cycle(color=class_to_plot.junctioncolors[class_to_plot.njuncs])
            for junc in range(Vdarkmid.shape[1]):  # plot Vdiode of each junction
                plns = dax.plot(Vdarkmid[:, junc], Idark, lw=2, label="pjunction" + str(junc))
                dax.plot(Vdarkmid[:, junc], -Idark, lw=2, c=plns[0].get_color(), label="njunction" + str(junc))

            dax.plot(Vdark, Idark, lw=2, c="black", label="pdark")  # IV curve
            dax.plot(Vdark, -Idark, lw=2, c="black", label="ndark")  # IV curve

            dax.set_yscale("log")  # logscale
            dax.set_autoscaley_on(True)
            dax.set_xlim(Vmin, Egmax * 1.1)
            dax.grid(color="gray")
            dax.set_title(class_to_plot.name + " Dark", size=size)  # Add a title to the axes.
            dax.set_xlabel("Voltage (V)", size=size)  # Add an x-label to the axes.
            dax.set_ylabel("Current (A)", size=size)  # Add a y-label to the axes.
            # dax.legend()
            return dfig, dax

        else:
            # light plot
            lfig, lax = plt.subplots()
            lax.set_prop_cycle(color=class_to_plot.junctioncolors[class_to_plot.njuncs])
            if class_to_plot.njuncs > 1:
                for junc in range(class_to_plot.njuncs):  # plot Vdiode of each junction
                    dlns = lax.plot(Vdarkmid[:, junc], Idark * scale, marker="", ls="--", label="djunction" + str(junc))
                    lax.plot(Vlightmid[:, junc], Ilight * scale, marker="", ls="-", c=dlns[0].get_color(), label="ljunction" + str(junc))

            lax.plot(Vdark, Idark * scale, lw=2, ls="--", c="black", label="dark")  # dark IV curve
            lax.plot(Vlight, Ilight * scale, lw=2, c="black", label="light")  # IV curve
            lax.plot(class_to_plot.Vpoints, class_to_plot.Ipoints * scale, marker="x", ls="", ms=12, c="black", label="points")  # special points
            if pplot:  # power curve
                laxr = lax.twinx()
                laxr.plot(Vlight, Plight * scale, ls="--", c="cyan", zorder=0, label="power")
                laxr.set_ylabel("Power (mW)", c="cyan")
            lax.set_xlim((Vmin - 0.1), max(min(Egmax, Voc * 1.1), 0.1))
            lax.set_ylim(-Imax * 1.5 * scale, Imax * 1.5 * scale)
            lax.set_title(class_to_plot.name + " Light", size=size)  # Add a title to the axes.
            lax.set_xlabel("Voltage (V)", size=size)  # Add an x-label to the axes.
            lax.set_ylabel("Current (mA)", size=size)  # Add a y-label to the axes.
            lax.axvline(0, ls="--", c="gray", label="_vline")
            lax.axhline(0, ls="--", c="gray", label="_hline")
            # lax.legend()

            if False:
                # annotate
                snote = "T = {0:.1f} C, Rs2T = {1:g} Ohm*cm^2, A = {2:g} cm^2".format(class_to_plot.TC, class_to_plot.Rs2T, class_to_plot.lightarea)
                snote += "\nEg = " + str(Eg_list) + " eV"
                snote += "\nJext = " + str(Jext_list * 1000) + " mA/cm^2"
                snote += "\nVoc = {0:.3f} V, Isc = {1:.2f} mA/cm^2\nFF = {2:.1f}%, Pmp = {3:.1f} mW".format(Voc, MPP["Isc"] * 1000, MPP["FF"] * 100, MPP["Pmp"] * 1000)

                # lax.text(Vmin+0.1,Imax/2,snote,zorder=5,bbox=dict(facecolor='white'))
                lax.text(
                    0.05,
                    0.95,
                    snote,
                    verticalalignment="top",
                    label="mpptext",
                    bbox=dict(facecolor="white"),
                    transform=lax.transAxes,
                )
            return lfig, lax

    def update_junction(self, junction):
        # update Junction self.ui controls
        if self.ui:  # junction user interface has been created
            if junction.RBB_dict:
                if junction.RBB_dict["method"]:
                    RBB_keys = list(junction.RBB_dict.keys())
                else:
                    RBB_keys = []

            cntrls = self.ui.children
            for cntrl in cntrls:
                desc = cntrl.trait_values().get("description", "nodesc")  # control description
                cval = cntrl.trait_values().get("value", "noval")  # control value
                if desc == "nodesc" or cval == "noval":
                    break
                elif desc.endswith("]") and desc.find("[") > 0:
                    key, ind = parse("{}[{:d}]", desc)
                else:
                    key = desc
                    ind = None

                if key in junction.ATTR:  # Junction scalar controls to update
                    attrval = getattr(junction, key)  # current value of attribute
                    if cval != attrval:
                        with self.debugout:
                            print("Jupdate: " + desc, attrval)
                        cntrl.value = attrval
                elif key in junction.ARY_ATTR:  # Junction array controls to update
                    attrval = getattr(junction, key)  # current value of attribute
                    if isinstance(ind, int):
                        if isinstance(attrval, np.ndarray):
                            if cval != attrval[ind]:
                                with self.debugout:
                                    print("Jupdate: " + desc, attrval[ind])
                                cntrl.value = attrval[ind]
                elif key in RBB_keys:
                    attrval = junction.RBB_dict[key]
                    if cval != attrval:
                        with self.debugout:
                            print("Jupdate: " + desc, attrval)
                        cntrl.value = attrval

    def update_Multi2T(self):
        # update Multi2T self.ui controls with manually entered values

        for junc in self.class_to_plot.j:
            self.update_junction(junc)

        if self.ui:  # Multi2T user interface has been created
            Boxes = self.ui.children
            for cntrl in Boxes[2].children:  # Multi2T controls
                desc = cntrl.trait_values().get("description", "nodesc")  # does not fail when not present
                cval = cntrl.trait_values().get("value", "noval")  # does not fail when not present
                if desc in ["name", "Rs2T"]:  # Multi2T controls to update
                    key = desc
                    attrval = getattr(self.class_to_plot, key)  # current value of attribute
                    if cval != attrval:
                        # with self.debugout:
                        #     print("Mupdate: " + key, attrval)
                        cntrl.value = attrval
                if desc == "Recalc":
                    cntrl.click()  # click button

    def update_3T(self):
        # update Tandem3T self.ui controls

        # for junc in self.j:
        for junc in [self.class_to_plot.top, self.class_to_plot.bot]:  # two junctions
            self.update_junction(junc)

        if self.ui:  # Tandem3T user interface has been created
            Boxes = self.ui.children
            for cntrl in Boxes[2].children:  # Multi2T controls
                desc = cntrl.trait_values().get("description", "nodesc")  # does not fail when not present
                cval = cntrl.trait_values().get("value", "noval")  # does not fail when not present
                if desc in ["name", "Rz"]:  # Multi2T controls to update
                    key = desc
                    attrval = getattr(self.class_to_plot, key)  # current value of attribute
                    if cval != attrval:
                        # with self.debugout:
                        #     print("Tupdate: " + key, attrval)
                        cntrl.value = attrval
                if desc == "Recalc":
                    cntrl.click()  # click button
