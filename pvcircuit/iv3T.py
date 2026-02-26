# -*- coding: utf-8 -*-
"""
This is the PVcircuit Package.
    pvcircuit.IV3T()       # many forms of operational conditions of 3T tandems
"""

import copy
import math  # simple math
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt  # plotting
import numpy as np  # arrays
import pandas as pd  # data frames

# conversion matrices
SQ2 = math.sqrt(2.0)
SQ3 = math.sqrt(3.0)
SQ6 = math.sqrt(6.0)
# note all matrices tranposed relative to Igor entry
M_isoC = np.transpose(np.array([[SQ3, 1, SQ2], [0, 2, -SQ2], [-SQ3, 1, SQ2]]))  # two rotations
M_isoC /= SQ6
M_isoB = np.transpose(np.array([[1, 0, 0], [0, 0, 0], [0, -1, 0]]))  # two rotations
dev2hex = M_isoB @ M_isoC  # octant viewpoint
hex2dev = dev2hex  # wrong but placeholder
# hex2dev = np.linalg.inv(dev2hex)   #inverse of M_isometric....singular
CR_Vload2dev = np.transpose(np.array([[1, -1, 0], [-1, 0, 1], [0, 0, 0]]))
CR_Iload2dev = np.transpose(np.array([[-1, 1, 0], [-1, 0, 1], [0, 0, 0]]))
CR_Vdev2load = np.transpose(np.array([[0, 0, 0], [-1, 0, 0], [0, 1, 0]]))
CR_Idev2load = np.transpose(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))
CT_Vload2dev = np.transpose(np.array([[0, 1, -1], [1, -1, 0], [0, 0, 0]]))
CT_Iload2dev = np.transpose(np.array([[1, 0, -1], [0, 1, -1], [0, 0, 0]]))
CT_Vdev2load = np.transpose(np.array([[0, 1, 0], [0, 0, 0], [-1, 0, 0]]))
CT_Idev2load = np.transpose(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
CZ_Vload2dev = np.transpose(np.array([[0, 1, -1], [-1, 0, 1], [0, 0, 0]]))
CZ_Iload2dev = np.transpose(np.array([[1, -1, 0], [0, -1, 1], [0, 0, 0]]))
CZ_Vdev2load = np.transpose(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]))
CZ_Idev2load = np.transpose(np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]]))
meas_dict = {
    "CZ": {"VA": "Vrz", "VB": "-Vzt", "IA": "Iro", "IB": "Ito"},
    "CR": {"VA": "-Vrz", "VB": "Vtr", "IA": "Izo", "IB": "Ito"},
    "CT": {"VA": "-Vtr", "VB": "Vzt", "IA": "Iro", "IB": "Izo"},
    "CF": {"VA": "-Vfr", "VB": "Vzf", "IA": "Iro", "IB": "Izo"},
}

"""
CR:A = ZR,B = TR
	Jro = -(JB+JA)
	Jto = JB
	Jzo = JA
	Vzt = VA-VB
	Vtr = VB
	Vrz = -VA
CF:A = RF,B = ZF
CT:A = RT,B = ZT
	Jro = JA
	Jto = -(JB+JA)
	Jzo = JB
	Vzt = VB
	Vtr = -VA
	Vrz = VA-VB
CZ:A = RZ, B = TZ
	Jro = JA
	Jto = JB
	Jzo = -(JB+JA)
	Vzt = -VB
	Vtr = VB-VA
	Vrz = VA
"""


class IV3T(object):
    """
    operational state of 3T tandem
    (Iro,Izo,Ito), (Vzt,Vrz,Vtr), etc
    """

    arraykeys = ["Iro", "Izo", "Ito", "Vzt", "Vrz", "Vtr", "IA", "IB", "VA", "VB", "Ptot", "Ixhex", "Iyhex", "Vxhex", "Vyhex"]
    Idevlist = ["Iro", "Izo", "Ito"]
    Vdevlist = ["Vzt", "Vrz", "Vtr"]

    def __init__(self, name="iv3T", meastype="CZ", shape=0, fillname="", area=1):

        self.name = name
        self.meastype = meastype
        self.area = area  # area (cm2) used to calculate J <-> I/A

        # force shape to be tuple
        if np.ndim(shape) == 0:  # not iterable
            self.shape = (shape,)
        else:  # iterable
            self.shape = tuple(shape)

        for key in self.arraykeys:
            setattr(self, key, np.full(shape, np.nan, dtype=np.float64))

        size = getattr(self, key).size  # size of the last arraykey
        self.names = [fillname for i in range(size)]  # list of names of point in flat arrays
        #### future use string numpy for self.names instead of list  ####

    def copy(self):
        """
        create a separate complete copy of a IV3T
        """
        return copy.copy(self)

    def __str__(self):
        #   string description of IV3T object
        strout = self.name + ": <pvcircuit.iv3T.IV3T class>"
        attr_list = dict(self.__dict__.items()).copy()  # all attributes
        nshape = len(str(self.shape))  # number letters in largest index string
        if len(self.names) > 0:
            nnames = len(max(self.names, key=len))  # number of letters in larges names[i] string
        else:
            nnames = 0
        nshape = max(nshape, nnames)
        title = "#".center(nshape)
        sizes = self.sizes(self.arraykeys)
        nmin, nmax = sizes
        width = 9
        prntmax = 10
        fmtV = "{0:^" + str(width) + ".3f}"
        fmtI = "{0:^" + str(width) + ".2f}"

        if "x" in attr_list:
            attr_list["x"] = (np.min(self.x), np.max(self.x))
        if "y" in attr_list:
            attr_list["y"] = (np.min(self.y), np.max(self.y))

        del attr_list["names"]
        for key in self.arraykeys:
            if key in attr_list:
                del attr_list[key]

            if key[:1] == "V":
                title += key.center(width)
            else:  # append 'm'
                title += ("m" + key).center(width)

        strout += "\n" + str(attr_list)
        strout += "\nsizes" + str(sizes)
        strout += "\n\n" + title

        imax = 1
        for i, index in enumerate(np.ndindex(self.shape)):
            strline = ""
            if hasattr(self, "names"):
                if i < len(self.names) and self.names[i]:
                    strline += "\n" + self.names[i].center(nshape)
                else:
                    strline += "\n" + str(index).center(nshape)
            else:
                strline += "\n" + str(index).center(nshape)

            for key in self.arraykeys:
                array = getattr(self, key)
                imax = max(imax, array.size)
                if i < array.size:
                    if key[:1] == "V":
                        # sval = fmtV.format(array.flat[i])
                        sval = fmtV.format(array[index])
                    else:  # print mA or mW
                        sval = fmtI.format(1000 * array[index])
                else:
                    sval = "".center(width)
                strline += sval

            if i < prntmax:  # head
                strout += strline
            elif i > imax - prntmax:  # tail
                strout += strline
            elif i == prntmax:
                strout += "\n\n"

        # Clean string from numpy types present after numpy > 2.x.x
        strout = re.sub(r"np\.(?:int|float|int64|float64)\(([^)]+)\)", r"\1", strout)
        # replace negative 0
        strout = re.sub(r'-0\.(0+)\b', r'0.\1 ', strout)

        return strout

    def __repr__(self):
        return str(self)

    def set(self, **kwargs):
        """
        set variables
        """
        for key, value in kwargs.items():
            if key in set(self.arraykeys):  # this should be numpy.ndarray
                if np.isscalar(value):  # value = scalar
                    # shape = getattr(self, key).size
                    setattr(self, key, np.full(self.shape, value, dtype=np.float64))
                else:  # value = full array
                    setattr(self, key, np.array(value))

            # raise error if the key is not in the class attributes
            elif key not in list(self.__dict__.keys()):
                raise ValueError(f"invalid class attribute {key}")
            else:
                setattr(self, key, value)

    def line(self, xkey, x0, x1, xn, ykey, yconstraint, log=False):
        """
        create a 1D ndarray on xkey with evenly spaced values (log=False)
        or log spaced values (log=True) +10^x0 to +10^x1 and -10^x0 to -10^x1
        ykey is constrained to xkey with eval expression using 'x'
        """

        if log:
            lolog = x0
            hilog = x1
            Ifor = np.logspace(hilog, lolog, num=xn, dtype=np.float64)
            Irev = np.logspace(lolog, hilog, num=xn, dtype=np.float64) * (-1)
            x = np.concatenate((Ifor, Irev), axis=None)
        else:
            x = np.linspace(x0, x1, xn, dtype=np.float64)

        shape = x.shape

        for key in self.arraykeys:
            if key == xkey:
                setattr(self, key, x)
            else:
                setattr(self, key, np.full(shape, np.nan, dtype=np.float64))

        y = eval(yconstraint)
        if type(y) is not np.array:
            y = np.full(shape, y, dtype=np.float64)

        setattr(self, ykey, y)
        self.kirchhoff([xkey, ykey])  # calculate everything

        # remember
        self.shape = shape
        self.xkey = xkey
        self.ykey = ykey
        self.x = x
        self.y = y

    def box(self, xkey, x0, x1, xn, ykey, y0, y1, yn):
        """
        create a 2D ndarray for xkey and ykey with shape (xn, yn)
        with evenly spaced values
        """
        shape = (yn, xn)
        x = np.linspace(x0, x1, xn, dtype=np.float64)
        y = np.linspace(y0, y1, yn, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        for key in self.arraykeys:
            if key == xkey:
                setattr(self, key, xx)
            elif key == ykey:
                setattr(self, key, yy)
            else:
                setattr(self, key, np.full(shape, np.nan, dtype=np.float64))

        self.kirchhoff([xkey, ykey])  # calculate everything

        # remember
        self.shape = shape
        self.xkey = xkey
        self.ykey = ykey
        self.x = x
        self.y = y

    def hexgrid(self, ax, VorI, step, xn=2, maxlines=10):
        """
        add hexagonal grid lines to axis
        range determined from self box IV3T object
        """
        xhex = VorI + "xhex"
        yhex = VorI + "yhex"
        if VorI == "V":
            devlist = IV3T.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
            scale = 1.0
        else:
            devlist = IV3T.Idevlist.copy()  # ['Iro','Izo','Ito']
            scale = 1000.0

        hex3T = IV3T(name="hexgrid", area=self.area)  # create IV3T object

        for ykey in devlist:
            others = devlist.copy()
            others.remove(ykey)
            xkey = others[0]
            zkey = others[1]
            xd = getattr(self, xkey)
            x0 = np.nanmin(xd) * scale
            x1 = np.nanmax(xd) * scale
            yd = getattr(self, ykey)
            y0 = np.nanmin(yd) * scale
            y1 = np.nanmax(yd) * scale
            zd = getattr(self, zkey)
            z0 = np.nanmin(zd) * scale
            z1 = np.nanmax(zd) * scale

            numlines = 0
            for ycon in [step * i for i in range(-maxlines, maxlines) if y0 <= step * i <= y1]:

                xalt0 = -ycon - z1
                xalt1 = -ycon - z0
                hex3T.line(xkey, max(x0, xalt0), min(x1, xalt1), xn, ykey, str(ycon))  # define a line
                hex3T.convert(VorI=VorI, oper="dev2hex")
                xx = getattr(hex3T, xhex)
                yy = getattr(hex3T, yhex)
                # label
                ylab = ykey + " = " + str(ycon)
                yindex = devlist.index(ykey)
                if yindex == 0:
                    rot = 60
                    nn = 2.25
                elif yindex == 2:
                    rot = -60
                    nn = 1.75
                else:
                    rot = 0
                    nn = 2.25
                xt = (np.nanmin(xx) + np.nanmax(xx) * (nn - 1.0)) / nn
                yt = (np.nanmin(yy) + np.nanmax(yy) * (nn - 1.0)) / nn
                ax.text(xt, yt, ylab, rotation=rot, rotation_mode="anchor", clip_on=True)  # ,bbox=dict(facecolor='white'))
                # plot
                if ycon == 0:  # origin
                    ax.plot(xx, yy, c="gray", label="_" + ylab)
                else:
                    ax.plot(xx, yy, ls=(0, (1, 3)), c="gray", label="_" + ylab)

    def nanpnt(self, index):
        """
        make indexed point in each keyarray an nan
        index tuple i = (i, ) or (i,j)
        """
        for key in self.arraykeys:
            array = getattr(self, key)
            array[index] = np.nan

    def MPP(self, name=""):
        """
        find max power point of existing IV3T class datapoints
        """

        if self.Ptot.size == 0:
            return 0

        temp3T = IV3T(name="MPP" + name, shape=1, meastype=self.meastype, area=self.area)

        nmax = np.argmax(self.Ptot)

        for key in self.arraykeys:
            sarray = getattr(self, key)
            tarray = getattr(temp3T, key)
            tarray[0] = sarray.flat[nmax]

        return temp3T

    def sizes(self, klist):
        # array length
        # for all use .sizes(self.arraykeys)

        sizes = []  # initialize list of array lengths
        allowlist = [key for key in klist if key in self.arraykeys]
        # ignore items that are not array attributes
        for key in allowlist:
            num = getattr(self, key).size
            sizes.append(num)

        nmin = np.amin(sizes)
        nmax = np.amax(sizes)
        return (nmin, nmax)

    def resize(self, shape, fillname=""):
        # resize arrays and clear values

        # force shape to be tuple
        if np.ndim(shape) == 0:  # not iterable
            self.shape = (shape,)
        else:  # iterable
            self.shape = tuple(shape)

        for key in self.arraykeys:
            array = getattr(self, key)
            setattr(self, key, np.resize(array, shape))

        oldsize = len(self.names)
        newsize = getattr(self, key).size  # last array

        if newsize > oldsize:
            for i in range(oldsize, newsize):
                self.names.append(fillname)  # append
        elif newsize < oldsize:
            for i in range(oldsize, newsize, -1):
                del self.names[-1]  # delete

        return oldsize, newsize

    def delete(self, ind):
        # remove point(s) from arrays in self

        for key in self.arraykeys:
            array = getattr(self, key)
            oldarray = np.copy(array)
            if oldarray.ndim > 1:
                print("too many dimensions", oldarray.ndim)
            newarray = np.delete(oldarray, ind, 0)
            setattr(self, key, newarray)

        # need to convert self.name to np.array and back to list
        oldarray = np.array(self.names, copy=True, dtype=str)
        newarray = np.delete(oldarray, ind, 0)
        self.names = list(newarray)

        nmin, nmax = self.sizes(self.arraykeys)
        if nmin == nmax:  # everything in inlist has same length
            self.shape = self.Ptot.shape
        else:
            print("not all same size", nmin, nmax)

    def append(self, iv3T):
        # append another iv3T object to self

        nmin, nmax = self.sizes(self.arraykeys)
        if nmin != nmax:
            print(nmin, nmax, "not same size")
            return 1
        addmin, addmax = iv3T.sizes(self.arraykeys)
        newsize = nmax + addmax

        self.resize(newsize, fillname=iv3T.name)  # resize and flatten

        #        for i in range(addmax):
        for i in range(len(iv3T.names)):  # prevent out of range error
            if iv3T.names[i]:  # not null string
                self.names[i + nmax] = iv3T.names[i]  # add names

        for key in self.arraykeys:
            selfarray = getattr(self, key)
            addarray = getattr(iv3T, key)

            for i in range(addmax):
                selfarray[i + nmax] = addarray[i]  # add key values

        if self.meastype != iv3T.meastype:  # correct all load values
            self.convert("V", "dev2load")
            self.convert("I", "dev2load")
            # print('meastypes different', self.meastype , iv3T.meastype)

        return 0

    def sort(self, key):
        # sort a iv3T line based on array key
        # unexpected results for iv3T box
        sortarray = getattr(self, key)
        p = np.argsort(sortarray, kind="stable")

        for key in self.arraykeys:
            array = getattr(self, key)
            oldarray = np.copy(array)
            setattr(self, key, oldarray[p])

        # need to convert self.name to np.array and back to list
        oldarray = np.array(self.names, copy=True, dtype=str)
        self.names = list(oldarray[p])

    def init(self, inlist, outlist):
        # initialize output arrays to nan if input arrays are consistent

        nmin, nmax = self.sizes(inlist)
        if nmin == nmax:  # everything in inlist has same length
            # record shape of inlist
            self.shape = getattr(self, inlist[0]).shape

            # initialize output
            for key in outlist:
                setattr(self, key, np.full(self.shape, np.nan, dtype=np.float64))

            return 0

        else:
            return 4  # inlist arrays not same length

    def kirchhoff(self, two):
        """
        apply or check kirchoff's law on
        [Iro,Izo,Ito] or [Vzt,Vrz,Vtr]
        input 2 or 3 of the device input keys:
            2 -> calculate the third device value from other two knowns
            3 -> check the validity of 3 device parameters
        """

        stwo = set(two)
        if stwo.issubset(set(self.Vdevlist)):
            klist = self.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
        elif stwo.issubset(set(self.Idevlist)):
            klist = self.Idevlist.copy()  # ['Iro','Izo','Ito']
        else:
            # print(str(two) + ' not device parameters')
            return 1  # not Vdev or Idev

        for key in two:
            klist.remove(key)
        if klist:
            scalc = klist[0]  # remaining item to calculate

        ltwo = len(two)
        nmin, nmax = self.sizes(two)
        if nmin == nmax:  # everything in two has same length
            array0 = getattr(self, two[0])
            array1 = getattr(self, two[1])
            if ltwo == 2:  # calc third
                calcarray = -array0 - array1
                setattr(self, scalc, calcarray)
            elif ltwo == 3:  # check
                array2 = getattr(self, two[2])
                akirch = array0 + array1 + array2
                atol = 1e-3 * max(1.0, self.area)
                for i, kzero in enumerate(akirch.flat):
                    if not math.isclose(kzero, 0.0, abs_tol=atol):
                        array0.flat[i] = np.nan
                        array1.flat[i] = np.nan
                        array2.flat[i] = np.nan

        else:
            print(str(two) + " arrays not same length")
            return 3  # klist arrays not same length

        return 0  # success

    def Pcalc(self, oper="dev2load", meastype=None):
        """
        calculate Ptot using oper = 'dev2load' or 'load2dev'
        """

        if meastype != None:  # optionally change the attribute here
            self.meastype = meastype

        if oper == "dev2load":
            devlist = self.Idevlist.copy() + self.Vdevlist.copy()
            # ['Iro', 'Izo', 'Ito', 'Vzt', 'Vrz', 'Vtr'])
            nmin, nmax = self.sizes(devlist)
        elif oper == "load2dev":
            nmin, nmax = self.sizes(["IA", "IB", "VA", "VB"])
        else:
            return 1  # invalid oper

        if nmin == nmax:
            # calc either direction using self.meastype
            self.convert("V", oper, meastype=meastype)
            self.convert("I", oper, meastype=meastype)
            self.convert("V", "dev2hex", meastype=meastype)
            self.convert("I", "dev2hex", meastype=meastype)
        else:
            return 2  # need all input values of same length

        self.Ptot = -self.IA * self.VA - self.IB * self.VB
        np.nan_to_num(self.Ptot, copy=False, nan=-100.0)

        return 0

    def loadlabel(self, load, meastype=None):
        """
        return descriptive axis label for load variables
        add an extra character to swap the loads: 'CRo','CTo','CZo', 'CFo'
        """

        if meastype == None:  # change the attribute here
            meastype = self.meastype

        if "A" not in load.upper() and "B" not in load.upper():
            return load  # not load variable -> don't change

        load = load.upper()  # fix lower case
        if len(meastype) > 2:  # swap loads for alternate measurements, e.g CTo
            if "A" in load:
                swload = load.replace("A", "B")
            if "B" in load:
                swload = load.replace("B", "A")
        else:
            swload = load

        return load + " = " + meas_dict[meastype[0:2]][swload]

    def convert(self, VorI, oper, meastype=None):
        """
        calculate hexagonal coordinates
        VorI: 'V' or 'I'
        oper: 'load2dev', 'dev2load', 'dev2hex', 'hex2dev' (not developed yet)
        can optionally set the meastype here
        meastype: 'CR','CT','CZ','CF'
        add an extra character to swap the loads: 'CRo','CTo','CZo', 'CFo'
        """

        oper = oper.lower()
        VorI = VorI.upper()

        if VorI == "V":
            devlist = self.Vdevlist.copy()  # ['Vzt','Vrz','Vtr']
            loadlist = ["VA", "VB"]
            hexlist = ["Vxhex", "Vyhex"]
        elif VorI == "I":
            devlist = self.Idevlist.copy()  # ['Iro','Izo','Ito']
            loadlist = ["IA", "IB"]
            hexlist = ["Ixhex", "Iyhex"]
        else:
            print("VorI err", VorI)
            return 1  # invalid VorI

        if meastype != None:  # change the attribute here
            self.meastype = meastype

        smatrix = self.meastype[0:2].upper().replace("F", "T") + "_" + VorI + oper

        if len(self.meastype) > 2:  # swap loads for alternate measurements, e.g CTo
            loadlist = loadlist[::-1]
            # print(self.meastype, '->', loadlist, ' swapped')

        if oper == "dev2hex":
            inlist = devlist
            outlist = hexlist
            smatrix = oper
        elif oper == "hex2dev":
            inlist = hexlist
            outlist = devlist
            smatrix = oper
        elif oper == "load2dev":
            inlist = loadlist
            outlist = devlist
        elif oper == "dev2load":
            inlist = devlist
            outlist = loadlist
        else:
            print("oper err", oper)
            return 2  # invalid oper

        try:
            matrix = eval(smatrix)
        except:
            print("matrix err", smatrix)
            return 3  # could not evaluate smatrix

        nmin, nmax = self.sizes(inlist)
        if nmin == nmax:  # everything in klist has same length
            inarray0 = getattr(self, inlist[0])
            inarray1 = getattr(self, inlist[1])
            if len(inlist) == 3:
                inarray2 = getattr(self, inlist[2])
            else:
                inarray2 = np.zeros(self.shape)

            # initialize local empty output to same size
            outarray0 = np.full(self.shape, np.nan, dtype=np.float64)
            outarray1 = np.full(self.shape, np.nan, dtype=np.float64)
            outarray2 = np.full(self.shape, np.nan, dtype=np.float64)
        else:
            return 4  # klist arrays not same length

        i = 0
        for in0, in1, in2 in zip(inarray0.flat, inarray1.flat, inarray2.flat):
            vector_in = np.array([in0, in1, in2])
            vector_out = matrix @ vector_in
            outarray0.flat[i] = vector_out[0]
            outarray1.flat[i] = vector_out[1]
            outarray2.flat[i] = vector_out[2]
            i += 1

        setattr(self, outlist[0], outarray0)
        setattr(self, outlist[1], outarray1)
        if len(outlist) == 3:  # otherwise they are just zeros
            setattr(self, outlist[2], outarray2)

        return 0
    @classmethod
    def from_csv(cls,name, path, fileA, fileB, VorI, meastype, Iscale=1000.0, area=1):
        """
        import csv file as data table into iv3T object
        two 2D arrays with x and y index on top and left
        load variables:
        VA(IA,IB) & VB(IA,IB) .......... VorI='I'
            or
        IA(VA,VB) & IB(VA,VB) .......... VorI='V'
        Iscale converts current mA -> A or mA/cm2-> A
        """

        xkey = VorI + "A"
        ykey = VorI + "B"

        if VorI == "I":
            indscale = Iscale / area  # mA
            dscale = 1.0
        else:
            indscale = 1.0
            dscale = Iscale / area  # mA

        # read into dataframe
        dfA = pd.read_csv(os.path.join(path, fileA), index_col=0)
        dfB = pd.read_csv(os.path.join(path, fileB), index_col=0)

        # y index
        indA = np.array(dfA.index)
        indB = np.array(dfB.index)
        x0 = indA[0]  # first
        x1 = indA[-1]  # last
        xn = len(indA)
        if x0 != indB[0]:
            return 1
        if x1 != indB[-1]:
            return 2
        if xn != len(indB):
            return 3

        # x columns...convert from string labels
        colA = np.float64(np.array(dfA.columns))
        colB = np.float64(np.array(dfB.columns))
        y0 = colA[0]  # first
        y1 = colA[-1]  # last
        yn = len(colA)
        if y0 != colB[0]:
            return 4
        if y1 != colB[-1]:
            return 5
        if yn != len(colB):
            return 6

        # create iv3T class
        iv3T = cls(name=name, meastype=meastype, area=area)
        iv3T.box(xkey, x0 / indscale, x1 / indscale, xn, ykey, y0 / indscale, y1 / indscale, yn)
        # print(xkey, x0/indscale, x1/indscale, xn, ykey, y0/indscale, y1/indscale, yn)

        # measured data
        IorV = "VI".strip(VorI)
        Akey = IorV + "A"
        Bkey = IorV + "B"

        # assign values
        dataA = np.array(dfA).transpose() / dscale
        dataB = np.array(dfB).transpose() / dscale
        # print('A',dataA.shape,Akey)
        # print('B',dataB.shape,Bkey)

        # put into iv3T object
        setattr(iv3T, Akey, dataA)
        setattr(iv3T, Bkey, dataB)

        iv3T.Pcalc(oper="load2dev")  # calc device parameters then Ptot

        return iv3T

    def plot(
        self,
        xkey=None,
        ykey=None,
        zkey="Ptot",
        inplot=None,
        cmap="terrain",
        ccont="black",
        bar=True,
        log=False,
        density=False,
        size="x-large",
    ):
        """
        plot 2D IV3T object
            zkey(xkey,ykey)
        as image if evenly spaced
        or scatter if randomly spaced
        with contours
        """

        # defaults
        if xkey == None:
            xkey = self.xkey
        if ykey == None:
            ykey = self.ykey
        if zkey == None:
            zkey = "Ptot"

        dim = len(self.shape)
        if dim != 2:
            fig, ax = plt.subplots()
            return ax, ["error dim=" + str(dim)]
        if xkey.replace("f", "t") not in self.arraykeys:
            fig, ax = plt.subplots()
            return ax, ["error xkey=" + xkey]
        if ykey.replace("f", "t") not in self.arraykeys:
            fig, ax = plt.subplots()
            return ax, ["error ykey=" + ykey]

        VorI = xkey[0]
        if VorI == "I":
            if density:
                unit = " (mA/cm2)"
                scale = 1000.0 / self.area
            else:
                unit = " (mA)"
                scale = 1000.0
            step = 10
        else:
            unit = " (V)"
            scale = 1.0
            step = 0.5

        z0 = zkey[0]
        if z0 == "P":
            if density:
                zlab = "Power (mW/cm2)"
                zscale = 1000.0 / self.area
            else:
                zlab = "Power (mW)"
                zscale = 1000.0
            lstep = 5.0
        elif z0 == "I":
            if density:
                zlab = zkey + " (mA/cm2)"
                zscale = 1000.0 / self.area
            else:
                zlab = zkey + " (mA)"
                zscale = 1000.0
            lstep = 5.0
        else:
            zlab = zkey + " (V)"
            zscale = 1.0
            lstep = 0.5

        x = self.x * scale  # 1D
        y = self.y * scale  # 1D
        xx = getattr(self, xkey.replace("f", "t")) * scale  # 2D
        yy = getattr(self, ykey.replace("f", "t")) * scale  # 2D
        zz = getattr(self, zkey.replace("f", "t")) * zscale  # 2D
        extent = [np.nanmin(xx), np.nanmax(xx), np.nanmin(yy), np.nanmax(yy)]
        if log:
            zlab = "log(|" + zlab + "|)"
            lz = np.log10(np.abs(zz))
            lstep = 1
            Pmax = np.ceil(np.nanmax(lz) / lstep) * lstep
            Pmin = np.floor(np.nanmin(lz) / lstep) * lstep
            levels = np.arange(Pmin, Pmax + 1, lstep)
        else:
            lstep *= np.ceil(np.nanmax(zz) / 10.0 / lstep)
            Pmax = np.ceil(np.nanmax(zz) / lstep) * lstep
            levels = [ll * lstep for ll in range(10) if ll * lstep <= Pmax]  # for contours

        if len(levels) == 0:
            levels = None

        # print(self.name) # tag this instance
        handles = []
        labels = []
        if inplot:  # start with old plot
            ax, objs = inplot
            fig = ax.get_figure()  # figure from axes
            fig.set_figheight(4)
            # handles, labels = ax.get_legend_handles_labels()  # legend items to be added
            oldlegend = ax.get_legend()
            if type(oldlegend) is mpl.legend.Legend:
                texts = oldlegend.get_texts()
                lines = oldlegend.get_lines()
                for text, line in zip(texts, lines):
                    # print('get',line, text.get_text())
                    handles.append(line)
                    labels.append(text.get_text())
        else:
            fig, ax = plt.subplots()
            # handles, labels = ax.get_legend_handles_labels()  # legend items to be added
            objs = []  # list of objects in plot for further manipulation
            ax.set_aspect(1)
            if "hex" in xkey:
                # isometric hexagonal coordinates
                ax.set_axis_off()  # turn off confusing x-axis and y-axis
                # add hexgrids
                step = np.ceil(max(np.nanmax(xx) - np.nanmin(xx), np.nanmax(yy) - np.nanmin(yy)) / 10.0 / step) * step
                self.hexgrid(ax, VorI, step)
            else:
                # cartisian coordinates
                ax.set_xlabel(self.loadlabel(xkey) + unit, size=size)  # Add an x-label to the axes.
                ax.set_ylabel(self.loadlabel(ykey) + unit, size=size)  # Add a y-label to the axes.
                ax.axhline(0, ls="--", color="gray", label="_hzero")
                ax.axvline(0, ls="--", color="gray", label="_vzero")

        if cmap:  # don't add image if cmap == None
            cmap = copy.copy(plt.cm.get_cmap(cmap))  # start with existing cmap
            cmap.set_under(color="white")  # white for Ptot < 0 and nan
            if xkey == self.xkey and ykey == self.ykey:
                # image if evenly spaced
                if log:
                    imag = ax.imshow(lz, origin="lower", extent=extent, cmap=cmap)
                else:
                    imag = ax.imshow(zz, vmin=0, vmax=Pmax, origin="lower", extent=extent, cmap=cmap)
            else:
                # scatter if randomly spaced
                msize = round(4000 / len(zz))  # marker size to fill in
                imag = ax.scatter(xx, yy, s=msize, c=zz, marker="h", cmap=cmap, vmin=0, vmax=Pmax)
            objs.append(imag)  # output image as object

            if bar == True:
                # colorbar
                cb = plt.colorbar(imag, ax=ax, shrink=0.6, ticks=levels)
                cb.set_label(zlab)
                objs.append(cb)  # output colorbar as ColorBar

        if ccont:  # don't add contour if ccont == None
            # contour
            if log:
                cont = ax.contour(xx, yy, lz, colors=ccont, levels=levels)
            else:
                cont = ax.contour(xx, yy, zz, colors=ccont, levels=levels)
            ax.clabel(cont, inline=True, fontsize=10)
            objs.append(cont)  # output contour as QuadContourSet object
            hands, labs = cont.legend_elements()  # lists of each line in contour
            if self.name in labels:
                handles[labels.index(self.name)]  # change handle of self.name
            else:
                handles.append(hands[0])
                labels.append(self.name)
            ax.legend(handles, labels, title="Contours")

        return ax, objs  # fig = ax.get_figure()

    def addpoints(self, ax, xkey, ykey, density=True, **kwargs):
        # add iv3T points to existing axes
        VorI = xkey[0]
        if VorI == "I":
            if density:
                scale = 1000.0 / self.area
            else:
                scale = 1000.0
        else:  #'V'
            scale = 1.0
        xp = getattr(self, xkey) * scale
        yp = getattr(self, ykey) * scale
        lns = ax.plot(xp, yp, **kwargs)
        return lns[0]  # return first line

    def plotIVslice(self, step=2, log=True, inplots=None, labelplus="", size="x-large"):
        # plot iv slices through box iv3T data
        if len(self.shape) == 2:
            na, nb = self.shape
        else:
            return 1  # must be box
        xkey = self.xkey
        ykey = self.ykey
        scale = 1000.0

        # fig, (Lax, Rax) = plt.subplots(1, 2, constrained_layout=True)
        if inplots == None:
            Lfig, Lax = plt.subplots()  # constrained_layout=True)
            Rfig, Rax = plt.subplots()  # constrained_layout=True)
            kwargs = {"lw": 0, "marker": "o"}  # markers for data
        else:
            Lax, Rax = inplots
            kwargs = {"lw": 1}  # lines for fits

        VorI = xkey[0]
        if VorI == "V":
            Vxkey = xkey
            Vykey = ykey
            Ixkey = xkey.replace("V", "I", 1)
            Iykey = ykey.replace("V", "I", 1)
        else:  # I
            Ixkey = xkey
            Iykey = ykey
            Vxkey = xkey.replace("I", "V", 1)
            Vykey = ykey.replace("I", "V", 1)

        Vxp = getattr(self, Vxkey)
        Vyp = getattr(self, Vykey)

        if log:
            # Ixp = np.log10(np.abs(getattr(self, Ixkey)*scale))
            # Iyp = np.log10(np.abs(getattr(self, Iykey)*scale))
            Ixp = np.abs(getattr(self, Ixkey) * scale)
            Iyp = np.abs(getattr(self, Iykey) * scale)
            Rax.set_ylabel("|" + self.loadlabel(Ixkey) + "| (mA)", size=size)
            Lax.set_ylabel("|" + self.loadlabel(Iykey) + "| (mA)", size=size)
            Rax.set_yscale("log")
            Lax.set_yscale("log")

        else:
            Ixp = getattr(self, Ixkey) * scale
            Iyp = getattr(self, Iykey) * scale
            Rax.set_ylabel(self.loadlabel(Ixkey) + " (mA)", size=size)
            Lax.set_ylabel(self.loadlabel(Iykey) + " (mA)", size=size)
            Lax.axhline(0, ls="--", color="gray", label="_hzero")
            Rax.axhline(0, ls="--", color="gray", label="_hzero")

        Rax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])  # reset color cycle
        for i in range(0, na, step):  # rear
            kwargs["label"] = labelplus + ykey + "={0:.1f}".format(self.y[i])
            Rax.plot(Vxp[i, :], Ixp[i, :], **kwargs)

        Lax.set_prop_cycle(plt.rcParams["axes.prop_cycle"])  # reset color cycle
        for i in range(0, nb, step):  # top
            kwargs["label"] = labelplus + xkey + "={0:.1f}".format(self.x[i])
            Lax.plot(Vyp[:, i], Iyp[:, i], **kwargs)

        if inplots == None:
            Rax.set_xlabel(self.loadlabel(Vxkey) + " (V)", size=size)
            Lax.set_xlabel(self.loadlabel(Vykey) + " (V)", size=size)
            Rax.axvline(0, ls="--", color="gray", label="_vzero")
            Lax.axvline(0, ls="--", color="gray", label="_vzero")
        Rax.legend()  # bbox_to_anchor=(1.05, 1))
        Lax.legend()  # bbox_to_anchor=(1.05, 1))

        return Lax, Rax
