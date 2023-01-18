# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import matplotlib.pyplot as plt


def pltset(
    ticksize=12,
    labelsize=12,
    legendsize=12,
    linewidth=1.5,
    axeswidth=1,
    tickwidth=1,
    ticklength=3,
    tickdirection="in",
    mtickvisible=False,
    fontkind="Times New Roman",
    fontweight=False,
):
    """This is a drawing template function.

    Args:
        ticksize (int, optional): axis ticklabel fontsize. Defaults to 12.
        labelsize (int, optional): label fontsize. Defaults to 12.
        legendsize (int, optional): legend fontsize. Defaults to 12.
        linewidth (float, optional): width of line. Defaults to 1.5.
        axeswidth (int, optional): width of axis line. Defaults to 1.
        tickwidth (int, optional): width of axis tick. Defaults to 1.
        ticklength (int, optional): length of axis tick. Defaults to 3.
        tickdirection (str, optional): axis tick direction. Defaults to "in".
        mtickvisible (bool, optional): whether show minor axis tick. Defaults to False.
        fontkind (str, optional): font family. Defaults to "Times New Roman".
        fontweight (bool, optional): bold font or not. Defaults to False.
    """
    try:
        plt.style.use(["science", "notebook"])
    except Exception:
        pass
        # print("One should install SciencePlots package: pip install SciencePlots")
    plt.rcParams["legend.fontsize"] = legendsize
    plt.rcParams["lines.linewidth"] = linewidth
    plt.rcParams["axes.linewidth"] = axeswidth
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["xtick.major.width"] = tickwidth
    plt.rcParams["ytick.major.width"] = tickwidth
    plt.rcParams["xtick.major.size"] = ticklength
    plt.rcParams["ytick.major.size"] = ticklength
    plt.rcParams["xtick.direction"] = tickdirection
    plt.rcParams["ytick.direction"] = tickdirection
    plt.rcParams["axes.labelsize"] = labelsize
    try:
        plt.rcParams["font.sans-serif"] = fontkind
    except Exception:
        pass
    plt.rcParams["ytick.minor.visible"] = mtickvisible
    plt.rcParams["xtick.minor.visible"] = mtickvisible
    if fontweight:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"


def cm2inch(value):
    """Centimeters to feet.

    Args:
        value (float): centimeters.

    Returns:
        float: feet.
    """
    return value / 2.54
