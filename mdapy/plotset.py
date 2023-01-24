# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import matplotlib.pyplot as plt
import platform


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
    fontkind=None,
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
        fontkind (str, optional): font family. Defaults to "Times New Roman" for Windows.
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
    if fontkind is None:
        if platform.platform().split("-")[0] == "Windows":
            plt.rcParams["font.sans-serif"] = "Times New Roman"
    else:
        plt.rcParams["font.sans-serif"] = fontkind

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


if __name__ == "__main__":
    pltset()
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(6)), dpi=150)
    plt.subplots_adjust(
        top=0.935, bottom=0.18, left=0.115, right=0.965, hspace=0.2, wspace=0.2
    )
    plt.plot(range(10), "o-")
    plt.xlabel("x label")
    plt.ylabel("y label")

    plt.show()
