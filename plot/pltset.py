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
    try:
        plt.style.use(["science", "notebook"])
    except:
        print("One should install SciencePlots package: pip install SciencePlots")
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
    plt.rcParams["font.sans-serif"] = fontkind
    plt.rcParams["ytick.minor.visible"] = mtickvisible
    plt.rcParams["xtick.minor.visible"] = mtickvisible
    if fontweight:
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["axes.labelweight"] = "bold"


def cm2inch(value):
    return value / 2.54
