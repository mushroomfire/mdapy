# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import matplotlib.pyplot as plt
from cycler import cycler


def pltset(color_cycler=None, **kargs):
    """This is a drawing template function, which refers to the science style from `SciencePlots <https://github.com/garrettj403/SciencePlots>`_
    and color also comes from `Paul Tot's website <https://personal.sron.nl/~pault/>`_.

    Args:
        color_cycler (str | list[str] | tuple[str], optional): One can use other color cycler belongs to ['bright', 'high-contrast', 'high-vis', 'light', 'muted', 'retro', 'std-colors', 'vibrant'], or directly provide a list or tuple contains colors, such as ['#FFAABB', '#99DDFF', '#44BB99']. Defaults to None.

    One also can modify the default parameters by provide a dict. For example, if you want to use 'Times New Roman' font and hide the minor ticks: you can try:

    pltset(**{'font.serif':'Times New Roman', 'xtick.minor.visible':False, 'ytick.minor.visible':False})

    If you want to change the axes linewidth and major tick width, and make bolded font, you can try:

    pltset(**{"xtick.major.width":1., "ytick.major.width":1., "axes.linewidth":1., "font.weight":'bold', "axes.labelweight":'bold'})

    More parameters can be found in plt.rcParams.keys().

    """

    plt.rcParams.clear()
    if color_cycler is None:
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#0C5DA5",
                "#00B945",
                "#FF9500",
                "#FF2C00",
                "#845B97",
                "#474747",
                "#9e9e9e",
            ],
        )
    elif color_cycler == "bright":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#4477AA",
                "#EE6677",
                "#228833",
                "#CCBB44",
                "#66CCEE",
                "#AA3377",
                "#BBBBBB",
            ],
        )
    elif color_cycler == "high-contrast":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", ["#004488", "#DDAA33", "#BB5566"]
        )
    elif color_cycler == "high-vis":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", ["#0d49fb", "#e6091c", "#26eb47", "#8936df", "#fec32d", "#25d7fd"]
        ) + cycler("ls", ["-", "--", "-.", ":", "-", "--"])
    elif color_cycler == "light":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#77AADD",
                "#EE8866",
                "#EEDD88",
                "#FFAABB",
                "#99DDFF",
                "#44BB99",
                "#BBCC33",
                "#AAAA00",
                "#DDDDDD",
            ],
        )
    elif color_cycler == "muted":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#CC6677",
                "#332288",
                "#DDCC77",
                "#117733",
                "#88CCEE",
                "#882255",
                "#44AA99",
                "#999933",
                "#AA4499",
                "#DDDDDD",
            ],
        )
    elif color_cycler == "retro":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", ["#4165c0", "#e770a2", "#5ac3be", "#696969", "#f79a1e", "#ba7dcd"]
        )
    elif color_cycler == "std-colors":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#0C5DA5",
                "#00B945",
                "#FF9500",
                "#FF2C00",
                "#845B97",
                "#474747",
                "#9e9e9e",
            ],
        )
    elif color_cycler == "vibrant":
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color",
            [
                "#EE7733",
                "#0077BB",
                "#33BBEE",
                "#EE3377",
                "#CC3311",
                "#009988",
                "#BBBBBB",
            ],
        )
    elif isinstance(color_cycler, list) or isinstance(color_cycler, tuple):
        plt.rcParams["axes.prop_cycle"] = cycler(
            "color", [color for color in color_cycler]
        )
    else:
        raise "color_style must belong to ['bright', 'high-contrast', 'high-vis', 'light', 'muted', 'retro', 'std-colors', 'vibrant'], or a list, or a tuple!"

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.size"] = 1.5
    plt.rcParams["xtick.minor.width"] = 0.5
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["xtick.top"] = True

    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.size"] = 1.5
    plt.rcParams["ytick.minor.width"] = 0.5
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["ytick.right"] = True

    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["lines.markersize"] = 4

    plt.rcParams["legend.frameon"] = False

    plt.rcParams["font.serif"] = ["cmr10", "Computer Modern Serif", "DejaVu Serif"]
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["font.size"] = 11

    for key, value in kargs.items():
        try:
            plt.rcParams[key] = value
        except Exception as e:
            print(e)
            pass


def pltset_old(color_cycler=None, **kargs):
    """This function used to generate the same style with mdapy<0.9.9. Note that the
    'Times New Roman' font is required.

    Args:
        color_cycler (str | list[str] | tuple[str], optional): One can use other color cycler belongs to ['bright', 'high-contrast', 'high-vis', 'light', 'muted', 'retro', 'std-colors', 'vibrant'], or directly provide a list or tuple contains colors, such as ['#FFAABB', '#99DDFF', '#44BB99']. Defaults to None.

    One also can modify the default parameters by provide a dict. For example, if you want to let the font normal and set the font size, you can try:

    pltset_old(**{"font.weight":'normal', "axes.labelweight":'normal', "font.size" : 10})

    More parameters can be found in plt.rcParams.keys().
    """
    pltset(color_cycler=color_cycler)
    default = {
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.linewidth": 1.0,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "font.serif": "Times New Roman",
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.top": False,
        "ytick.right": False,
        "font.size": 12,
        "legend.fontsize": 10,
    }
    for key, value in default.items():
        try:
            plt.rcParams[key] = value
        except Exception as e:
            print(e)
            pass
    for key, value in kargs.items():
        try:
            plt.rcParams[key] = value
        except Exception as e:
            print(e)
            pass


def cm2inch(value):
    """Centimeters to feet.

    Args:
        value (float): centimeters.

    Returns:
        float: feet.
    """
    return value / 2.54


def set_figure(
    figsize=(10, 6),
    figdpi=150,
    top=0.935,
    bottom=0.18,
    left=0.115,
    right=0.965,
    hspace=0.2,
    wspace=0.2,
    nrow=1,
    ncol=1,
    use_pltset=False,
    use_pltset_old=False,
):
    """This function can generate a Figure and a Axes object easily.

    Args:
        figsize (tuple, optional): figsize with units of Centimeters. Defaults to (10, 6).
        figdpi (int, optional): figure dpi to show. Defaults to 150.
        top (float, optional): axes range in figure at top, should be [0., 1.]. Defaults to 0.935.
        bottom (float, optional): axes range in figure at bottom, should be [0., 1.]. Defaults to 0.18.
        left (float, optional): axes range in figure at left, should be [0., 1.]. Defaults to 0.115.
        right (float, optional): axes range in figure at right, should be [0., 1.]. Defaults to 0.965.
        hspace (float, optional): space between two subplots along height direction, should be [0., 1.]. Defaults to 0.2.
        wspace (float, optional): space between two subplots along width direction, should be [0., 1.]. Defaults to 0.2.
        nrow (int, optional): the rows number. Defaults to 1.
        ncol (int, optional): the columns number. Defaults to 1.
        use_pltset (bool, optional): whether use the pltset. Defaults to False.
        use_pltset_old (bool, optional): whether use the old pltset. This will be used only if use_pltset is False. Defaults to False.

    Returns:
        tuple: Figure and Axes object.
    """
    if use_pltset:
        pltset()
    elif use_pltset_old:
        pltset_old()
    fig, ax = plt.subplots(
        nrow, ncol, figsize=tuple(cm2inch(size) for size in figsize), dpi=figdpi
    )
    plt.subplots_adjust(
        top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace
    )
    return fig, ax


if __name__ == "__main__":
    # pltset_old(color_cycler="high-vis", **{"font.weight":'normal', "axes.labelweight":'normal', })
    pltset_old()
    # pltset_old(**{"font.weight":'normal', "axes.labelweight":'normal', })
    # pltset('high-vis', **{'font.serif':'Times New Roman', 'xtick.minor.visible':False, 'ytick.minor.visible':False,})
    fig, ax = set_figure(
        ncol=2,
        nrow=1,
        figsize=(16, 6),
        wspace=0.3,
        use_pltset=False,
        use_pltset_old=False,
    )
    for i in range(3):
        ax[0].plot(range(i, 10 + i), label=i)
    ax[1].plot(range(12), "o-")
    for i in range(2):
        ax[i].set_xlabel("x label")
        ax[i].set_ylabel("y label")
    ax[0].legend()
    plt.show()
