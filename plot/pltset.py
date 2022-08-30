import matplotlib.pyplot as plt


def pltset(ticksize=12, labelsize=12, legendsize=12): 
    try:
    	plt.style.use(['science', 'notebook'])
    except:
        print('One should install SciencePlots package: pip install SciencePlots')
    plt.rcParams['legend.fontsize']  = legendsize
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['xtick.major.size'] = 3
    plt.rcParams['ytick.major.size'] = 3
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['font.sans-serif'] =  'Times New Roman'
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['xtick.minor.visible'] = False
    # plt.rcParams['font.weight'] = 'bold'
    # plt.rcParams['axes.labelweight'] = 'bold'
    
def cm2inch(value):
    return value/2.54
