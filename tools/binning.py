import numpy as np
import pandas as pd


def OneBinning(data, direction, hbin, vbin, method="mean"):

    """
    对体系进行一维划分，然后对每一个区间/bin中的物理量进行统计平均，比如求和，平均等。
    输入参数:
    data : pandas.DataFrame
    direction : str, binnning direction, such as 'x','y','z', must included in data
    hbin : float, heigh/width of each bin with unit of \AA, such 5.0
    vbin : str/list, value to be binned, such as 'vx', ['vx', 'temp'], mush included in data
    method : str, method applied to value in each bin, supported in ['mean', 'sum', 'min', 'max']
           default : 'mean'
    输出参数：
    raw : pandas.DataFrame
    """

    if isinstance(vbin, str):
        vbin = [vbin]

    left, right = data[direction].min(), data[direction].max()
    nbin = np.round((right - left) / hbin)
    bins = np.linspace(left - 0.1, right + 0.1, int(nbin + 1))
    slices = data.groupby(pd.cut(data[direction], bins=bins, labels=False))
    if method == "mean":
        result = slices.mean()
    elif method == "sum":
        result = slices.sum()
    elif method == "min":
        result = slices.min()
    elif method == "max":
        result = slices.max()
    else:
        raise ValueError(
            "Unrecgonized method, choosen in ['mean', 'sum', 'min', 'max']."
        )

    num = slices.count()["id"].values
    value = result[vbin].values

    if not len(result["id"]) == len(bins) - 1:
        value_new = np.zeros((len(bins - 1), value.shape[1] + 1))
        index = result.index.values
        value_new[index, 1:] = value
        value_new[index, 0] = num
        value = value_new[:-1, :]
    else:
        value = np.hstack((num.reshape(-1, 1), value.reshape(-1, len(vbin))))
    x = (bins[1:] + bins[:-1]) / 2
    raw = np.hstack((x.reshape(-1, 1), value))
    raw = pd.DataFrame(raw, columns=["coor", "num"] + vbin)
    return raw


def TwoBinning(system, direction, hbin, vbin, method="mean"):

    """
    对体系进行2维划分，并对物理量进行统计。
    输入参数：
    system : system class.
    direction : [0, 1] means xy plane.
    hbin : float, heigh/width of each bin with unit of \AA, such 5.0
    vbin : str, value to be binned, such as 'vx', mush included in system
    method : str, method applied to value in each bin, supported in ['mean', 'sum']
           default : 'mean'
    输出参数：
    data : 2D array
    """

    assert len(direction) == 2
    assert max(direction) < 3
    assert isinstance(vbin, str)

    if vbin in system.data.columns:
        value = system.data[vbin].values
    else:
        print("This vbin is not in the system! Please check that!")
        return None

    pos = system.data[["x", "y", "z"]].values
    box = system.box
    ncel = np.floor((system.box[:, 1] - system.box[:, 0]) / hbin).astype(int)
    data = np.zeros(ncel[direction])
    start = box[direction, 0]
    row_max, col_max = np.array(data.shape) - 1
    jishu = np.zeros_like(data)
    if method in ["mean", "sum"]:
        for i in range(pos.shape[0]):
            row, col = np.floor((pos[i, direction] - start) / hbin).astype(int)
            if row > row_max:
                row = row_max
            if row < 0:
                row = 0
            if col > col_max:
                col = col_max
            if col < 0:
                col = 0
            data[row, col] += value[i]
            jishu[row, col] += 1
        if method == "mean":
            data[data > 0] = data[data > 0] / jishu[data > 0]
        return data.T
    else:
        raise ValueError("Unrecgonized method, choosen in ['mean', 'sum'].")
