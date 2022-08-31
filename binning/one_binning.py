import pandas as pd
import numpy as np 


def OneBinning(data, direction, hbin, vbin):
    
    if isinstance(vbin, str):
        vbin = [vbin]

    left, right = data[direction].min(), data[direction].max()
    nbin = np.round((right-left)/hbin) 
    bins = np.linspace(left-0.1, right+0.1, int(nbin+1))
    slices = data.groupby(pd.cut(data[direction], bins=bins, labels=False))
    result = slices.mean()
    num = slices.count()['id'].values
    value = result[vbin].values

    if not len(result['id']) == len(bins)-1:
        value_new = np.zeros((len(bins-1), value.shape[1]+1))

        new = np.arange(len(bins)-1, dtype=float)
        index = result.index.values
        index1 = np.array(list(set(new).difference(index)), dtype=int)
        value_new[index, 1:] = value
        value_new[index, 0] = num
        value = value_new[:-1, :]
    else:
        value = np.hstack((num.reshape(-1, 1), value.reshape(-1, len(vbin))))
    x = (bins[1:]+bins[:-1])/2
    raw = np.hstack((x.reshape(-1, 1), value))
    raw = pd.DataFrame(raw, columns =['coor', 'num'] + vbin)
    return raw
