import numpy as np

def TwoBinning(system, direction, hbin, vbin):
    
    """
    direction : [0, 1] means xy plane.
    hbin: 3.0, the length of bin.
    vbin: property you want to average. Only one!
    """
    
    assert len(direction) == 2
    assert max(direction) < 3
    assert isinstance(vbin, str)
    
    if vbin in system.data.columns:
        value = system.data[vbin].values
    else:
        print('This vbin is not in the system! Please check that!')
        return None
    
    pos = system.pos
    box = system.box
    ncel = np.floor((system.box[:,1] - system.box[:,0])/hbin).astype(int)
    data = np.zeros(ncel[direction])
    start = box[direction, 0]
    row_max, col_max = np.array(data.shape)-1
    jishu = np.zeros_like(data)
    for i in range(pos.shape[0]):
        row, col = np.floor((pos[i, direction] - start)/hbin).astype(int)
        if row > row_max:
            row = row_max
        if col > col_max:
            col = col_max
        data[row, col] += value[i]
        jishu[row, col] += 1
    if vbin == 'density':
        return data.T
    else:
        data[data>0] = data[data>0]/jishu[data>0]
        return data.T