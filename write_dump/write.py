import pandas as pd 
import numpy as np
from pyAnalysis.timer.timer import timer


@timer
def write_dump(system):
    head, data, filename = system.head, system.data, system.filename
    for dtype, name in zip(data.dtypes, data.columns):
        if dtype == 'int64':
            data[name] = data[name].astype(np.int32)
        elif dtype == 'float64':
            data[name] = data[name].astype(np.float32)
    #if data.isnull().values.any():
    #    data = data.fillna(0.0)
    a = filename.split('.')
    a.insert(-1, 'output')
    output_name = '.'.join(a)
    head[-1] += '\n'
    with open(output_name, 'w') as op:
        op.write(''.join(head))
    data.to_csv(output_name, header=None, index=False, sep=" ", mode="a", na_rep = "nan")