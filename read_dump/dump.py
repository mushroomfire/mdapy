import pandas as pd
import numpy as np

class read_data():
    
    def __init__(self, filename ,amass, rc, units, max_neigh = 100, num_threads = 1):
        self.filename = filename 
        self.amass = amass 
        self.rc = rc 
        self.units = units
        self.num_threads = num_threads
        self.max_neigh = max_neigh
        self.boundary, self.box, self.head, self.names = self.read_box()
        self.data, self.N, self.mass, self.type_list = self.read_dump(self.names, self.amass)
        self.pos = self.data[['x', 'y', 'z']].values 
        self.vel = self.data[['vx', 'vy', 'vz']].values 
        self.box_l = self.box[:,1] - self.box[:, 0]
        self.vol = self.box_l[0]*self.box_l[1]*self.box_l[2]
        self.neigh = False
    
    def read_box(self):
        with open(self.filename) as op:
            file = op.readlines()
        boundary = np.array([1 if i == 'pp' else 0 for i in file[4].split()[-3:]])
        box = np.array([i.split()[:2] for i in file[5:8]]).astype(float)
        head = file[:9]
        names = file[8].split()[2:]
        return boundary, box, head, names
    
    def read_dump(self, names, amass):

        data = pd.read_csv(self.filename, skiprows=9, index_col=False, header=None, sep=' ', names=names)
        N = data.shape[0]
        mass = np.zeros(N)
        type_list = np.unique(data['type'].values).astype(int)
        for i in type_list:
            mass[data['type']==i] = amass[i-1]
        return data, N, mass, type_list
        
        
if __name__ == '__main__':
    filename, amass, rc, units = r'../example/test.dump', [58.933, 63.546, 55.847, 58.693, 106.42], 5.0, 'metal'
    system = read_data(filename ,amass, rc, units)
    print('system box:')
    print(system.box)
    print('system first pos :')
    print(system.pos[0])
    print('system first vel :')
    print(system.vel[0])
    print('system first atom mass :')
    print(system.mass[0])