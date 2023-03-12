import mdapy as mp

mp.init()

system = mp.System("solidliquid.dump")

system.cal_steinhardt_bond_orientation(solidliquid=True)
print(system.data)
system.write_dump()
