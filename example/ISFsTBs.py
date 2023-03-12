import mdapy as mp

mp.init()

system = mp.System("ISF.dump")
system.cal_identify_SFs_TBs()
print(system.data)
system.write_dump()

