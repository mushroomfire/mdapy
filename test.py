from lattice_maker.lattice_maker import LatticeMaker

Al = LatticeMaker(4.05, "FCC", 5, 5, 5)
pos = Al.get_pos()
print(pos.shape)
print(pos[0])
