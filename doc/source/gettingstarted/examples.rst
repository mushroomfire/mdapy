Examples
=========

1. Structure analysis
----------------------

.. code-block:: python

    import mdapy as mp
    mp.init('cpu') # use cpu, mp.init('gpu') will use gpu to compute.

    system = mp.System('./CoCuFeNiPd-4M.dump') # read dump file to generate a system class
    system.cal_centro_symmetry_parameter() # calculate the centrosymmetry parameters
    system.cal_atomic_entropy() # calculate the atomic entropy
    system.write_dump() # save results to a new dump file

2. Mean squared displacement and Lindemann index
--------------------------------------------------

.. code-block:: python 

    import mdapy as mp
    mp.init('cpu')

    dump_list = [f'melt.{i}.dump' for i in range(100)] # obtain all the dump filenames in a list
    MS = mp.MultiSystem(dump_list) # read all the dump file to generate a MultiSystem class
    MS.cal_mean_squared_displacement() # calculate the mean squared displacement
    MS.MSD.plot() # one can plot the MSD per frame
    MS.cal_lindemann_parameter() # calculate the lindemann index
    MS.Lindemann.plot() # one can plot lindemann index per frame
    MS.write_dumps() # save results to a serials of dump files

3. Calculate WCP matrix in high-entropy alloy
-----------------------------------------------

.. code-block:: python 

    import mdapy as mp

    mp.init(arch="cpu")

    system = mp.System("CoCuFeNiPd-4M.data")
    system.cal_warren_cowley_parameter()  # calculate WCP parameter
    fig, ax = system.WarrenCowleyParameter.plot(
        elements_list=["Co", "Cu", "Fe", "Ni", "Pd"]
    )  # plot WCP matrix
    fig.savefig("WCP.png", dpi=300, bbox_inches="tight", transparent=True)

4. Create polycrystalline with graphene boundary
--------------------------------------------------

.. code-block:: python 

    import mdapy as mp
    import numpy as np
    mp.init('cpu')

    box = np.array([[0, 800.], [0, 200.], [0, 200.]]) # create a box
    seednumber = 20 # create 20 seeds to generate the voronoi polygon
    metal_lattice_constant = 3.615 # lattice constant of metallic matrix
    metal_lattice_type = 'FCC' # lattice type of metallic matrix
    randomseed = 1 # control the crystalline orientations per grains
    add_graphene=True # use graphen as grain boundary
    poly = mp.CreatePolycrystalline(box, seednumber, metal_lattice_constant, metal_lattice_type, randomseed=randomseed, add_graphene=add_graphene, gra_overlap_dis=1.2)
    poly.compute() # generate a polycrystalline with graphene boundary

5. Calculate the EOS curve
----------------------------

.. code-block:: python 

    import numpy as np
    import matplotlib.pyplot as plt
    import mdapy as mp
    from mdapy import pltset, cm2inch
    mp.init('cpu')

    def get_enegy_lattice(potential, pos, box):
        
        neigh = mp.Neighbor(pos, box, potential.rc, max_neigh=150) # build neighbor list
        neigh.compute()
        Cal = mp.Calculator(
                potential,
                ["Al"],
                np.ones(pos.shape[0], dtype=np.int32),
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
                pos,
                [1, 1, 1],
                box,
            ) # calculate the energy
        Cal.compute()
        return Cal.energy.mean()

    eos = []
    lattice_constant = 4.05
    x, y, z = 3, 3, 3
    FCC = mp.LatticeMaker(lattice_constant, "FCC", x, y, z) # build a FCC lattice
    FCC.compute()
    potential = mp.EAM("Al_DFT.eam.alloy") # read a eam.alloy potential file
    for scale in np.arange(0.9, 1.15, 0.01): # loop to get different energies
        energy = get_enegy_lattice(potential, FCC.pos*scale, FCC.box*scale)
        eos.append([scale*lattice_constant, energy])
    eos = np.array(eos)

    # plot the eos results
    pltset()
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
    plt.subplots_adjust(bottom=0.18, top=0.92, left=0.2, right=0.98)
    plt.plot(eos[:,0], eos[:,1], 'o-')
    e_coh = eos[:,1].min()
    a_equi = eos[np.argmin(eos[:, 1]), 0]
    plt.plot([a_equi], [e_coh], 'o', mfc='white')
    plt.title(r'$\mathregular{E_{Coh}}$ : %.2f eV, a : %.2f $\mathregular{\AA}$' % (e_coh, a_equi), fontsize=10)
    plt.xlim(eos[0,0]-0.2, eos[-1,0]+0.2)
    plt.xlabel("a ($\mathregular{\AA}$)")
    plt.ylabel(r"PE (eV/atom)")
    ax = plt.gca()
    plt.savefig('eos.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

6. Collaborative use with Ovito
--------------------------------

This function can run in script environment of Ovito. (Tested in Ovito 3.0.0-dev581).

.. code-block:: python

    from ovito.data import *
    import mdapy as mp
    import numpy as np
    mp.init()

    def modify(frame, data):

        yield "Transforming mdapy system..."
        cell_ovito = data.cell[...]
        box = np.array([[cell_ovito[i, -1], cell_ovito[i, -1]+cell_ovito[i, i]] for i in range(3)])
        pos = np.array(data.particles['Position'][...])
        boundary = [int(i) for i in data.cell.pbc]
        system = mp.System(pos=pos, box=box, boundary=boundary)

        yield "Performing compute entropy..."
        system.cal_atomic_entropy(
            rc=3.6*1.4,
            sigma=0.2,
            use_local_density=True,
            compute_average=True,
            average_rc=3.6*0.9,
            max_neigh=80,
        )
        data.particles_.create_property("entropy", data=system.data['ave_atomic_entropy'].values)
        
        yield "Performing compute CNP..."
        system.cal_common_neighbor_parameter(rc=3.6*0.8536)
        data.particles_.create_property("cnp", data=system.data['cnp'].values)


7. Identify stacking faults (SFs) and twin boundaries (TBs) in Ovito
----------------------------------------------------------------------

This function can run in script environment of Ovito. (Tested in Ovito 3.0.0-dev581).
Here the stacking faults include intrinsic SFs and multi layer SFs.

.. code-block:: python

    from ovito.data import *
    from ovito.modifiers import ExpressionSelectionModifier, AssignColorModifier
    import mdapy as mp
    import numpy as np

    mp.init()


    def modify(
        frame,
        data,
        other=(243 / 255, 243 / 255, 243 / 255),
        fcc=(102 / 255, 255 / 255, 102 / 255),
        bcc=(102 / 255, 102 / 255, 255 / 255),
        hcp=(255/255, 85/255, 0/255),
        ISF=(0.9, 0.7, 0.2),
        ESF=(132/255, 25/255, 255/255),
        TB=(255 / 255, 102 / 255, 102 / 255),
    ):

        yield "Transforming mdapy system..."
        cell_ovito = data.cell[...]
        box = np.array(
            [[cell_ovito[i, -1], cell_ovito[i, -1] + cell_ovito[i, i]] for i in range(3)]
        )
        pos = np.array(data.particles["Position"][...])
        boundary = [int(i) for i in data.cell.pbc]
        system = mp.System(pos=pos, box=box, boundary=boundary)

        yield "Performing identify SFTB..."
        system.cal_identify_SFs_TBs()
        data.particles_.create_property(
            "structure_types", data=system.data["structure_types"].values
        )
        data.particles_.create_property(
            "fault_types", data=system.data["fault_types"].values
        )

        yield "Coloring atoms..."
        data.apply(
            ExpressionSelectionModifier(expression="structure_types==0 || fault_types==1")
        )
        data.apply(AssignColorModifier(color=other))
        data.apply(ExpressionSelectionModifier(expression="structure_types==1"))
        data.apply(AssignColorModifier(color=fcc))
        data.apply(ExpressionSelectionModifier(expression="structure_types==3"))
        data.apply(AssignColorModifier(color=bcc))
        data.apply(ExpressionSelectionModifier(expression="fault_types==4"))
        data.apply(AssignColorModifier(color=hcp))
        data.apply(ExpressionSelectionModifier(expression="fault_types==2"))
        data.apply(AssignColorModifier(color=ISF))
        data.apply(ExpressionSelectionModifier(expression="fault_types==5"))
        data.apply(AssignColorModifier(color=ESF))
        data.apply(ExpressionSelectionModifier(expression="fault_types==3"))
        data.apply(AssignColorModifier(color=TB))
        data.apply(ExpressionSelectionModifier(expression="structure_types==10"))

.. image:: https://img.pterclub.com/images/2023/03/20/SF.png



