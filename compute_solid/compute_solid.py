
try:
    import freud 
except ImportError:
    raise ModuleNotFoundError("compute_solid method needs freud compiled as a library. Either install with conda - conda install -c conda-forge freud,\
                or see here - https://freud.readthedocs.io/en/latest/")


from pyAnalysis.timer.timer import timer


@timer
def get_solid(system):
    freud.parallel.set_num_threads(system.num_threads)
    f = freud.order.SolidLiquid(6, 0.7, 8, True)
    box = freud.box.Box(*system.box_l, 0, 0, 0, False)
    box.periodic = [True if i == 1 else False for i in system.boundary]
    new_system = (box, system.pos)
    f.compute(new_system, neighbors={"num_neighbors": 12})
    system.data['nbond'] = f.num_connections
    system.head[-1] = system.head[-1].strip() + ' nbond'