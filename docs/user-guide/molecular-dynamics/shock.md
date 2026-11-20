# Choosing a method

<p class="runs-on" markdown>**Runs on** the processor · the [GPU](gpu.md) · [LAMMPS](in-lammps.md) as `fix wall/piston`, `nphug` and `msst`</p>

A shock wave takes matter from rest to a state of higher pressure, density and
temperature in a front a few atomic planes thick. MDAPY simulates it in three ways,
which answer different questions:

| | [Piston](walls.md) | [Hugoniostat](hugoniostat.md) | [Multiscale shock technique](msst.md) |
|---|---|---|---|
| Call | a moving [`Wall`](walls.md) | `ensemble="nphug"` | `ensemble="msst"` |
| Gives | the shock wave itself: its front, its velocity, and the matter behind it as the wave passes | the state behind a shock of a given pressure, with its temperature and its shock velocity | the matter behind a shock of a given velocity, in time, with its pressure |
| Takes | the velocity of the piston, the velocity of the matter behind the shock | the pressure | the shock velocity |
| System | a long crystal, open along the axis of the shock | a small periodic box, compressed homogeneously | a small periodic box, compressed along the axis |
| Each run | one shock, with its front | one point of the Hugoniot | the path from the start to the shocked state |

- **A piston** shows what the wave does: the width of its front, and the plasticity or
  the phase change behind it. It needs a crystal many times
  longer than the front, and as many steps as the wave takes to cross it.
  [A shock wave in copper](../../examples/shock.md) drives one into 102400 atoms on a
  GPU.
- **The Hugoniostat** maps the shocked states, one pressure per run, in small boxes: the Hugoniot, the temperature along it, and where the crystal
  melts or changes phase.
- **MSST** follows one shock in time, without its front, in a small box: how the
  matter behind a shock of a given velocity heats, yields or transforms as time goes
  on.

A [still wall](walls.md) is a momentum mirror: a crystal that moves against it is
shocked as by a piston of its velocity.
