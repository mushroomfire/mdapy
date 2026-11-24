# mdapy-rocm6

The GPU part of [MDAPY](https://github.com/mushroomfire/mdapy) for AMD GPUs, built with
HIP of ROCm 6 for the Instinct MI200 and MI300 series (gfx90a, gfx942), for Linux on
x86-64. It needs the runtime of ROCm 6 on the machine (`libamdhip64.so.6`, as
`module load rocm` gives it on a cluster). Install it with MDAPY, at the same version:

```bash
pip install "mdapy[rocm6]"
```

MDAPY then runs its potentials and molecular dynamics on the GPU with `device="cuda"`,
the name of the GPU whoever made it.
See [On the GPU](https://mushroomfire.github.io/mdapy/user-guide/simulation/gpu/).
