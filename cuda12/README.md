# mdapy-cuda12

The GPU part of [MDAPY](https://github.com/mushroomfire/mdapy), built with CUDA 12
for NVIDIA GPUs from the V100 on (compute capability 7.0), for Linux on x86-64. It
needs the driver of the GPU, 525 or newer, and no CUDA toolkit. Install it with MDAPY,
at the same version:

```bash
pip install "mdapy[cuda12]"
```

MDAPY then runs its potentials and molecular dynamics on the GPU with `device="cuda"`.
See [On the GPU](https://mushroomfire.github.io/mdapy/user-guide/simulation/gpu/).
