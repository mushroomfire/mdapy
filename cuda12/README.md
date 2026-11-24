# mdapy-cuda12

The GPU part of [MDAPY](https://github.com/mushroomfire/mdapy), built with CUDA 12
for NVIDIA GPUs from the V100 on (compute capability 7.0), for Linux on x86-64 and on
ARM (aarch64, as the GH200 and GB200). It
needs the driver of the GPU, 525 or newer, and no CUDA toolkit. Install it with MDAPY,
at the same version:

```bash
pip install "mdapy[cuda12]"
```

MDAPY then runs its potentials and molecular dynamics on the GPU with `device="cuda"`.
See [On the GPU](https://mushroomfire.github.io/mdapy/user-guide/simulation/gpu/).

The package also holds MDAPY's plane-wave DFT, which needs NVIDIA's cuBLAS, cuFFT and
cuSOLVER. The extra `dft` installs them from NVIDIA's own packages on PyPI
(`nvidia-cublas-cu12`, `nvidia-cufft-cu12`, `nvidia-cusolver-cu12`), which a CUDA
build of PyTorch already has:

```bash
pip install "mdapy[cuda12,dft]"
```

See [Plane-wave DFT](https://mushroomfire.github.io/mdapy/user-guide/potentials/dft/).
