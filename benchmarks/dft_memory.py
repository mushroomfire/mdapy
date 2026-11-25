# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The memory of the GPU that MDAPY's plane-wave DFT holds: for cells of
aluminium (600 eV, kspacing 0.15 1/Å) and of iron with spin (900 eV,
kspacing 0.2 1/Å, 3 Bohr magnetons per atom), Marzari-Vanderbilt smearing of
0.25 eV, PseudoDojo 0.4 norm-conserving pseudopotentials, each cell perturbed
as a training set is (atoms displaced by 0.05 Å, the box strained by up to
0.01, seed 7), the memory pool's counters at the end of each step of the
self-consistent field and at their peaks, and the memory in use on the GPU
(cudaMemGetInfo, the CUDA context and the libraries included) polled every
0.2 s.

    python benchmarks/dft_memory.py PSEUDO OUT.jsonl alone CASE[,CASE...] [RUNS]
    python benchmarks/dft_memory.py PSEUDO OUT.jsonl together CASE THREADS [CELLS]
    python benchmarks/dft_memory.py PSEUDO OUT.jsonl after CASE THREADS NEXT
    python benchmarks/dft_memory.py report OUT.jsonl

PSEUDO is the folder of Al.upf and Fe.upf. alone runs RUNS cells (3) in a row
on one calculator, the first of the process; together runs THREADS threads at
once, each with a calculator of its own and CELLS cells (2), after one cell
that loads the libraries; after runs that, then one cell of NEXT alone in the
same process. Each run of the case "_ideal" takes the perfect crystal, its
symmetry with spglib. KERNELS and PRECISION in the environment set those of
the calculator ("auto" and "double" by default). One JSON line per run, in
MiB; report prints the table of the runs alone (markdown).
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MIB = 1 << 20


def cases():
    from mdapy.build import build_crystal
    from mdapy.transform import replicate

    fcc = build_crystal("fcc", 4.04, elements="Al")
    bcc = build_crystal("bcc", 2.83, elements="Fe")
    al = dict(encut=600.0, kspacing=0.15, smearing="marzari-vanderbilt", sigma=0.25)
    fe = dict(encut=900.0, kspacing=0.2, smearing="marzari-vanderbilt", sigma=0.25,
              magmoms={"Fe": 3.0})  # fmt: skip
    return {
        "Al4": (fcc, al),
        "Al32": (replicate(fcc, (2, 2, 2)), al),
        "Al32_mixed": (replicate(fcc, (2, 2, 2)), dict(al, precision="mixed")),
        "Al108_gamma": (replicate(fcc, (3, 3, 3)), dict(al, kpoints=(1, 1, 1))),
        "Al108": (replicate(fcc, (3, 3, 3)), al),
        "Fe16": (replicate(bcc, (2, 2, 2)), fe),
        "Fe16_mixed": (replicate(bcc, (2, 2, 2)), dict(fe, precision="mixed")),
        "Fe54": (replicate(bcc, (3, 3, 3)), fe),
        "Fe54_k222": (replicate(bcc, (3, 3, 3)), dict(fe, kpoints=(2, 2, 2))),
        "Fe54_ideal": (replicate(bcc, (3, 3, 3)), fe),
    }


def mib(x) -> float:
    return round(float(x) / MIB, 1)


class Watch:
    """The largest memory in use on the GPU while it runs, polled."""

    def __init__(self, module):
        self.module = module

    def __enter__(self):
        self.most, self.stop = 0, threading.Event()

        def poll():
            while not self.stop.is_set():
                m = self.module.memory()
                self.most = max(self.most, m["total"] - m["free"])
                self.stop.wait(0.2)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()


def measure(pseudo: Path, out: Path, mode: str, args: list[str]) -> None:
    from mdapy.calculators import DFT
    from mdapy.calculators._dft import loader, scf
    from mdapy.mlip import perturb

    _dft = loader.module()
    kernels = os.environ.get("KERNELS", "auto")
    precision = os.environ.get("PRECISION", "double")
    table = cases()
    captured = threading.local()
    original = scf.run

    def run(*a, **k):  # the result of the field, for its counters
        captured.r = original(*a, **k)
        return captured.r

    scf.run = run

    def calculator(settings):
        s = dict(settings)
        s.setdefault("precision", precision)
        files = {e: pseudo / f"{e}.upf" for e in ("Al", "Fe")}
        return DFT(pseudopotentials=files, kernels=kernels, **s)

    def cells(name, count):
        crystal, _ = table[name]
        size = 0.0 if name.endswith("_ideal") else 1.0
        rng = np.random.default_rng(7)
        return perturb(
            crystal, count, displacement=0.05 * size, strain=0.01 * size, seed=rng
        )

    def one(dft, cell):
        start = time.time()
        energy = dft.compute(cell, ("energy", "forces")).energy
        seconds = time.time() - start
        r = captured.r
        return dict(seconds=round(seconds, 2), energy=float(energy), iterations=int(r["iterations"]),
                    kpoints=len(r["weights"]), spins=int(r["spins"]), bands=int(r["bands"]),
                    grid=[int(g) for g in r["grid"]], operations=int(r["operations"]),
                    steps={k: [mib(v) for v in r["memory"][k]] for k in r["memory"]})  # fmt: skip

    def counters(before, after, watch):
        return dict(before={k: mib(v) for k, v in before.items()},
                    after={k: mib(v) for k, v in after.items()}, device_peak=mib(watch.most))  # fmt: skip

    def write(row):
        row.update(gpu=_dft.device_name(), kernels=kernels, precision=precision)
        print(json.dumps(row), flush=True)
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def alone(name, cell, dft, **extra):
        _dft.reset_memory_peaks()
        before = _dft.memory()
        with Watch(_dft) as w:
            row = one(dft, cell)
        row.update(
            case=name,
            atoms=len(table[name][0]),
            **extra,
            **counters(before, _dft.memory(), w),
        )
        write(row)

    if mode == "alone":
        runs = int(args[1]) if len(args) > 1 else 3
        for name in args[0].split(","):
            dft = calculator(table[name][1])
            for i, cell in enumerate(cells(name, runs)):
                alone(name, cell, dft, mode="alone", run=i)
        return
    name, threads = args[0], int(args[1])
    each = int(args[2]) if mode == "together" and len(args) > 2 else 2
    some = cells(name, each * threads + 1)
    calculator(table[name][1]).compute(some[-1])  # the context and the libraries
    _dft.reset_memory_peaks()
    before = _dft.memory()
    local = threading.local()

    def job(i):
        if not hasattr(local, "dft"):
            local.dft = calculator(table[name][1])
        try:
            return one(local.dft, some[i])
        except Exception as error:  # out of memory, or what else
            return dict(error=f"{type(error).__name__}: {error}")

    start = time.time()
    with Watch(_dft) as w, ThreadPoolExecutor(threads) as pool:
        rows = list(pool.map(job, range(each * threads)))
    write(dict(case=name, mode="together", threads=threads, atoms=len(table[name][0]),
               wall=round(time.time() - start, 2), cells=each * threads,
               errors=[r["error"] for r in rows if "error" in r],
               seconds=[r.get("seconds") for r in rows], energies=[r.get("energy") for r in rows],
               **counters(before, _dft.memory(), w)))  # fmt: skip
    if mode == "after":
        alone(args[2], cells(args[2], 1)[0], calculator(table[args[2]][1]), mode="after",
              following=f"{name} x {threads}")  # fmt: skip


def report(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print("| Case | GPU | Atoms | k-points × spins | Bands | Grid | In use (GiB) | Reserved (GiB) "
          "| On the GPU (GiB) | Time (s) |")  # fmt: skip
    print("|---|---|---:|---|---:|---|---:|---:|---:|---:|")
    for r in rows:
        if r.get("mode") != "alone" or r.get("run") != 1:
            continue
        grid = " × ".join(str(g) for g in r["grid"])
        gib = lambda x: f"{x / 1024:.1f}"  # noqa: E731
        print(f"| {r['case']} | {r['gpu']} | {r['atoms']} | {r['kpoints']} × {r['spins']} | {r['bands']} "
              f"| {grid} | {gib(r['after']['in_use_peak'])} | {gib(r['after']['reserved_peak'])} "
              f"| {gib(r['device_peak'])} | {r['seconds']:.1f} |")  # fmt: skip


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(Path(sys.argv[2]))
    else:
        measure(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
