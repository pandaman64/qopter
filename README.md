# QOPTER: Quantum program OPTimizER
This repository contains source files for a optimizing compiler introduced in ["Extracting Success from IBM's 20-Qubit Machines Using Error-Aware Compilation"](https://arxiv.org/abs/1903.10963).

The repository depends on **outdated** crates (Rust libraries) and qiskit. Do NOT expect that the repository compiles as-is.
To use, try compiling the crate and copy the generated shared object into `qopter` directory. For the usage, please see `test_qopter.py`.

The repository is licensed under Apache License 2.0.