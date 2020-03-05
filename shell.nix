with import <nixpkgs> {};
stdenv.mkDerivation {
  name = "qopter-rs";
  buildInputs = [
    bashInteractive
    rustup
    mypy
    python3
    python3Packages.jupyter
    python3Packages.jupyterlab
    python3Packages.qiskit
    python3Packages.numpy
    python3Packages.matplotlib
    lldb
  ];
}
