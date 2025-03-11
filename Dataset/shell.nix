# shell.nix
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.gcc
    pkgs.libsigcxx
    pkgs.libstdcxx5
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    echo "Setting up virtualenv..."
    if [ ! -d "venv" ]; then
      python3 -m venv venv
      source venv/bin/activate
      pip install --upgrade demoparser2
    else
      source venv/bin/activate
    fi
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nix/store/agp6lqznayysqvqkx4k1ggr8n1rsyi8c-gcc-13.2.0-lib/lib
    export PYTHONPATH=$PYTHONPATH:${toString pkgs.python3Packages.pip}
  '';
}
