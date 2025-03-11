{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.setuptools
    python3Packages.wheel
    python3Packages.virtualenv
    cmake
    gcc
    pkgs.stdenv.cc.cc.lib  # Ensure libstdc++.so.6 is available
  ];

  shellHook = ''
    # Ensure the virtual environment exists
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    
    # Activate the virtual environment
    source .venv/bin/activate

    # Upgrade pip and install demoparser2 if not already installed
    if ! python -c "import demoparser2" &>/dev/null; then
      pip install --upgrade pip
      pip install demoparser2
    fi
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

    export PYTHONPATH="$(pwd)/.venv/lib/python3.10/site-packages:$PYTHONPATH"
  '';
}