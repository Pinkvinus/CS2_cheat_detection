{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "cs2_env";
  buildInputs = [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.numpy     # Demoparser2 dependency
    pkgs.python3Packages.pandas    # Demoparser2 dependency
    pkgs.python3Packages.polars    # Demoparser2 dependency
    pkgs.python3Packages.pyarrow   # Demoparser2 dependency
    pkgs.python3Packages.tqdm      # Demoparser2 dependency
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.wheel
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    # Set up virtual environment if not already created
    #     For some reason, dependencies for demoparser2 are not correctly set up with pip.
    #     So the way it is done here is dependencies are installed in the mkShell.

    if [ ! -d ".venv" ]; then
      python -m venv .venv
      source .venv/bin/activate
      pip install --upgrade pip
      pip install demoparser2
    else
      source .venv/bin/activate
    fi
  '';
}
