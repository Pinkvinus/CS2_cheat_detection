{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "cs2_env";
  nativeBuildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.numpy     # Demoparser2 dependency
    python3Packages.pandas    # Demoparser2 dependency
    python3Packages.polars    # Demoparser2 dependency
    python3Packages.pyarrow   # Demoparser2 dependency
    python3Packages.tqdm      # Demoparser2 dependency
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
      pip install csgo
    else
      source .venv/bin/activate
    fi
  '';
}
