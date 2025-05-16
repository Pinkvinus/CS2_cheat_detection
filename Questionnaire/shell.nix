{ pkgs ? import <nixpkgs> {} }:

# In order to run jupyternb in vscode, run 'nix-shell' in a 
# terminal, and open vscode in that same terminal using 'code .'

pkgs.mkShell {
  name = "cs2_env";
  nativeBuildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.pandas
    python3Packages.matplotlib
    python3Packages.ipython
    python3Packages.jupyter
    python3Packages.pyarrow
    python3Packages.fastparquet
  ];
}