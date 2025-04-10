{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "cs2_env";
  nativeBuildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.pandas
    python3Packages.matplotlib
    python3Packages.ipython
    python3Packages.jupyter
  ];
}