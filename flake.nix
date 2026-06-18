{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    simwork.url = "git+ssh://git@git.novadiscovery.net/jinko/jinko.git";
  };
  outputs = { nixpkgs, self, flake-utils, simwork, ... }: flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = nixpkgs.outputs.legacyPackages.${system};
  in {
    legacyPackages.simwork = simwork;
    devShells.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        poetry
        gcc
        openssl
        libz
      ];
      shellHook = ''
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libz}/lib
      '';
    };
  });
}
