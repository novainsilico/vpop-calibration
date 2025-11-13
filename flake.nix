{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { nixpkgs, self, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = nixpkgs.outputs.legacyPackages.${system};
  in {
    devShells.default = pkgs.mkShell {
      buildInputs = with pkgs; [
        (poetry.override { python3 = python312; })
        gcc
        openssl
      ];
      shellHook = ''
        export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libz}/lib
      '';
    };
  });
}
