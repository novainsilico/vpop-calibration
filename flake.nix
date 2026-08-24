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
    devShells = {
      default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python313
          uv
          gcc
          openssl
          libz
          ruff
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libz}/lib
          python -m venv .venv-full
          source .venv-full/bin/activate
          export UV_PROJECT_ENVIRONMENT=.venv-full
          uv sync --group dev --group test --active --extra full
        '';
      };
      slim = pkgs.mkShell {
        buildInputs = with pkgs; [
          python313
          uv
          gcc
          openssl
          libz
          ruff
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libz}/lib
          python -m venv .venv-slim
          source .venv-slim/bin/activate
          export UV_PROJECT_ENVIRONMENT=.venv-slim
          uv sync --group test_slim
        '';
      };
    };
  });
}
