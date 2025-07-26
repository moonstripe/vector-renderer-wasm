{
  description = "Vector-Renderer LibTorch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };

      # Update this path to wherever you extracted LibTorch
      libtorchPath = "/Users/${pkgs.lib.getEnv "USER"}/libtorch/libtorch";
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "libtorch-dev";

        buildInputs = [
          pkgs.clang
          pkgs.cmake
          pkgs.pkg-config
        ];

        shellHook = ''
          export LIBTORCH_PATH=${libtorchPath}
          export CPLUS_INCLUDE_PATH=$LIBTORCH_PATH/include:$CPLUS_INCLUDE_PATH
          export LIBRARY_PATH=$LIBTORCH_PATH/lib:$LIBRARY_PATH
          export DYLD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$DYLD_LIBRARY_PATH
          echo "✅ Dev shell ready. LibTorch path: $LIBTORCH_PATH"
        '';
      };
    };
}
