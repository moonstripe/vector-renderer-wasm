{
  description = "Vector-Renderer LibTorch";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "aarch64-darwin";
      pkgs = import nixpkgs { inherit system; };

      # Update this path to wherever you extracted LibTorch
      libtorchPath = "/Users/kojinglick/libtorch/libtorch";
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "libtorch-dev";

        buildInputs = [
          pkgs.bzip2
          pkgs.clang
          pkgs.cmake
          pkgs.pkg-config
        ];

        shellHook = ''
          export LIBTORCH=${libtorchPath}
          export CPLUS_INCLUDE_PATH=$LIBTORCH_PATH/include:$CPLUS_INCLUDE_PATH
          export LIBRARY_PATH=$LIBTORCH_PATH/lib:$LIBRARY_PATH
          export DYLD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$DYLD_LIBRARY_PATH
          export LIBTORCH_BYPASS_VERSION_CHECK=1
          echo "✅ Dev shell ready. LibTorch path: $LIBTORCH_PATH"
        '';
      };
    };
}
