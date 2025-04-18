# An out-of-tree dialect template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a
approxMLIR `opt`-like tool to operate on that dialect.

## How to build

First, clone the LLVM to a directory called `llvm-project` and initialize the submodules:

```sh
git clone git@github.com:llvm/llvm-project.git
cd llvm-project
# This is the LLVM commit we will use 
git checkout 26eb4285b56edd8c897642078d91f16ff0fd3472
```

Build MLIR, LLVM, and Clang

```sh
git submodule update --init --recursive --progress
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG 
ninja
ninja check-mlir
```

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$PWD/../llvm-project/build/bin/llvm-lit
cmake --build . --target check-approxMLIR-opt
```
To build the documentation from the TableGen description of the dialect
operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with
CMake so that it installs `FileCheck` to the chosen installation prefix.

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
