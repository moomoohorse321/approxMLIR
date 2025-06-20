# ApproxMLIR 

ApproxMLIR is an MLIR dialect for representing approximate computations for various applications. It is designed to facilitate the development of algorithms that can tolerate some level of approximation, which can lead to improved performance and reduced resource usage in various applications.

## How to build

First, clone the LLVM to a directory called `llvm-project` and initialize the submodules and build Clang, MLIR, and LLVM

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

## Developer

* Add a pass : see [commit link](https://github.com/moomoohorse321/approxMLIR/commit/ff1bdb01154f5190e3c836aea376de3b870b870f)

## License

This dialect template is made available under the Apache License 2.0 with LLVM Exceptions. See the `LICENSE.txt` file for more details.
