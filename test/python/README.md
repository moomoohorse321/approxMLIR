# approxMLIR-IREE testing suite

The testing suite is completely independent (modular building block) to the parent project. The way of testing is 
* generate the MLIR file from any front-ends / compile mlir using this tesing suite
* optimize it using the parent project
* pass the mlir file to the current testing suite
* load and run it using IREE runtime



## Building the testing suite
You should first install `IREE` in your environment.

```bash
# create a venv
python3 -m venv venv
source venv/bin/activate
```
Refer to this [iree versions](https://iree.dev/developers/general/release-management/) to choose the right version to download.
```bash
pip install iree-base-compiler==3.3.0
pip install iree-base-runtime==3.3.0
pip install iree-tools-tf==20250320.1206
```

## Run the test
The `substitute` provide a run-time library for you to do function substitution.

Your approximate app should use such library, to approximte, substitute, compile and run the app. 

One application that replaces the DNN + MNIST is here
```bash
python3 test_substitute_mnist.py
```

Write your own application by following the example in `test_substitute_mnist.py`.

