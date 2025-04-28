# approxMLIR-IREE testing suite

## Building the testing suite
You should first install `IREE` in your environment.

```bash
# create a venv
python3 -m venv venv
source venv/bin/activate
```
Refer to this [iree versions](https://iree.dev/developers/general/release-management/) to choose the right version to download.
```bash
pip install iree-base-compiler==2.9.0
pip install iree-base-runtime==2.9.0
pip install iree-tools-tf==20241108.1073
```

## Run the test
The `substitute` provide a run-time library for you to do function substitution.

Your approximate app should use such library, to approximte, substitute, compile and run the app. 

One application that replaces the DNN + MNIST is here
```bash
python3 test_substitute_mnist.py
```

Write your own application by following the example in `test_substitute_mnist.py`.

