## Local Integration Notes

- `external-dialects/stablehlo` no longer needs local worktree patches.
- A trackable alternative was validated by moving the submodule to:
  - `0dc0fd71ccb5` (`Integrate LLVM at llvm/llvm-project@5e14916fa6ab`)

- Repo-tracked integration updates live in `approxMLIR` itself:
  - `external-tools/approx-triton-plugin/CMakeLists.txt`
  - `external-tools/approx-triton-plugin/pass/CMakeLists.txt`
  - `external-tools/approx-triton-plugin/pass/ApproxTritonPlugin.cpp`

- Local-only Triton state that is not represented by commits:
  - `triton` stashes:
    - `stash@{0}: codex-pre-llvm-head-switch-20260418`
    - `stash@{1}: codex-pre-update-triton-2026-04-18`

- Runtime note:
  - current Triton plugin loading uses `TRITON_PLUGIN_PATHS`
  - existing ApproxMLIR runtime code still mainly writes `TRITON_PASS_PLUGIN_PATH`
  - for now, export both to run the plugin reliably

- Python environment note:
  - installing packages that depend on `torch` can cause `pip` to reinstall the
    PyPI `triton==3.2.0` wheel into `.venv`
  - that breaks the local source-Triton workflow (`triton.knobs` disappears)
  - recovery command:
    - `source .venv/bin/activate && MAX_JOBS=12 TRITON_EXT_ENABLED=ON LLVM_SYSPATH=/home/hao/.triton/llvm/llvm-87717bf9-ubuntu-x64 python -m pip install -e /home/hao/triton-trails/triton --no-build-isolation`
