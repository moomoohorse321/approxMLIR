# Diffusion Smoke Data

This directory contains a tiny real-prompt smoke profile for the laptop exact
benchmark. The prompts are adapted from the public `nateraw/parti-prompts`
PartiPrompts dataset and are stored as request JSONL rows so the benchmark can
run without network access after model staging.

The committed profiles are intentionally small:

- `smoke_requests.jsonl`: 4 text-to-image requests, 512x512, batch-1 friendly.
- `latency_memory_bound_requests.jsonl`: 8 text-to-image requests, 64x64 with
  2 denoise steps, paired with the runner's large prompt-conditioning tower.
  Each request scores 32 conditioning candidates, which gives a small realistic
  serving batch without hiding weight traffic behind large-batch compute.

Large prompt shards, generated images, model snapshots, exact goldens,
calibration artifacts, and approximate outputs should stay out of git.
