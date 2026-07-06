# HW3 README

## Summary

This homework implements an image generation pipeline based on a DDPM-style
U-Net model. The submission includes training scripts, generation scripts,
packaged zip files, and generated output images.

## Repository contents

- `assignment.pdf`: original homework handout
- `submission/hw3_RE6144051.pdf`: report
- `submission/code_RE6144051/`: source code
- `submission/code_RE6144051.zip`: packaged code submission
- `submission/img_RE6144051/outputs/`: generated images
- `submission/img_RE6144051.zip`: packaged image submission

## Current verification status

- The generated output folder contains `10000` images.
- The lightweight model smoke test passes on this machine.
- `test_model.py` runs successfully when the console encoding is set to UTF-8.
- The dataset path expected by `quick_check.py` is currently missing:
  `D:\\CJK\\114-1\\CV\\hw3_RE6144051\\hw3_mnist`

## What I checked

- Verified the source files exist and are not empty.
- Ran the model shape smoke test successfully.
- Confirmed that generated output images are already included in the repo.

## Reproducibility notes

- Full retraining was not rerun because the referenced local dataset path is
  missing.
- For Windows terminals that use a non-UTF-8 code page, set
  `PYTHONIOENCODING=utf-8` before running the test script if you want the emoji
  status output to print cleanly.

## Useful commands

Smoke test:

```bash
set PYTHONIOENCODING=utf-8
python test_model.py
```

Training:

```bash
python train.py
```
