# HW2 README

## Summary

This homework implements long-tailed multi-class object detection with
`YOLOv8m`. The submitted code handles dataset conversion, class-aware
thresholding, training, validation, and prediction export.

## Repository contents

- `assignment.pdf`: original homework handout
- `submission/report_RE6I44051.pdf`: report
- `submission/code/src/model.py`: main training and inference script
- `submission/code/README.md`: original usage notes from the submission
- `submission/code/requirements.txt`: dependencies

## Current verification status

- The report and source code are present.
- The main implementation file is present and structurally complete.
- The code uses a hardcoded dataset path:
  `D:\\CJK\\114-1\\CV\\hw2_RE6144051\\CVPDL_hw2\\CVPDL_hw2`
- That dataset path is currently missing on this machine.

## Reproducibility assessment

- I did not rerun HW2 locally because the training / test dataset referenced by
  the code is not available at the expected path.
- Based on source inspection, the submission logic is substantially complete,
  but the repo does not currently include regenerated output weights or final
  prediction CSV artifacts.

## If you want to rerun later

1. Restore the dataset to the path expected by `src/model.py`, or edit
   `BASE_PATH`.
2. Create the environment from `requirements.txt`.
3. Run the main script:

```bash
python src/model.py
```
