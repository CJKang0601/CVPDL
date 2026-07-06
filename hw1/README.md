# HW1 README

## Summary

This homework implements single-class object detection with `YOLOv8s`.
The code comments and existing submission notes indicate the model uses a
frozen backbone setup and produces a Kaggle-style `submission.csv`.

## Repository contents

- `assignment.pdf`: original homework handout
- `submission/report_RE6144051.pdf`: report
- `submission/code/src/model.py`: training and inference pipeline
- `submission/code/requirements.txt`: dependencies
- `submission/code/submission.csv`: prediction output

## Current verification status

- The submission artifact is present.
- The local dataset path referenced by the code exists on this machine:
  `C:\\Users\\cjkan\\Desktop\\CJK\\114-1\\CV\\HW1`
- `submission.csv` contains `1865` lines.
- The local test image folder contains `1864` images, which is consistent with
  a header row plus one prediction row per image.

## Notes on reproducibility

- The runnable entry point in the current submission is `src/model.py`.
- The code contains a hardcoded `BASE_PATH`; update it before rerunning on a
  different machine.
- I did not retrain the model here because the existing artifacts already look
  complete and retraining YOLO would be time-consuming.

## Suggested rerun command

```bash
python src/model.py
```
