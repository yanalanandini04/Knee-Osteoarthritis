# AI-Based Knee Osteoarthritis Grading and Rehabilitation Assistance System

A final-year project reference implementation using PyTorch EfficientNet-B0 for KL grading, Grad-CAM explanations, a JSON exercise mapping, and rule-based MediaPipe Pose feedback.

## Safety and scope

This is an educational decision-support prototype, not a diagnostic tool or a replacement for a doctor or physiotherapist. Model output must be validated by a qualified professional. Exercise should stop if pain, dizziness, or other concerning symptoms occur.

## Dataset layout

Place the Knee Osteoarthritis Dataset under `data/` in this structure. Folder names may be `0` through `4`, or `class_0` through `class_4`.

```text
data/
  train/<grade>/*.png
  val/<grade>/*.png
  test/<grade>/*.png
```

The loader also accepts `validation` instead of `val`. It does not create or invent images.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train and evaluate

```powershell
python -m src.train --data-dir data --epochs 15 --batch-size 32 --output-dir checkpoints
python -m src.evaluate --data-dir data --checkpoint checkpoints/best.pt --output-dir artifacts
python -m pytest -q
```

Training uses ImageNet initialization when available, training-only augmentation, inverse-frequency class weights, and reports accuracy, macro precision/recall/F1, a confusion matrix, and quadratic weighted kappa. Evaluation writes `metrics.json` and `confusion_matrix.png`.

## Run the application

```powershell
streamlit run app.py
```

Upload a knee X-ray after training to receive the predicted KL grade, severity label, confidence, and Grad-CAM. Then choose an exercise, open its instructional video, and start camera monitoring. Camera monitoring requires a local camera and the installed MediaPipe package.

## Notes

- The app intentionally shows a clear setup message until a real trained checkpoint exists.
- The monitoring rules are transparent heuristics for a student prototype: they estimate knee angle from visible landmarks and count controlled repetitions. They are not clinical movement assessments.
- LSTM is intentionally omitted; it can be evaluated later as a separate research extension.
