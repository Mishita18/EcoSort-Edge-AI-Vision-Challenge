# EcoSort — Intelligence SIG Edge AI submission workspace

## What is included
- Compact PyTorch depthwise-separable CNN (`src/model.py`)
- Trained checkpoint (`outputs/best_model.pth`)
- Public training/validation dataset (`data/public/`)
- Reproducible training script, ONNX export script, prediction script
- Official starter-kit verifier (`verify_submission.py`)

## Baseline model
The architecture has **14,350 parameters** (well below 350,000). A training run on the supplied 900-image training split selected a checkpoint with **100.0% validation accuracy on the supplied 240-image validation split**. This is a local validation result, not a guarantee of hidden-test accuracy; such a high score should be checked for dataset leakage or duplicates and independently reproduced before claiming it.

## Run
Create a Python environment and install dependencies:
```bash
pip install -r requirements.txt
```
Train from scratch:
```bash
python src/train.py --data_dir data/public --epochs 20 --batch_size 32 --lr 0.001
```
Export ONNX:
```bash
python src/export_onnx.py --checkpoint outputs/best_model.pth --output model.onnx
```
Verify using the organizer's checker:
```bash
python verify_submission.py --model model.onnx --val_dir data/public/val
```
Predict one image:
```bash
python src/predict.py --image path/to/image.jpg
```

## Critical compatibility note
The official verifier expects ONNX input in NCHW format `(Batch, 3, 96, 96)` and assumes ImageNet normalization happens before inference. The model therefore expects already-normalized float32 NCHW tensors. Use the supplied training/export code; do not move normalization into the model unless you also update the official evaluator.

The ONNX file is not included because this runtime lacks the `onnx` and `onnxruntime` packages and cannot install packages from the internet. Generate it in your environment with the export command above, then run the official verifier. Do not submit until export and all official checks pass.
