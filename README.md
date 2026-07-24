# ReFuseGait Event–Gray Gait Recognition

This repository contains the ReFuseGait implementation for multimodal gait recognition. The model processes synchronized person bounding boxes, event representations, and grayscale frames.

This document describes environment setup, dataset preparation, pretrained-model placement, training, and evaluation. It also summarizes the experimental protocols used for DAVIS346-Gait, DAVIS346-Gait-RGE, and EV-CASIA-B.

## 1. Repository Structure

```text
configs/
├── default.yaml
├── refusegait_davis346_day.yaml
└── final_retest/
    └── refusegait_davis346_day_eval.yaml
datasets/
└── HNU-Gait/HNU-Gait.json
opengait/                         Model, data loading, training, and evaluation
requirements-cu121.txt            PyTorch 2.1.2 with CUDA 12.1
requirements.txt                  Other Python dependencies
train.sh                          Distributed training launcher
test.sh                           Distributed evaluation launcher
```

Datasets, pretrained models, outputs, and experiment checkpoints are not stored in the repository.

## 2. Requirements

The recommended configuration is:

- Linux;
- Python 3.10;
- PyTorch 2.1.2;
- CUDA 12.1;
- two NVIDIA GPUs with at least 24 GB of memory each.

Check the NVIDIA driver and available GPUs:

```bash
nvidia-smi
```

The NVIDIA driver must support the CUDA runtime used by the installed PyTorch package. If CUDA 12.1 is not supported, install a compatible PyTorch build instead of using `requirements-cu121.txt` directly.

## 3. Conda Environment

Create and activate a Conda environment:

```bash
conda create -n refusegait python=3.10 -y
conda activate refusegait

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-cu121.txt
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python - <<'PY'
import torch
import torchvision
import numpy
import scipy
import yaml

print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("PyYAML:", yaml.__version__)

assert torch.cuda.is_available(), "CUDA is unavailable in PyTorch"
PY
```

xFormers is optional. The message `xFormers not available` does not prevent training or evaluation.

## 4. Dataset Preparation

### 4.1 Processed DAVIS346-Gait Data

The provided training configuration reads the Day/Light split from:

```text
./db/HNU-Gait/light
```

If the processed dataset is stored at `/path/to/HNU-Gait`, create a symbolic link:

```bash
DATASET_ROOT=/path/to/HNU-Gait

mkdir -p db
ln -sfn "$(readlink -f "$DATASET_ROOT")" db/HNU-Gait

test -d db/HNU-Gait/light
readlink -f db/HNU-Gait/light
```

The expected hierarchy is:

```text
db/HNU-Gait/light/
└── <identity>/
    └── <sequence>/
        └── <view>/
            ├── bboxes.pkl
            ├── continuous.npz
            └── gray.npz
```

The three inputs decoded by the data loader are:

```text
bbox:  (T, 4)
Event: (T, 8, 64, 64)
Gray:  (T, 3, 64, 64)
```

The three modalities must be temporally aligned and have the same number of frames. Avoid unrelated `.pkl` or `.npz` files in a sequence directory because the loader selects modalities according to sorted file order. In particular, a file such as `count.npz` between `continuous.npz` and `gray.npz` can replace the intended Gray input.

### 4.2 Starting from Raw DAVIS346-Gait Data

This repository contains training and evaluation code but does not include the raw-data preprocessing implementation. Use the preprocessing pipeline from the official EdinoGait repository in a separate working directory.

Official resources:

- EdinoGait repository: <https://github.com/C19h/EdinoGait>
- DAVIS346-Gait dataset: <https://pan.baidu.com/s/1joc62krCik4rsoItncdlRA?pwd=yy26> (extraction code: `yy26`)
- Official pretrained models: <https://drive.google.com/drive/folders/1blvrnbZP3IPNOOYhQzdz5Ml4z-vnKBb9?usp=drive_link>

Clone the upstream repository:

```bash
WORK_ROOT=/path/to/preprocessing_workspace
RAW_DATASET_ROOT=/path/to/HNU-Gait

mkdir -p "$WORK_ROOT"
cd "$WORK_ROOT"
git clone --depth 1 https://github.com/C19h/EdinoGait.git
cd EdinoGait

mkdir -p db
ln -sfn "$(readlink -f "$RAW_DATASET_ROOT")" db/HNU-Gait
```

Download the complete upstream `pretrained/` directory from the official link and place it at:

```text
/path/to/preprocessing_workspace/EdinoGait/pretrained/
```

Activate the Conda environment and run the official preprocessing stages from the upstream EdinoGait root:

```bash
conda activate refusegait
cd /path/to/preprocessing_workspace/EdinoGait

python datasets/bboxes.py
python datasets/preprocess.py
python datasets/to_cef.py
```

After preprocessing, link the resulting `HNU-Gait` directory into this repository as `db/HNU-Gait`. Each sequence used by ReFuseGait must provide `bboxes.pkl`, `continuous.npz`, and `gray.npz` in the format described above.

Before running the scripts, inspect the upstream scripts for their `root_dir` and pretrained-weight settings because these details may change between upstream revisions.

## 5. Pretrained Edino Model

ReFuseGait initialization requires the pretrained Edino model named exactly `Edino.pt`.

Download it from the **official EdinoGait Google Drive folder**:

<https://drive.google.com/drive/folders/1blvrnbZP3IPNOOYhQzdz5Ml4z-vnKBb9?usp=drive_link>

Open the link in a browser, locate `Edino.pt`, select it, and choose **Download**. Google Drive does not provide a stable direct `wget` URL for this shared folder, so browser download is the recommended method. Do not rename another checkpoint to `Edino.pt`.

After downloading it, transfer the file to the machine running ReFuseGait. The final repository layout must be:

```text
ReFuseGait/
└── pretrained/
    └── Edino.pt
```

If the downloaded file is already available at `/absolute/path/to/Edino.pt`, either copy it into the repository:

```bash
cd /path/to/ReFuseGait
mkdir -p pretrained
cp /absolute/path/to/Edino.pt pretrained/Edino.pt
```

or create a symbolic link without duplicating the file:

```bash
cd /path/to/ReFuseGait
EDINO_WEIGHT=/absolute/path/to/Edino.pt

mkdir -p pretrained
ln -sfn "$(readlink -f "$EDINO_WEIGHT")" pretrained/Edino.pt

```

Confirm that the path resolves to a regular model file:

```bash

test -f pretrained/Edino.pt
readlink -f pretrained/Edino.pt
ls -lh pretrained/Edino.pt
```

Optionally verify that PyTorch can deserialize it:

```bash
python - <<'PY'
from pathlib import Path
import torch

path = Path("pretrained/Edino.pt")
assert path.is_file(), path
checkpoint = torch.load(path, map_location="cpu")
print("Loaded:", path)
print("Object type:", type(checkpoint).__name__)
if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint)[:20])
PY
```

If `torch.load` reports an HTML, ZIP, or invalid-pickle error, the downloaded item is not the actual `Edino.pt` model file. Download the file itself again from the official Google Drive folder.

The training configuration should contain:

```yaml
model_cfg:
  pretrained_edinov2: ./pretrained/Edino.pt
```

Verify the main paths before training:

```bash
python - <<'PY'
from pathlib import Path
import yaml

config_path = Path("configs/refusegait_davis346_day.yaml")
with config_path.open() as f:
    cfg = yaml.safe_load(f)

paths = {
    "training data": cfg["data_cfg"]["dataset_root_train"],
    "evaluation data": cfg["data_cfg"]["dataset_root_eval"],
    "partition file": cfg["data_cfg"]["dataset_partition"],
    "Edino weights": cfg["model_cfg"]["pretrained_edinov2"],
}

for name, value in paths.items():
    exists = Path(value).exists()
    print(f"{name}: {value} exists={exists}")
    assert exists, (name, value)

assert cfg["data_cfg"]["data_in_use"][:3] == [True, True, True]
print("Input data: bbox + Event + Gray")
PY
```

## 6. Training

Make the launchers executable:

```bash
chmod +x train.sh test.sh
```

### 6.1 Foreground Training

Run the standard 40,000-iteration training configuration on two GPUs:

```bash
conda activate refusegait
mkdir -p logs

./train.sh configs/refusegait_davis346_day.yaml 0,1 29501 \
  2>&1 | tee logs/refusegait_davis346_day.log
```

The second argument specifies visible GPU indices, and the third specifies the distributed communication port.

### 6.2 Background Training

```bash
conda activate refusegait
mkdir -p logs

nohup ./train.sh configs/refusegait_davis346_day.yaml 0,1 29501 \
  > logs/refusegait_davis346_day.log 2>&1 &

echo $! > logs/refusegait_davis346_day.pid
```

Monitor the log:

```bash
tail -f logs/refusegait_davis346_day.log
```

Display recent training iterations and evaluation results:

```bash
grep -E 'Iteration [0-9]+' logs/refusegait_davis346_day.log | tail

grep -E 'NM@R1:.*BG@R1:.*CL@R1:.*PT@R1:' \
  logs/refusegait_davis346_day.log | tail
```

The default configuration trains for 40,000 iterations and saves a checkpoint every 1,000 iterations. Checkpoints are written to:

```text
output/HNU-Gait/ReFuseGait/
└── refusegait_davis346_day/checkpoints/
    └── refusegait_davis346_day-xxxxx.pt
```

## 7. Evaluation with a Checkpoint

Set `CHECKPOINT` to the absolute path of the ReFuseGait checkpoint to evaluate. The following command creates a local evaluation configuration without modifying the tracked configuration:

```bash
conda activate refusegait

CHECKPOINT=/absolute/path/to/model.pt
export CHECKPOINT

test -f "$CHECKPOINT"

python - <<'PY'
from pathlib import Path
import os
import yaml

source = Path("configs/final_retest/refusegait_davis346_day_eval.yaml")
target = Path("configs/local_test_checkpoint.yaml")
checkpoint = str(Path(os.environ["CHECKPOINT"]).resolve())

with source.open() as f:
    cfg = yaml.safe_load(f)

cfg["trainer_cfg"]["restore_hint"] = checkpoint
cfg["evaluator_cfg"]["restore_hint"] = checkpoint

with target.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Evaluation config:", target)
print("Checkpoint:", checkpoint)
PY

mkdir -p logs
./test.sh configs/local_test_checkpoint.yaml 0,1 29601 \
  2>&1 | tee logs/test_checkpoint.log
```

Extract the final results:

```bash
grep -E 'Rank-1|NM@R1|BG@R1|CL@R1|PT@R1|Traceback|Error' \
  logs/test_checkpoint.log | tail -30
```


## 8. Experimental Protocols

### 8.1 DAVIS346-Gait

- Identities `01`--`31` are used for training.
- Identities `32`--`41` are used for evaluation.
- Day/Light and Dark conditions are evaluated separately.
- The reported conditions are normal walking (NM), carrying a bag (BG), clothing variation (CL), and other walking-pattern variation (PT).
- Rank-1 accuracy is reported after excluding identical-view gallery/probe pairs.
- The ReFuseGait Day/Light experiment uses synchronized bbox, Event, and Gray inputs.

The partition used by the included configuration is stored in:

```text
datasets/HNU-Gait/HNU-Gait.json
```

### 8.2 DAVIS346-Gait-RGE

- The protocol uses a 21/20 identity split.
- Identities `01`--`21` are used for training.
- Identities `22`--`41` are used for evaluation.
- Day and Dark conditions are evaluated separately.
- NM, BG, CL, and PT Rank-1 accuracies are reported.
- Identical-view gallery/probe pairs are excluded.
- The reference partition files are `RGE-Day-21-20.json` and `RGE-Night-21-20.json` in the full experimental codebase.

### 8.3 EV-CASIA-B

- The pretraining stage uses 74 identities.
- The fine-tuning and closed-set evaluation stage uses 50 identities, corresponding to identities `075`--`124` in the processed split.
- Sequences `nm-01`, `nm-02`, `nm-03`, and `nm-04` are used for fine-tuning.
- Sequences `nm-05` and `nm-06` are used for evaluation.
- Evaluation covers 11 views: `000`, `018`, `036`, `054`, `072`, `090`, `108`, `126`, `144`, `162`, and `180` degrees.
- The reported metrics are overall Top-1 accuracy, mean 11-view Top-1 accuracy, and overall Top-5 accuracy.
- The ReFuseGait experiment uses synchronized bbox, Event, and Gray inputs.

The configuration and partition files for DAVIS346-Gait-RGE and EV-CASIA-B are not distributed in this repository.

## Acknowledgements

ReFuseGait is implemented on top of the following open-source research projects. We thank their authors for making the code and pretrained models available:

- **OpenGait**, associated with *OpenGait: Revisiting Gait Recognition Toward Better Practicality* (Fan et al., CVPR 2023): <https://github.com/ShiqiYu/OpenGait>.
- **EdinoGait**, associated with *Fusing Events and Frames for Robust Gait Recognition*: <https://github.com/C19h/EdinoGait>. ReFuseGait uses its event-based gait-recognition pipeline, preprocessing workflow, and pretrained Edino model as the starting point.
- **DINOv2**, associated with *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., TMLR 2024): <https://github.com/facebookresearch/dinov2>.

The upstream projects remain under their respective licenses. ReFuseGait does not claim ownership of their original implementations, pretrained weights, or datasets. Users should follow the citation and license requirements of OpenGait, EdinoGait, DINOv2, and each dataset when redistributing or publishing results.
# RefuseGait
