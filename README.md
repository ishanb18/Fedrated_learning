# Federated Learning for Remote Sensing

Simulation of **Federated Learning (FedAvg)** on aerial/satellite image classification using PyTorch. The project supports the **UC Merced Land Use** dataset (21 classes) and **NWPU-RESISC45** (45 classes), with optional checkpointing and full metrics (Accuracy, F1, AUC).

## Features

- **Federated Averaging (FedAvg)** — clients train locally; server aggregates model updates
- **IID data partition** — training data split across simulated clients
- **Metrics** — global Accuracy, macro F1, and macro AUC on a held-out test set
- **Checkpointing** — save and resume training (supported in main scripts)
- **Plots** — global metrics vs. rounds, per-client accuracy, cumulative communication cost

## Requirements

- Python 3.8+
- PyTorch 2.x (CPU or CUDA)
- See [requirements.txt](requirements.txt) for full dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest of the requirements.

## Datasets

| Dataset | Classes | Notes |
|--------|---------|--------|
| **UC Merced Land Use** | 21 | Single folder of class subdirs (e.g. `UCMerced_LandUse/Images/`). Scripts split train/test internally. |
| **NWPU-RESISC45** | 45 | Expects `train/` and `test/` folders under e.g. `data/NWPU-RESISC45/`. |

- **UC Merced**: [UCMerced Dataset](http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.html) — 2,100 images, 256×256.
- **NWPU-RESISC45**: Place the dataset so that `data/NWPU-RESISC45/train` and `data/NWPU-RESISC45/test` exist, each with one subfolder per class.

See [database_discriptions.md](database_discriptions.md) for more dataset details.

## Project Structure

| Path | Description |
|------|-------------|
| **notebooks/** | Jupyter notebooks: `ucm_fl_final.ipynb` (UC Merced), `colab_fed.ipynb`, `favg_eurosat.ipynb`. |
| **scripts/** | Python scripts: `newfed.py`, `newfed1.py` (NWPU-RESISC45 FL with checkpointing). |
| **report/** | Report: add your report PDF here. |
| **results/** | Result images (`.png`): accuracy plots, per-client accuracy, communication cost, etc. |
| **models/** | Trained model checkpoints (`.pth`): add best checkpoints here to share on GitHub. |
| **requirements.txt** | Python dependencies. |
| **database_discriptions.md** | Short descriptions of supported datasets. |

## How to Run

### UC Merced (notebook)

1. Place the UC Merced dataset so that the path to the **Images** folder (with 21 class subfolders) is correct in the notebook (e.g. `./UCMerced_LandUse/Images`).
2. Open `notebooks/ucm_fl_final.ipynb` and run all cells.

### NWPU-RESISC45 (script)

1. From the **project root**, create `data/NWPU-RESISC45/train` and `data/NWPU-RESISC45/test` with one subfolder per class.
2. In `scripts/newfed.py`, set `TRAIN_DATA_PATH` and `TEST_DATA_PATH` if your paths differ (paths are relative to project root).
3. Run from project root:

```bash
python scripts/newfed.py
```

Checkpoints are saved to `fl_checkpoint.pth` in the project root by default; to save into `models/`, set `CHECKPOINT_PATH = './models/fl_checkpoint.pth'` in the script. Re-running continues from the last round if the checkpoint exists.

## Configuration (typical)

- **NUM_CLIENTS**: 5–10
- **NUM_ROUNDS**: 20–30
- **LOCAL_EPOCHS**: 1–6 (lower can reduce client overfitting)
- **BATCH_SIZE**: 8–16
- **LEARNING_RATE**: ~0.001

Adjust these in the top of the script or notebook.

## Report & Results

- Put your **project report** (e.g. PDF) and result **plots/tables** in the [**report/**](report/) folder.
- Use [**report/RESULTS.md**](report/RESULTS.md) to summarize metrics (accuracy, F1, AUC) and link to any images.
- Files in `report/` (PDF, PNG, MD) are tracked and pushed to GitHub.

## Model

- Put your **trained model checkpoints** (`.pth` / `.pt`) in the [**models/**](models/) folder.
- Checkpoints in `models/` are tracked and pushed (GitHub file size limit: 100 MB; use [Git LFS](https://git-lfs.github.com/) for larger files).
- To save from a script into `models/`, set e.g. `CHECKPOINT_PATH = './models/fl_ucmerced_final.pth'` and run from the project root.

## License

This project is for educational and research use. Respect the license terms of the datasets (UC Merced, NWPU-RESISC45) when downloading and using them.
