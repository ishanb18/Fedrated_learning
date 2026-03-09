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

| File | Description |
|------|-------------|
| **ucm_fl_final.ipynb** | UC Merced FL notebook — 5 clients, 25 rounds, 6 local epochs; good for quick runs and visualization. |
| **newfed.py** | NWPU-RESISC45 FL script — SimpleCNN, checkpoint at `fl_checkpoint.pth`, configurable rounds. |
| **newfed1.py** | Variant of the NWPU-RESISC45 FL pipeline. |
| **colab_fed.ipynb** | Federated learning notebook (Colab-friendly). |
| **favg_eurosat.ipynb** | FedAvg experiment on EuroSAT-style setup. |
| **requirements.txt** | Python dependencies. |
| **database_discriptions.md** | Short descriptions of supported datasets. |

## How to Run

### UC Merced (notebook)

1. Place the UC Merced dataset so that the path to the **Images** folder (with 21 class subfolders) is correct in the notebook (e.g. `./UCMerced_LandUse/Images`).
2. Open `ucm_fl_final.ipynb` and run all cells.

### NWPU-RESISC45 (script)

1. Create `data/NWPU-RESISC45/train` and `data/NWPU-RESISC45/test` with one subfolder per class.
2. In `newfed.py`, set `TRAIN_DATA_PATH` and `TEST_DATA_PATH` if your paths differ.
3. Run:

```bash
python newfed.py
```

Checkpoints are saved to `fl_checkpoint.pth`; re-running continues from the last round if the file exists.

## Configuration (typical)

- **NUM_CLIENTS**: 5–10
- **NUM_ROUNDS**: 20–30
- **LOCAL_EPOCHS**: 1–6 (lower can reduce client overfitting)
- **BATCH_SIZE**: 8–16
- **LEARNING_RATE**: ~0.001

Adjust these in the top of the script or notebook.

## License

This project is for educational and research use. Respect the license terms of the datasets (UC Merced, NWPU-RESISC45) when downloading and using them.
