# Federated Learning for Remote Sensing Image Classification

This project implements a **Federated Learning framework for remote sensing image classification** using a lightweight Convolutional Neural Network (CNN).  
The system allows multiple clients to collaboratively train a global model **without sharing their raw data**, preserving data privacy while maintaining strong performance.

---

# Project Overview

Remote sensing imagery is widely used in applications such as:

- Environmental monitoring
- Land use classification
- Disaster management
- Agricultural analysis

Traditional machine learning approaches require **centralized datasets**, which can introduce issues such as:

- Data privacy concerns
- High communication cost
- Data ownership restrictions
- Storage limitations

**Federated Learning (FL)** solves these challenges by allowing distributed clients to train models locally while sharing only model parameters with a central server.

The server aggregates these updates to build a **global model that benefits from all clients’ knowledge without accessing their data.**

---

# Key Features

- Federated Learning implementation using **FedAvg**
- Lightweight **CNN architecture**
- Support for **multiple remote sensing datasets**
- Simulation of **non-IID data distribution**
- Performance evaluation using:
  - Accuracy
  - F1 Score
  - AUC
  - Communication Cost

---

# Datasets

The project evaluates federated learning across three popular remote sensing datasets.

| Dataset | Classes | Images | Resolution |
|--------|--------|--------|------------|
| EuroSAT | 10 | 27,000 | 64×64 |
| UC Merced | 21 | 2,100 | 256×256 |
| NWPU-RESISC45 | 45 | 31,500 | 256×256 |

### Dataset Characteristics

**EuroSAT**
- Satellite imagery dataset
- Based on Sentinel-2 satellite images

**UC Merced**
- High quality aerial scene dataset
- Suitable for small-scale experiments

**NWPU-RESISC45**
- Large-scale remote sensing scene dataset
- Highly diverse scene categories

---

# Federated Learning Architecture

The system follows a **server-client federated architecture**.

### Training Workflow

1. Server initializes the global model.
2. The global model is sent to all clients.
3. Each client trains the model locally on its private dataset.
4. Clients send updated model weights to the server.
5. Server aggregates updates using **Federated Averaging (FedAvg)**.
6. The updated global model is redistributed to clients.
7. Steps repeat for multiple communication rounds until convergence.

Important property:

**Raw data never leaves the client devices.**

---

# CNN Model Architecture

A lightweight CNN is used to balance **performance and communication efficiency**.
Input Image (3 × 224 × 224)

Conv1 → 16 filters + MaxPool
Conv2 → 32 filters + MaxPool
Conv3 → 64 filters + MaxPool
Conv4 → 128 filters + MaxPool

Flatten

Fully Connected Layer (512)


Total parameters ≈ **13 million** depending on dataset classes.

---

# Federated Averaging (FedAvg)

The global model is updated using weighted averaging of client updates.

W_global = Σ (n_i / Σ n_j) * W_i


Where:

- `W_i` = model weights from client i  
- `n_i` = number of samples on client i  
- `K` = total number of clients  

Clients with larger datasets contribute more to the global model.

---

# Experimental Setup

### Environment

- Python 3.8
- PyTorch
- NumPy
- scikit-learn

### Training Parameters

| Parameter | Value |
|----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 16–32 |
| Local Epochs | 1–5 |

### Communication Rounds

| Dataset | Rounds |
|--------|-------|
| EuroSAT | 20 |
| UC Merced | 28 |
| NWPU-RESISC45 | 92 |

---

# Results

### EuroSAT

| Metric | Value |
|------|------|
Accuracy | 0.88 |
F1 Score | 0.88 |
AUC | 0.99 |

The model converges quickly within **20 communication rounds**.

---

### NWPU-RESISC45

| Metric | Value |
|------|------|
Accuracy | 0.73 – 0.74 |
F1 Score | 0.74 |
AUC | 0.98 – 0.99 |

Higher dataset complexity requires **more training rounds**.

---

### UC Merced

| Metric | Value |
|------|------|
Accuracy | 0.76 – 0.79 |
F1 Score | 0.74 |
AUC | 0.98 – 0.99 |

The dataset is smaller and converges faster.
