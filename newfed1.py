import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision
import torchvision.transforms as transforms
import torchmetrics
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import copy
import sys

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# !! IMPORTANT !!
# Update this path to point to your *unzipped* 'NWPU-RESISC45' dataset folder
# See README.md for instructions on directory structure.
TRAIN_DATA_PATH = './data/NWPU-RESISC45/train' # Path to the 'train' folder
TEST_DATA_PATH = './data/NWPU-RESISC45/test'   # Path to the 'test' folder

# --- Checkpoint Path ---
CHECKPOINT_PATH = './fl_checkpoint_check.pth' # File to save/load model and logs

# --- Hyperparameters ---
NUM_CLIENTS = 10        # Number of clients to simulate
NUM_ROUNDS = 1          # Number of communication rounds
LOCAL_EPOCHS = 1        # Number of local training epochs on client data
BATCH_SIZE = 16         # A good balance of speed and memory
LEARNING_RATE = 0.001   # A standard, fixed learning rate
CLIENT_TEST_SPLIT = 0.2 # Each client holds out 20% of its data for local testing
NUM_CLASSES = 45        # NWPU-RESISC45 has 45 classes

# --- System Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Running on device: {DEVICE}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================

class SimpleCNN(nn.Module):
    """
    A simple 4-layer CNN architecture for 224x224 images and 45 classes.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        # Input: (batch, 3, 224, 224)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # -> (batch, 16, 112, 112)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # -> (batch, 32, 56, 56)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # -> (batch, 64, 28, 28)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ) # -> (batch, 128, 14, 14)
        
        self.flatten = nn.Flatten()
        
        # Calculate flattened size: 128 * 14 * 14 = 25,088
        self.fc = nn.Sequential(
            nn.Linear(128 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        print(f"Initialized SimpleCNN (4-layer) (FC in_features: {128*14*14}).")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

# ==============================================================================
# 3. DATA PREPARATION & PARTITIONING
# ==============================================================================

def get_dataloaders(train_path, test_path, num_clients):
    """
    Loads the NWPU-RESISC45 dataset from pre-split 'train' and 'test' folders
    and partitions the training data among clients in an IID fashion.
    """
    if not os.path.exists(train_path):
        print(f"Error: Training data path not found at '{train_path}'")
        raise FileNotFoundError(f"Training path not found: {train_path}")
    if not os.path.exists(test_path):
        print(f"Error: Test data path not found at '{test_path}'")
        raise FileNotFoundError(f"Test path not found: {test_path}")

    # Data Augmentation for Training Data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])
    
    # Test Transform (No Augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
    ])

    # Load the entire TRAINING dataset with augmentation
    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    
    # Load the entire TEST dataset without augmentation
    global_test_dataset = ImageFolder(root=test_path, transform=test_transform)
    global_test_loader = DataLoader(
        global_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    pool_size = len(train_dataset)
    client_pool_indices = list(range(pool_size))
    np.random.shuffle(client_pool_indices)
    
    # Partition client pool indices among clients (IID)
    client_indices = np.array_split(client_pool_indices, num_clients)
    
    client_loaders = [] # List of (train_loader, test_loader)
    client_data_sizes = [] # List of train data sizes for FedAvg
    
    for i in range(num_clients):
        client_idx = client_indices[i]
        
        # Split this client's data into their local train and local test
        local_test_size = int(len(client_idx) * CLIENT_TEST_SPLIT)
        local_train_size = len(client_idx) - local_test_size
        
        # Ensure indices are shuffled before splitting
        np.random.shuffle(client_idx)
        
        local_train_idx = client_idx[:local_train_size]
        local_test_idx = client_idx[local_train_size:]
        
        # Create subsets: local_train_dataset uses the original (augmented) transform
        local_train_dataset = Subset(train_dataset, local_train_idx)
        
        # NOTE: For simplicity in this environment, local_test_dataset reuses 
        # the augmented training dataset object as its base.
        local_test_dataset = Subset(train_dataset, local_test_idx)
        
        # Create dataloaders
        train_loader = DataLoader(
            local_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            local_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        
        client_loaders.append((train_loader, test_loader))
        client_data_sizes.append(local_train_size)
        
    print(f"Data loaded: {pool_size} TRAIN samples for {num_clients} clients, {len(global_test_dataset)} for global test.")
    return global_test_loader, client_loaders, client_data_sizes

# ==============================================================================
# 4. FEDERATED LEARNING CORE (CLIENT CLASS)
# ==============================================================================

class Client:
    """
    Simulates a single client device.
    """
    def __init__(self, client_id, model_template, train_loader, test_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Each client gets a *copy* of the model architecture
        self.model = copy.deepcopy(model_template).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        
        # Per-client metrics
        self.acc_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=NUM_CLASSES
        ).to(device)
        self.train_acc_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes=NUM_CLASSES
        ).to(device)

    def set_parameters(self, global_state_dict):
        """Load global model weights."""
        self.model.load_state_dict(global_state_dict)

    def train(self, local_epochs):
        """Train the model on local data."""
        self.model.train()
        print(f"  > Client {self.client_id}: Starting local training...")
        
        for epoch in range(local_epochs):
            self.train_acc_metric.reset()
            total_epoch_loss = 0.0
            num_batches = 0
            
            try:
                for batch_idx, (images, labels) in enumerate(self.train_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    total_epoch_loss += loss.item()
                    self.train_acc_metric.update(outputs, labels)
                    num_batches += 1
            
            except Exception as e:
                print(f"    ! Client {self.client_id} Error: Failed to process a batch. Skipping. Error: {e}")
                pass

            if num_batches > 0:
                avg_epoch_loss = total_epoch_loss / num_batches
                epoch_train_acc = self.train_acc_metric.compute().item()
                print(f"    > Client {self.client_id} Epoch {epoch+1}/{local_epochs} Summary: Avg Loss: {avg_epoch_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            else:
                print(f"    > Client {self.client_id} Epoch {epoch+1}/{local_epochs} Summary: No batches processed successfully.")

        print(f"  > Client {self.client_id}: Local training complete.")

    def get_parameters(self):
        """Return local model weights."""
        return self.model.state_dict()

    def evaluate(self):
        """Evaluate the model on the local test set."""
        self.model.eval()
        self.acc_metric.reset()
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                self.acc_metric.update(outputs, labels)
                
        accuracy = self.acc_metric.compute().item()
        return accuracy

# ==============================================================================
# 5. AGGREGATION FUNCTION (FedAvg)
# ==============================================================================

def fed_avg(client_weights_list, client_data_sizes):
    """
    Performs the Federated Averaging (FedAvg) aggregation.
    """
    # Calculate weights
    total_data_size = sum(client_data_sizes)
    weights = [size / total_data_size for size in client_data_sizes]
    
    # Initialize a new state_dict for the averaged model
    avg_state_dict = copy.deepcopy(client_weights_list[0])
    
    # Zero out all parameters
    for key in avg_state_dict.keys():
        avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])
        
    # Perform weighted average
    for key in avg_state_dict.keys():
        for i, state_dict in enumerate(client_weights_list):
            avg_state_dot = state_dict.get(key)
            if avg_state_dot is not None:
                avg_state_dict[key] += avg_state_dot * weights[i]
            
    return avg_state_dict

# ==============================================================================
# 6. EVALUATION & METRICS
# ==============================================================================

def evaluate_global(model, test_loader, device, metrics_dict):
    """
    Evaluates the global model on the hold-out global test set.
    """
    model.eval()
    
    # Reset all metrics
    for metric in metrics_dict.values():
        metric.reset()
        
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Update metrics
            metrics_dict['acc'].update(outputs, labels)
            metrics_dict['f1'].update(outputs, labels)
            
            # AUROC needs probabilities
            probs = F.softmax(outputs, dim=1)
            metrics_dict['auc'].update(probs, labels)

    # Compute final results
    results = {
        'acc': metrics_dict['acc'].compute().item(),
        'f1': metrics_dict['f1'].compute().item(),
        'auc': metrics_dict['auc'].compute().item(),
    }
    return results

def get_model_bytes(model, only_trainable=False):
    """
    Calculates the total size of a model's parameters in bytes.
    """
    total_bytes = 0
    for param in model.parameters():
        if only_trainable and not param.requires_grad:
            continue
        total_bytes += param.numel() * param.element_size()
    return total_bytes

# ==============================================================================
# 7. MAIN SIMULATION
# ==============================================================================

def main():
    print(f"--- Federated Learning Simulation: NWPU-RESISC45 ---")
    print(f"Simulating {NUM_CLIENTS} clients for {NUM_ROUNDS} rounds, with {LOCAL_EPOCHS} local epochs.")

    # --- Load Checkpoint if it exists ---
    start_round = 0
    global_acc_log = []
    global_f1_log = []
    global_auc_log = []
    cumulative_comm_log = []
    # NEW LOG: Stores accuracy for each client at the end of each round
    client_acc_per_round_log = [] 

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            start_round = checkpoint['round']
            global_acc_log = checkpoint['global_acc_log']
            global_f1_log = checkpoint['global_f1_log']
            global_auc_log = checkpoint['global_auc_log']
            cumulative_comm_log = checkpoint.get('cumulative_comm_log', [])
            client_acc_per_round_log = checkpoint.get('client_acc_per_round_log', [])
            print(f"Resuming from Round {start_round + 1}...")
        except Exception as e:
            print(f"Warning: Could not load checkpoint. Starting from scratch. Error: {e}")
            start_round = 0
    else:
        print("No checkpoint found. Starting from Round 1...")
    
    # 1. Load and partition data
    try:
        global_test_loader, client_loaders, client_data_sizes = get_dataloaders(
            TRAIN_DATA_PATH, TEST_DATA_PATH, NUM_CLIENTS
        )
    except Exception as e:
        print(f"Failed to load data: {e}")
        print("Please ensure your paths are correct and you have run the data setup.")
        return

    # 2. Initialize models and clients
    model_template = SimpleCNN(num_classes=NUM_CLASSES) # CPU template
    global_model = copy.deepcopy(model_template).to(DEVICE)
    
    # Load model state from checkpoint
    if start_round > 0 and 'model_state_dict' in checkpoint:
        try:
            global_model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded saved model weights from checkpoint.")
        except RuntimeError as e:
            print(f"ERROR: Could not load model weights: {e}")
            print("Please DELETE your 'fl_checkpoint.pth' file and restart.")
            return
    
    clients = []
    for i in range(NUM_CLIENTS):
        train_loader, test_loader = client_loaders[i]
        client = Client(
            client_id=i,
            model_template=model_template,
            train_loader=train_loader,
            test_loader=test_loader,
            device=DEVICE
        )
        clients.append(client)
        
    # 3. Initialize global metrics
    global_metrics_dict = {
        'acc': torchmetrics.classification.MulticlassAccuracy(
            num_classes=NUM_CLASSES
        ).to(DEVICE),
        'f1': torchmetrics.classification.MulticlassF1Score(
            num_classes=NUM_CLASSES, average='macro'
        ).to(DEVICE),
        'auc': torchmetrics.classification.MulticlassAUROC(
            num_classes=NUM_CLASSES, average='macro', thresholds=None
        ).to(DEVICE)
    }
    
    # 4. Initialize logs and calculate communication cost
    bytes_per_client_down = get_model_bytes(global_model, only_trainable=False)
    bytes_per_client_up = get_model_bytes(global_model, only_trainable=True)
    
    print("\n--- Model Size & Communication Metrics ---")
    print(f"Full model size (Server -> Client): {bytes_per_client_down / (1024**2):.2f} MB")
    print(f"Trainable params size (Client -> Server): {bytes_per_client_up / (1024**2):.2f} MB")
    
    # 5. Start Federated Training Loop
    print("\n--- Starting Federated Averaging (FedAvg) Simulation ---")
    
    total_comm_bytes = cumulative_comm_log[-1] if cumulative_comm_log else 0
    
    for round_num in range(start_round, start_round + NUM_ROUNDS):
        
        print(f"\n--- Round {round_num + 1}/{start_round + NUM_ROUNDS} (LR: {LEARNING_RATE}) ---")
        
        global_state_dict = global_model.state_dict()
        client_weights_list = []
        current_client_data_sizes = []
        
        # --- Client Phase ---
        for client in clients:
            # 1. Send model to client
            client.set_parameters(global_state_dict)
            total_comm_bytes += bytes_per_client_down
            
            # 2. Client trains
            client.train(LOCAL_EPOCHS)
            
            # 3. Get model from client
            client_weights_list.append(client.get_parameters())
            current_client_data_sizes.append(len(client.train_loader.dataset))
            total_comm_bytes += bytes_per_client_up
        
        cumulative_comm_log.append(total_comm_bytes)
        
        # --- Server Phase ---
        print(f"...Server aggregating {len(client_weights_list)} client models using FedAvg...")
        # 4. Aggregate models (FedAvg)
        global_state_dict = fed_avg(client_weights_list, current_client_data_sizes)
        global_model.load_state_dict(global_state_dict)
        
        # 5. Evaluate the *new* global model on client local test sets (NEW STEP)
        current_round_client_accs = []
        for i, client in enumerate(clients):
            # Client.evaluate uses the model the client currently holds (which is the newly aggregated global model)
            local_acc = client.evaluate()
            current_round_client_accs.append(local_acc)
        client_acc_per_round_log.append(current_round_client_accs)
        
        # 6. Evaluate new global model on global test set
        print(f"...Server evaluating new global model on {len(global_test_loader.dataset)} global test images...")
        metrics = evaluate_global(global_model, global_test_loader, DEVICE, global_metrics_dict)
        
        global_acc_log.append(metrics['acc'])
        global_f1_log.append(metrics['f1'])
        global_auc_log.append(metrics['auc'])
        
        print(f"Round {round_num + 1} Global Metrics (on Global Test Set): "
              f"Acc: {metrics['acc']:.4f}, "
              f"F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")

    # 6. Simulation Finished - Print Final Report
    print("\n=======================================================")
    print("--- 🏁 Federated Learning Simulation Finished 🏁 ---")
    print("=======================================================")
    
    # --- Per-client performance
    print("\n## Final Per-Client Performance (Local Data)")
    print("---")
    # Get the last round's client accuracies
    client_final_acc = client_acc_per_round_log[-1] if client_acc_per_round_log else [client.evaluate() for client in clients]
    
    for i, acc in enumerate(client_final_acc):
        print(f"  Client {i:2}: Final Local Test Accuracy: {acc:.4f} (using global model)")
        
    acc_variance = np.var(client_final_acc)
    print(f"\n  **Client Accuracy Variance:** {acc_variance:.4f} (Lower is better for fairness)")

    # --- Communication Report ---
    print("\n## Communication and Efficiency Report")
    print("---")
    mb_sent_per_round = (bytes_per_client_down * NUM_CLIENTS) / (1024**2)
    mb_recd_per_round = (bytes_per_client_up * NUM_CLIENTS) / (1024**2)
    print(f"  MB Downstream (Server -> Clients) per round: {mb_sent_per_round:.2f} MB")
    print(f"  MB Upstream (Clients -> Server) per round: {mb_recd_per_round:.2f} MB")
    total_gb = total_comm_bytes / (1024**3)
    print(f"  **Total Cumulative Communication:** {total_gb:.2f} GB")
    
    # --- Save Checkpoint ---
    print(f"\n--- Checkpoint Saved to: {CHECKPOINT_PATH} ---")
    final_round_completed = start_round + NUM_ROUNDS
    try:
        torch.save({
            'round': final_round_completed,
            'model_state_dict': global_model.state_dict(),
            'global_acc_log': global_acc_log,
            'global_f1_log': global_f1_log,
            'global_auc_log': global_auc_log,
            'cumulative_comm_log': cumulative_comm_log,
            'client_acc_per_round_log': client_acc_per_round_log, # NEW LOGGING
        }, CHECKPOINT_PATH)
        print("Checkpoint saved successfully. Resume is available.")
    except Exception as e:
        print(f"Error: Could not save checkpoint. {e}")
    
    
    # --- Generate Analysis Graphs ---
    print("\n## Generating Analysis Graphs")
    
    total_rounds_run = len(global_acc_log)
    if total_rounds_run == 0:
        print("No rounds were run, skipping plot generation.")
    else:
        rounds_axis = range(1, total_rounds_run + 1)
        
        # --- Plot 1: Combined Global Metrics (Aesthetics Enhanced) ---
        plt.figure(figsize=(10, 6))
        plt.plot(rounds_axis, global_acc_log, marker='o', linestyle='-', color='dodgerblue', label='Accuracy')
        plt.plot(rounds_axis, global_f1_log, marker='s', linestyle='--', color='darkorange', label='F1 Score (Macro)')
        plt.plot(rounds_axis, global_auc_log, marker='^', linestyle=':', color='forestgreen', label='AUC (Macro)')
        
        plt.title(f'Global Model Convergence on Global Test Set (Total Rounds: {total_rounds_run})', fontsize=14, fontweight='bold')
        plt.xlabel('Global Communication Round ($R$)', fontsize=12)
        plt.ylabel('Performance Metric Value (0.0 to 1.0)', fontsize=12)
        plt.legend(loc='lower right', frameon=True, shadow=True)
        plt.grid(True, linestyle='--')
        plt.xticks(list(rounds_axis)[::max(1, len(list(rounds_axis))//10)])
        plt.ylim(min(0.0, min(global_acc_log + global_f1_log + global_auc_log) - 0.05), 1.05)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Final Per-Client Accuracy Variance (Aesthetics Enhanced) ---
        client_ids = [f'C{i}' for i in range(NUM_CLIENTS)]
        plt.figure(figsize=(12, 6))
        bars = plt.bar(client_ids, client_final_acc, color='skyblue', edgecolor='darkblue', linewidth=1.5)
        
        plt.title(f'Final Model Accuracy Across Clients (Local Test Data)', fontsize=14, fontweight='bold')
        plt.xlabel('Simulated Client ID', fontsize=12)
        plt.ylabel('Final Local Test Accuracy (Fraction)', fontsize=12)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        mean_acc = np.mean(client_final_acc)
        plt.axhline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean Accuracy: {mean_acc:.3f}')
        plt.legend()
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', color='black', fontsize=10)
            
        plt.tight_layout()
        plt.show()

        # ----------------------------------------------------------------------
        # --- Plot 4 (NEW): Client Test Accuracy Over Rounds ---------------------
        # ----------------------------------------------------------------------
        
        # Only plot if we have more than one round to show a trajectory
        if total_rounds_run > 1 and client_acc_per_round_log:
            plt.figure(figsize=(12, 7))
            # Transpose the log so we plot one line per client
            client_accs_transposed = np.array(client_acc_per_round_log).T 
            
            for i, acc_history in enumerate(client_accs_transposed):
                plt.plot(rounds_axis, acc_history, marker='.', linestyle='-', alpha=0.7, label=f'Client {i}')
                
            plt.title(f'Client Model Performance Trajectories on Local Test Data ({total_rounds_run} Rounds)', fontsize=14, fontweight='bold')
            plt.xlabel('Global Communication Round ($R$)', fontsize=12)
            plt.ylabel('Local Test Accuracy (After Global Aggregation)', fontsize=12)
            # Plot the global accuracy for context
            plt.plot(rounds_axis, global_acc_log, color='black', linewidth=3, linestyle='--', label='Global Test Acc (Reference)')
            plt.legend(loc='lower right', frameon=True, shadow=True, ncol=2)
            plt.grid(True, linestyle='--')
            plt.xticks(list(rounds_axis)[::max(1, len(list(rounds_axis))//10)])
            plt.ylim(0.0, 1.05)
            plt.tight_layout()
            plt.show()


        # ----------------------------------------------------------------------
        # --- Plot 5 (NEW): Round-by-Round Communication Cost --------------------
        # ----------------------------------------------------------------------

        cumulative_comm_gb = [b / (1024**3) for b in cumulative_comm_log]
        
        # Calculate per-round communication from cumulative log
        per_round_comm_gb = [cumulative_comm_gb[0]] + [cumulative_comm_gb[i] - cumulative_comm_gb[i-1] for i in range(1, len(cumulative_comm_gb))]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(rounds_axis, per_round_comm_gb, color='teal', edgecolor='darkslategray', linewidth=1.5)
        
        plt.title(f'Communication Cost Per Round ({NUM_CLIENTS} Clients, {total_rounds_run} Rounds)', fontsize=14, fontweight='bold')
        plt.xlabel('Global Communication Round ($R$)', fontsize=12)
        plt.ylabel('Data Transfer Per Round (Gigabytes - GB)', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rounds_axis)
        
        # Add labels on top of bars for non-zero communication
        for bar in bars:
            yval = bar.get_height()
            if yval > 0.01:
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', color='black', fontsize=9)
            
        plt.tight_layout()
        plt.show()


        # ----------------------------------------------------------------------
        # --- Plot 3 (OLD): Cumulative Communication Cost ------------------------
        # ----------------------------------------------------------------------
        plt.figure(figsize=(10, 6))
        plt.plot(rounds_axis, cumulative_comm_gb, marker='o', color='forestgreen', linewidth=3, label='Total Communication')
        
        plt.title(f'Cumulative Communication Cost of Federated Training (Total Rounds: {total_rounds_run})', fontsize=14, fontweight='bold')
        plt.xlabel('Global Communication Round ($R$)', fontsize=12)
        plt.ylabel('Cumulative Communication (Gigabytes - GB)', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--')
        plt.xticks(list(rounds_axis)[::max(1, len(list(rounds_axis))//10)])
        plt.tight_layout()
        plt.show()
    
# --- Execution Block ---
if __name__ == "__main__" or "ipykernel" in " ".join(sys.argv):
    main()