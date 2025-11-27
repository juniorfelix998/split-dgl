"""
Split Learning with Decoupled Greedy Learning (DGL)
====================================================
Backpropagation-Free Split Learning using Auxiliary Networks on Client Side

This implementation combines:
1. Split Learning architecture (Client-Server model split)
2. Decoupled Greedy Learning (DGL) for BP-free training
3. Auxiliary networks on client side for local gradient computation

Key Innovation: Clients don't need gradients from server - they use local
auxiliary networks to compute gradients, enabling true decoupled training.

Paper Reference: "Decoupled Greedy Learning of CNNs" (Belilovsky et al., 2019)
ArXiv: https://arxiv.org/abs/1901.08164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import copy
import time
import random
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# SECTION 1: SETUP AND CONFIGURATION
# ============================================================================

# Set random seeds for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Print formatting
def prRed(text): 
    print("\033[91m {}\033[00m".format(text))
    
def prGreen(text): 
    print("\033[92m {}\033[00m".format(text))

def prYellow(text):
    print("\033[93m {}\033[00m".format(text))

# ============================================================================
# SECTION 2: HYPERPARAMETERS
# ============================================================================

class Config:
    """Configuration class for all hyperparameters"""
    # Training configuration
    num_clients = 10          # Number of clients in the system
    epochs = 100              # Global training rounds
    client_frac = 1.0         # Fraction of clients participating per round
    local_epochs = 5          # Local epochs per client
    batch_size = 128          # Training batch size
    test_batch_size = 100     # Test batch size
    
    # Optimizer configuration
    lr_client = 0.01          # Learning rate for client-side model
    lr_aux = 0.01             # Learning rate for auxiliary network
    lr_server = 0.01          # Learning rate for server-side model
    momentum = 0.9
    weight_decay = 5e-4
    
    # Model architecture
    num_classes = 10          # CIFAR-10 classes
    
    # DGL specific
    aux_type = 'mlp-sr'       # Type of auxiliary network: 'mlp', 'mlp-sr', 'cnn'
    
    # Logging
    print_every = 10          # Print frequency
    
config = Config()

print("="*70)
print("SPLIT LEARNING WITH DECOUPLED GREEDY LEARNING (DGL)")
print("="*70)
print(f"Configuration:")
print(f"  - Clients: {config.num_clients}")
print(f"  - Epochs: {config.epochs}")
print(f"  - Batch Size: {config.batch_size}")
print(f"  - Auxiliary Network Type: {config.aux_type}")
print("="*70)

# ============================================================================
# SECTION 3: AUXILIARY NETWORK ARCHITECTURES
# ============================================================================

class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary Network for DGL
    
    This network is attached to the client-side model and predicts the final
    class labels directly from intermediate representations. It enables
    backpropagation-free training by providing local gradient signals.
    
    Architecture options:
    - 'mlp': Direct pooling to 2x2, then MLP
    - 'mlp-sr': Staged resolution reduction (4x) + 1x1 convs, then pooling + MLP
    - 'cnn': CNN layers + pooling + classifier
    
    Key Design Principle: Auxiliary network should be <5% of main network FLOPs
    """
    
    def __init__(self, input_channels, spatial_size, num_classes=10, 
                 aux_type='mlp-sr'):
        super(AuxiliaryClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.aux_type = aux_type
        
        if aux_type == 'mlp':
            # Direct averaging to 2x2, then MLP (most efficient)
            # Cost: ~0.7% of main network
            mlp_input = input_channels * 4  # 2x2 spatial
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, num_classes)
            )
            
        elif aux_type == 'mlp-sr':
            # Staged resolution: spatial reduction + 1x1 convs + pooling + MLP
            # Cost: ~4% of main network
            # First reduce spatial size by 4x using 1x1 convolutions
            self.spatial_reduce = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Conv2d(input_channels, input_channels, 
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Conv2d(input_channels, input_channels, 
                         kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
            )
            
            # Then pool to 2x2
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            
            # Finally MLP classifier
            mlp_input = input_channels * 4
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, num_classes)
            )
            
        elif aux_type == 'cnn':
            # CNN-based auxiliary (baseline from paper)
            # Cost: ~200% of main network (not recommended for distributed)
            self.cnn_layers = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Conv2d(input_channels, input_channels, 
                         kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            mlp_input = input_channels * 4
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass through auxiliary network
        
        Args:
            x: Intermediate representation from main network
            
        Returns:
            Class predictions
        """
        if self.aux_type == 'mlp':
            # Direct pooling
            out = self.avgpool(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            
        elif self.aux_type == 'mlp-sr':
            # Staged resolution reduction
            out = self.spatial_reduce(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            
        elif self.aux_type == 'cnn':
            # CNN processing
            out = self.cnn_layers(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            
        return out

# ============================================================================
# SECTION 4: CLIENT-SIDE MODEL (With Auxiliary Network)
# ============================================================================

class ResidualBlock(nn.Module):
    """Standard ResNet residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ClientSideModel(nn.Module):
    """
    Client-Side Model for Split Learning with DGL
    
    This model consists of:
    1. Main CNN layers (ResNet blocks)
    2. Auxiliary network for local gradient computation
    
    Key Feature: The auxiliary network enables BP-free training by providing
    local gradient signals without needing feedback from the server.
    """
    
    def __init__(self, num_classes=10, aux_type='mlp-sr'):
        super(ClientSideModel, self).__init__()
        
        # Main CNN pathway (first half of ResNet)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # ResNet layers - we use first 2 layers for client
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        
        # Auxiliary network for DGL (BP-free training)
        # This predicts final class labels from intermediate representation
        # Spatial size after layer2: 16x16 (CIFAR-10: 32 -> 16 after stride=2)
        self.auxiliary_net = AuxiliaryClassifier(
            input_channels=128,
            spatial_size=16,
            num_classes=num_classes,
            aux_type=aux_type
        )
        
        self.num_classes = num_classes
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper to create ResNet layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x, return_aux=False):
        """
        Forward pass through client model
        
        Args:
            x: Input images
            return_aux: If True, also return auxiliary predictions
            
        Returns:
            If return_aux=False: intermediate representation (smashed data)
            If return_aux=True: (smashed data, auxiliary predictions)
        """
        # Main pathway
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)  # Output: [batch, 128, 16, 16]
        
        if return_aux:
            # Also compute auxiliary predictions for DGL training
            aux_out = self.auxiliary_net(out)
            return out, aux_out
        else:
            return out
    
    def compute_auxiliary_loss(self, x, labels, criterion):
        """
        Compute loss using auxiliary network (for DGL training)
        
        This is the key method for BP-free training. The client computes
        gradients using only the auxiliary network, without any feedback
        from the server.
        
        Args:
            x: Input images
            labels: Ground truth labels
            criterion: Loss function
            
        Returns:
            loss, accuracy
        """
        # Forward through main network and auxiliary network
        _, aux_predictions = self.forward(x, return_aux=True)
        
        # Compute loss on auxiliary predictions
        loss = criterion(aux_predictions, labels)
        
        # Compute accuracy
        _, predicted = aux_predictions.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        
        return loss, accuracy

# ============================================================================
# SECTION 5: SERVER-SIDE MODEL
# ============================================================================

class ServerSideModel(nn.Module):
    """
    Server-Side Model for Split Learning
    
    This model contains the remaining layers of the network.
    In DGL, the server trains normally but doesn't send gradients back to
    clients (update unlocking).
    """
    
    def __init__(self, num_classes=10):
        super(ServerSideModel, self).__init__()
        
        # Continue from where client left off (128 channels, 16x16 spatial)
        # Remaining ResNet layers
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """Helper to create ResNet layer"""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through server model
        
        Args:
            x: Smashed data (intermediate representation from client)
            
        Returns:
            Final class predictions
        """
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# ============================================================================
# SECTION 6: DATA LOADING AND PREPROCESSING
# ============================================================================

class DatasetSplit(Dataset):
    """Custom Dataset for client data splits"""
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def create_iid_data_split(dataset, num_clients):
    """
    Create IID data split for clients
    
    Args:
        dataset: The full dataset
        num_clients: Number of clients
        
    Returns:
        dict: Dictionary mapping client_id -> data indices
    """
    num_items = len(dataset) // num_clients
    dict_users = {}
    all_idxs = list(range(len(dataset)))
    
    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, 
                                            replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users


def load_cifar10_data(num_clients):
    """
    Load and prepare CIFAR-10 dataset
    
    Returns:
        train_dataset, test_dataset, train_splits, test_splits
    """
    # Data normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms with data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform_test
    )
    
    # Create data splits for clients
    train_splits = create_iid_data_split(train_dataset, num_clients)
    test_splits = create_iid_data_split(test_dataset, num_clients)
    
    print(f"\nDataset loaded:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Samples per client: ~{len(train_dataset)//num_clients}")
    
    return train_dataset, test_dataset, train_splits, test_splits

# ============================================================================
# SECTION 7: SERVER TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy"""
    _, predicted = predictions.max(1)
    correct = predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy


# Global variables for tracking metrics
loss_train_collect = []
acc_train_collect = []
loss_test_collect = []
acc_test_collect = []

batch_acc_train = []
batch_loss_train = []
batch_acc_test = []
batch_loss_test = []

acc_train_collect_user = []
loss_train_collect_user = []
acc_test_collect_user = []
loss_test_collect_user = []

count_train = 0
count_test = 0

idx_collect = []
l_epoch_check = False
fed_check = False


def train_server(server_model, smashed_data, labels, criterion, optimizer,
                idx, local_epoch, local_epoch_count, len_batch):
    """
    Server-side training function
    
    This function:
    1. Receives smashed data (intermediate representations) from client
    2. Performs forward pass through server model
    3. Computes loss and backpropagates
    4. DOES NOT send gradients back to client (DGL: update unlocking)
    
    Args:
        server_model: Server-side neural network
        smashed_data: Intermediate representation from client
        labels: Ground truth labels
        criterion: Loss function
        optimizer: Server optimizer
        idx: Client index
        local_epoch: Total local epochs
        local_epoch_count: Current local epoch
        len_batch: Number of batches
        
    Returns:
        None (in DGL, we don't send gradients back)
    """
    global batch_acc_train, batch_loss_train, count_train
    global loss_train_collect_user, acc_train_collect_user
    global idx_collect, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect
    
    server_model.train()
    optimizer.zero_grad()
    
    # Move data to device
    smashed_data = smashed_data.to(device)
    labels = labels.to(device)
    
    # Forward pass through server
    predictions = server_model(smashed_data)
    
    # Compute loss and accuracy
    loss = criterion(predictions, labels)
    accuracy = calculate_accuracy(predictions, labels)
    
    # Backward pass (only updates server)
    loss.backward()
    optimizer.step()
    
    # Track metrics
    batch_loss_train.append(loss.item())
    batch_acc_train.append(accuracy)
    
    # Update counters and logging
    count_train += 1
    if count_train == len_batch:
        # Average over batch
        acc_avg = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg = sum(batch_loss_train) / len(batch_loss_train)
        
        batch_acc_train = []
        batch_loss_train = []
        count_train = 0
        
        prRed(f'Client{idx} Server Train => Epoch {local_epoch_count+1}/{local_epoch} | '
              f'Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}')
        
        # Check if local epoch completed
        if local_epoch_count == local_epoch - 1:
            l_epoch_check = True
            
            # Store metrics for this client
            acc_train_collect_user.append(acc_avg)
            loss_train_collect_user.append(loss_avg)
            
            # Track which clients have been served
            if idx not in idx_collect:
                idx_collect.append(idx)
        
        # Check if all clients completed (one federated round)
        if len(idx_collect) == config.num_clients * config.client_frac:
            fed_check = True
            idx_collect = []
            
            # Average across all clients
            acc_avg_all = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all = sum(loss_train_collect_user) / len(loss_train_collect_user)
            
            acc_train_collect.append(acc_avg_all)
            loss_train_collect.append(loss_avg_all)
            
            acc_train_collect_user = []
            loss_train_collect_user = []


def evaluate_server(server_model, smashed_data, labels, criterion,
                   idx, len_batch, global_round):
    """
    Server-side evaluation function
    
    Args:
        server_model: Server-side neural network
        smashed_data: Intermediate representation from client
        labels: Ground truth labels
        criterion: Loss function
        idx: Client index
        len_batch: Number of batches
        global_round: Current global round
    """
    global batch_acc_test, batch_loss_test, count_test
    global loss_test_collect_user, acc_test_collect_user
    global l_epoch_check, fed_check
    global loss_test_collect, acc_test_collect
    
    server_model.eval()
    
    with torch.no_grad():
        # Move data to device
        smashed_data = smashed_data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        predictions = server_model(smashed_data)
        
        # Compute loss and accuracy
        loss = criterion(predictions, labels)
        accuracy = calculate_accuracy(predictions, labels)
        
        # Track metrics
        batch_loss_test.append(loss.item())
        batch_acc_test.append(accuracy)
        
        count_test += 1
        if count_test == len_batch:
            # Average over batch
            acc_avg = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg = sum(batch_loss_test) / len(batch_loss_test)
            
            batch_acc_test = []
            batch_loss_test = []
            count_test = 0
            
            prGreen(f'Client{idx} Server Test  =>                  | '
                   f'Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}')
            
            # Check if local epoch completed
            if l_epoch_check:
                l_epoch_check = False
                
                # Store metrics
                acc_test_collect_user.append(acc_avg)
                loss_test_collect_user.append(loss_avg)
            
            # Check if all clients completed
            if fed_check:
                fed_check = False
                
                # Average across all clients
                acc_avg_all = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all = sum(loss_test_collect_user) / len(loss_test_collect_user)
                
                acc_test_collect.append(acc_avg_all)
                loss_test_collect.append(loss_avg_all)
                
                acc_test_collect_user = []
                loss_test_collect_user = []
                
                # Print summary
                print("\n" + "="*70)
                print(f"ROUND {global_round} SUMMARY (DGL Split Learning)")
                print("="*70)
                print(f"Train | Avg Accuracy: {acc_train_collect[-1]:.3f}% | "
                      f"Avg Loss: {loss_train_collect[-1]:.4f}")
                print(f"Test  | Avg Accuracy: {acc_avg_all:.3f}% | "
                      f"Avg Loss: {loss_avg_all:.4f}")
                print("="*70 + "\n")

# ============================================================================
# SECTION 8: CLIENT CLASS (DGL Training Logic)
# ============================================================================

class Client:
    """
    Client class for Split Learning with DGL
    
    Key Innovation: Uses auxiliary network for local gradient computation,
    enabling backpropagation-free training without server feedback.
    
    Training Flow (DGL):
    1. Forward pass through client model -> smashed data
    2. Forward pass through auxiliary network -> aux predictions
    3. Compute loss using auxiliary network
    4. Backpropagate through client model (no server gradients needed!)
    5. Send smashed data to server (for server training)
    
    This is different from standard Split Learning where:
    - Client waits for gradients from server
    - Client backpropagates using server gradients
    """
    
    def __init__(self, client_id, client_model, train_dataset, test_dataset,
                 train_idxs, test_idxs, config):
        self.client_id = client_id
        self.client_model = client_model
        self.config = config
        self.device = device
        
        # Create data loaders
        self.train_loader = DataLoader(
            DatasetSplit(train_dataset, train_idxs),
            batch_size=config.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            DatasetSplit(test_dataset, test_idxs),
            batch_size=config.test_batch_size,
            shuffle=False
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def train_dgl(self, server_model, server_optimizer, global_round):
        """
        DGL Training: Backpropagation-free training using auxiliary network
        
        This is the core innovation. The client:
        1. Trains its model using the auxiliary network (no server feedback)
        2. Sends smashed data to server for server training
        3. Does NOT receive or use gradients from server (update unlocking)
        
        Args:
            server_model: Server-side model
            server_optimizer: Server optimizer
            global_round: Current global round
            
        Returns:
            Updated client model state dict
        """
        self.client_model.train()
        
        # Optimizer for client model (includes main CNN + auxiliary network)
        optimizer = torch.optim.SGD(
            self.client_model.parameters(),
            lr=self.config.lr_client,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Local training for multiple epochs
        for local_epoch in range(self.config.local_epochs):
            len_batch = len(self.train_loader)
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # =============================================================
                # CLIENT-SIDE DGL TRAINING (BP-FREE)
                # =============================================================
                optimizer.zero_grad()
                
                # Forward pass through client model
                # Get both smashed data and auxiliary predictions
                smashed_data, aux_predictions = self.client_model(
                    images, return_aux=True
                )
                
                # Compute loss using AUXILIARY network (key for DGL)
                # This provides local gradient signal without server feedback
                aux_loss = self.criterion(aux_predictions, labels)
                
                # Backpropagate using auxiliary loss
                # This updates both main CNN and auxiliary network
                aux_loss.backward()
                optimizer.step()
                
                # =============================================================
                # SEND SMASHED DATA TO SERVER (for server training)
                # =============================================================
                # Detach smashed data (no gradients flow back to client)
                smashed_data_detached = smashed_data.clone().detach()
                
                # Server trains on smashed data
                train_server(
                    server_model,
                    smashed_data_detached,
                    labels,
                    self.criterion,
                    server_optimizer,
                    self.client_id,
                    self.config.local_epochs,
                    local_epoch,
                    len_batch
                )
                
                # Note: In DGL, we do NOT receive gradients from server
                # This is "update unlocking" - client updates independently
        
        return self.client_model.state_dict()
    
    def evaluate(self, server_model, global_round):
        """
        Evaluate client model with server model
        
        Args:
            server_model: Server-side model
            global_round: Current global round
        """
        self.client_model.eval()
        
        with torch.no_grad():
            len_batch = len(self.test_loader)
            
            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward through client
                smashed_data = self.client_model(images, return_aux=False)
                
                # Evaluate on server
                evaluate_server(
                    server_model,
                    smashed_data,
                    labels,
                    self.criterion,
                    self.client_id,
                    len_batch,
                    global_round
                )

# ============================================================================
# SECTION 9: MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function for DGL Split Learning"""
    
    print("\n" + "="*70)
    print("INITIALIZING DGL SPLIT LEARNING")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Step 1: Load and prepare data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, train_splits, test_splits = \
        load_cifar10_data(config.num_clients)
    
    # -------------------------------------------------------------------------
    # Step 2: Initialize models
    # -------------------------------------------------------------------------
    print("\n[2/5] Initializing models...")
    
    # Global client model (shared architecture, weights copied to clients)
    global_client_model = ClientSideModel(
        num_classes=config.num_classes,
        aux_type=config.aux_type
    ).to(device)
    
    # Global server model
    global_server_model = ServerSideModel(
        num_classes=config.num_classes
    ).to(device)
    
    print(f"\nClient Model:")
    print(f"  - Parameters: {sum(p.numel() for p in global_client_model.parameters()):,}")
    print(f"  - Auxiliary Network: {config.aux_type}")
    print(f"\nServer Model:")
    print(f"  - Parameters: {sum(p.numel() for p in global_server_model.parameters()):,}")
    
    # Server optimizer
    server_optimizer = torch.optim.SGD(
        global_server_model.parameters(),
        lr=config.lr_server,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    # -------------------------------------------------------------------------
    # Step 3: Create clients
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Creating {config.num_clients} clients...")
    clients = []
    for client_id in range(config.num_clients):
        client = Client(
            client_id=client_id,
            client_model=copy.deepcopy(global_client_model).to(device),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_idxs=train_splits[client_id],
            test_idxs=test_splits[client_id],
            config=config
        )
        clients.append(client)
    print(f"  - Created {len(clients)} clients successfully")
    
    # -------------------------------------------------------------------------
    # Step 4: Training loop
    # -------------------------------------------------------------------------
    print("\n[4/5] Starting DGL Split Learning training...")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    for global_round in range(1, config.epochs + 1):
        # Select participating clients
        num_selected = max(int(config.client_frac * config.num_clients), 1)
        selected_clients = np.random.choice(
            range(config.num_clients), num_selected, replace=False
        )
        
        # Sequential training across selected clients
        for client_idx in selected_clients:
            client = clients[client_idx]
            
            # =====================================================================
            # DGL TRAINING: Client trains using auxiliary network (BP-free)
            # =====================================================================
            updated_client_weights = client.train_dgl(
                server_model=global_server_model,
                server_optimizer=server_optimizer,
                global_round=global_round
            )
            
            # Update global client model (simple weight copying)
            # In more sophisticated FL, you'd use FedAvg or similar
            global_client_model.load_state_dict(updated_client_weights)
            
            # Update this client's model
            client.client_model.load_state_dict(updated_client_weights)
            
            # =====================================================================
            # EVALUATION
            # =====================================================================
            client.evaluate(
                server_model=global_server_model,
                global_round=global_round
            )
    
    elapsed_time = time.time() - start_time
    
    # -------------------------------------------------------------------------
    # Step 5: Save results and create plots
    # -------------------------------------------------------------------------
    print("\n[5/5] Saving results and creating plots...")
    
    # Save training metrics
    results = {
        'train_loss': loss_train_collect,
        'train_acc': acc_train_collect,
        'test_loss': loss_test_collect,
        'test_acc': acc_test_collect,
        'config': vars(config),
        'training_time': elapsed_time
    }
    
    torch.save(results, 'dgl_split_learning_results.pt')
    
    # Create plots
    create_plots(loss_train_collect, acc_train_collect,
                loss_test_collect, acc_test_collect)
    
    # Print final results
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Total Training Time: {elapsed_time/60:.2f} minutes")
    print(f"Final Train Accuracy: {acc_train_collect[-1]:.2f}%")
    print(f"Final Test Accuracy: {acc_test_collect[-1]:.2f}%")
    print(f"Results saved to: dgl_split_learning_results.pt")
    print("="*70 + "\n")
    
    return results

# ============================================================================
# SECTION 10: VISUALIZATION AND ANALYSIS
# ============================================================================

def create_plots(train_loss, train_acc, test_loss, test_acc):
    """Create training visualization plots"""
    
    rounds = list(range(1, len(train_loss) + 1))
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(rounds, train_loss, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(rounds, test_loss, 'r-s', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('DGL Split Learning - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(rounds, train_acc, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(rounds, test_acc, 'r-s', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('DGL Split Learning - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dgl_split_learning_plots.png', dpi=300, bbox_inches='tight')
    print(f"  - Plots saved to: dgl_split_learning_plots.png")


# ============================================================================
# SECTION 11: RUN TRAINING
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("DECOUPLED GREEDY LEARNING (DGL) + SPLIT LEARNING")
    print("Backpropagation-Free Distributed Training")
    print("="*70)
    
    results = main()
