"""
IMPROVED DGL Split Learning Implementation
===========================================
This version includes all Priority 1 improvements:

✅ Solution 1A: Dropout in auxiliary network
✅ Solution 1B: Label smoothing
✅ Solution 2A: Separate learning rates for auxiliary network
✅ Solution 1C: CNN auxiliary network option (easy switch)
✅ Solution 3: Stronger data augmentation

Expected Improvement: +5-8% test accuracy over original
Target: 88-91% test accuracy on CIFAR-10

Author: Felix Owino
Date: November 2024
Version: 2.0 (Improved)
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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# SETUP AND CONFIGURATION
# ============================================================================

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prRed(text):
    print("\033[91m {}\033[00m".format(text))


def prGreen(text):
    print("\033[92m {}\033[00m".format(text))


def prYellow(text):
    print("\033[93m {}\033[00m".format(text))


# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

class ImprovedConfig:
    """
    Improved configuration with all Priority 1 enhancements
    """
    # Training configuration
    num_clients = 10
    epochs = 100
    client_frac = 1.0
    local_epochs = 8  # ✅ IMPROVED: Increased from 5 to 8
    batch_size = 128
    test_batch_size = 100

    # Optimizer configuration - SEPARATE LEARNING RATES
    lr_client = 0.01  # Main CNN learning rate
    lr_aux = 0.02  # ✅ IMPROVED: 2x higher for auxiliary network
    lr_server = 0.01
    momentum = 0.9
    weight_decay = 1e-4  # ✅ IMPROVED: Reduced from 5e-4 to 1e-4

    # Loss configuration
    label_smoothing = 0.1  # ✅ IMPROVED: Added label smoothing

    # Model architecture
    num_classes = 10
    aux_type = 'mlp-sr'  # Options: 'mlp', 'mlp-sr', 'cnn'
    # ✅ Change to 'cnn' for Solution 1C

    # Dropout rates for auxiliary network
    dropout_2d = 0.2  # ✅ IMPROVED: For convolutional layers
    dropout_1d = 0.5  # ✅ IMPROVED: For fully connected layers

    # Data augmentation strength
    use_strong_augmentation = True  # ✅ IMPROVED: Enable strong augmentation

    # Logging
    print_every = 10


config = ImprovedConfig()

print("=" * 70)
print("IMPROVED DGL SPLIT LEARNING")
print("=" * 70)
print(f"Improvements Applied:")
print(f"  ✅ Dropout in auxiliary network ({config.dropout_2d}, {config.dropout_1d})")
print(f"  ✅ Label smoothing ({config.label_smoothing})")
print(f"  ✅ Separate auxiliary LR ({config.lr_aux} vs {config.lr_client})")
print(f"  ✅ Reduced weight decay ({config.weight_decay})")
print(f"  ✅ Increased local epochs ({config.local_epochs})")
print(f"  ✅ Strong data augmentation (enabled)")
print(f"  ✅ Auxiliary network type: {config.aux_type}")
print("=" * 70)


# ============================================================================
# IMPROVED AUXILIARY NETWORK WITH DROPOUT
# ============================================================================

class ImprovedAuxiliaryClassifier(nn.Module):
    """
    IMPROVED Auxiliary Network with Dropout

    Key improvements:
    - Dropout2d after convolutional layers
    - Dropout after fully connected layers
    - Prevents overfitting
    - Better generalization
    """

    def __init__(self, input_channels, spatial_size, num_classes=10,
                 aux_type='mlp-sr', dropout_2d=0.2, dropout_1d=0.5):
        super(ImprovedAuxiliaryClassifier, self).__init__()

        self.input_channels = input_channels
        self.spatial_size = spatial_size
        self.num_classes = num_classes
        self.aux_type = aux_type

        if aux_type == 'mlp':
            # ========================================================
            # MLP: Direct pooling + MLP
            # ========================================================
            mlp_input = input_channels * 4
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(dropout_1d),  # ✅ ADDED DROPOUT

                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(dropout_1d),  # ✅ ADDED DROPOUT

                nn.Linear(256, num_classes)
            )

        elif aux_type == 'mlp-sr':
            # ========================================================
            # MLP-SR: Staged resolution with 1x1 convs + MLP
            # ========================================================
            self.spatial_reduce = nn.Sequential(
                # Layer 1
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_2d),  # ✅ ADDED DROPOUT

                # Layer 2
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_2d),  # ✅ ADDED DROPOUT

                # Layer 3
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_2d),  # ✅ ADDED DROPOUT
            )

            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

            mlp_input = input_channels * 4
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(dropout_1d),  # ✅ ADDED DROPOUT

                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(dropout_1d),  # ✅ ADDED DROPOUT

                nn.Linear(256, num_classes)
            )

        elif aux_type == 'cnn':
            # ========================================================
            # CNN: Convolutional layers + MLP
            # ========================================================
            self.cnn_layers = nn.Sequential(
                # Layer 1
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_2d),  # ✅ ADDED DROPOUT

                # Layer 2
                nn.Conv2d(input_channels, input_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(input_channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_2d),  # ✅ ADDED DROPOUT
            )

            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

            mlp_input = input_channels * 4
            self.classifier = nn.Sequential(
                nn.Linear(mlp_input, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(dropout_1d),  # ✅ ADDED DROPOUT

                nn.Linear(256, num_classes)
            )

    def forward(self, x):
        if self.aux_type == 'mlp':
            out = self.avgpool(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

        elif self.aux_type == 'mlp-sr':
            out = self.spatial_reduce(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

        elif self.aux_type == 'cnn':
            out = self.cnn_layers(x)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

        return out


# ============================================================================
# CLIENT-SIDE MODEL (Same structure, improved auxiliary)
# ============================================================================

class ResidualBlock(nn.Module):
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


class ImprovedClientSideModel(nn.Module):
    """
    Client-Side Model with IMPROVED Auxiliary Network
    """

    def __init__(self, num_classes=10, aux_type='mlp-sr',
                 dropout_2d=0.2, dropout_1d=0.5):
        super(ImprovedClientSideModel, self).__init__()

        # Main CNN pathway (unchanged)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

        # IMPROVED Auxiliary network with dropout
        self.auxiliary_net = ImprovedAuxiliaryClassifier(
            input_channels=128,
            spatial_size=16,
            num_classes=num_classes,
            aux_type=aux_type,
            dropout_2d=dropout_2d,  # ✅ IMPROVED
            dropout_1d=dropout_1d  # ✅ IMPROVED
        )

        self.num_classes = num_classes

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_aux=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)

        if return_aux:
            aux_out = self.auxiliary_net(out)
            return out, aux_out
        else:
            return out

    def compute_auxiliary_loss(self, x, labels, criterion):
        _, aux_predictions = self.forward(x, return_aux=True)
        loss = criterion(aux_predictions, labels)
        _, predicted = aux_predictions.max(1)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        return loss, accuracy


# ============================================================================
# SERVER-SIDE MODEL (Unchanged)
# ============================================================================

class ServerSideModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ServerSideModel, self).__init__()

        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ============================================================================
# IMPROVED DATA LOADING WITH STRONGER AUGMENTATION
# ============================================================================

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def create_iid_data_split(dataset, num_clients):
    num_items = len(dataset) // num_clients
    dict_users = {}
    all_idxs = list(range(len(dataset)))

    for i in range(num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def load_cifar10_data_improved(num_clients, use_strong_augmentation=True):
    """
    Load CIFAR-10 with IMPROVED data augmentation
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if use_strong_augmentation:
        # ✅ IMPROVED: Stronger augmentation
        print("Using STRONG data augmentation")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),  # ✅ ADDED
            transforms.RandomRotation(15),  # ✅ ADDED
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),  # ✅ ADDED
        ])
    else:
        # Standard augmentation
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(
        './data', train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        './data', train=False, download=True, transform=transform_test
    )

    train_splits = create_iid_data_split(train_dataset, num_clients)
    test_splits = create_iid_data_split(test_dataset, num_clients)

    print(f"\nDataset loaded with {'STRONG' if use_strong_augmentation else 'standard'} augmentation:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Samples per client: ~{len(train_dataset) // num_clients}")

    return train_dataset, test_dataset, train_splits, test_splits


# ============================================================================
# SERVER FUNCTIONS (Label smoothing applied)
# ============================================================================

def calculate_accuracy(predictions, labels):
    _, predicted = predictions.max(1)
    correct = predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy


# Global metrics tracking
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
    global batch_acc_train, batch_loss_train, count_train
    global loss_train_collect_user, acc_train_collect_user
    global idx_collect, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect

    server_model.train()
    optimizer.zero_grad()

    smashed_data = smashed_data.to(device)
    labels = labels.to(device)

    predictions = server_model(smashed_data)
    loss = criterion(predictions, labels)
    accuracy = calculate_accuracy(predictions, labels)

    loss.backward()
    optimizer.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(accuracy)

    count_train += 1
    if count_train == len_batch:
        acc_avg = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count_train = 0

        prRed(f'Client{idx} Server Train => Epoch {local_epoch_count + 1}/{local_epoch} | '
              f'Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}')

        if local_epoch_count == local_epoch - 1:
            l_epoch_check = True
            acc_train_collect_user.append(acc_avg)
            loss_train_collect_user.append(loss_avg)

            if idx not in idx_collect:
                idx_collect.append(idx)

        if len(idx_collect) == config.num_clients * config.client_frac:
            fed_check = True
            idx_collect = []

            acc_avg_all = sum(acc_train_collect_user) / len(acc_train_collect_user)
            loss_avg_all = sum(loss_train_collect_user) / len(loss_train_collect_user)

            acc_train_collect.append(acc_avg_all)
            loss_train_collect.append(loss_avg_all)

            acc_train_collect_user = []
            loss_train_collect_user = []


def evaluate_server(server_model, smashed_data, labels, criterion,
                    idx, len_batch, global_round):
    global batch_acc_test, batch_loss_test, count_test
    global loss_test_collect_user, acc_test_collect_user
    global l_epoch_check, fed_check
    global loss_test_collect, acc_test_collect

    server_model.eval()

    with torch.no_grad():
        smashed_data = smashed_data.to(device)
        labels = labels.to(device)

        predictions = server_model(smashed_data)
        loss = criterion(predictions, labels)
        accuracy = calculate_accuracy(predictions, labels)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(accuracy)

        count_test += 1
        if count_test == len_batch:
            acc_avg = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count_test = 0

            prGreen(f'Client{idx} Server Test  =>                  | '
                    f'Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}')

            if l_epoch_check:
                l_epoch_check = False
                acc_test_collect_user.append(acc_avg)
                loss_test_collect_user.append(loss_avg)

            if fed_check:
                fed_check = False
                acc_avg_all = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all = sum(loss_test_collect_user) / len(loss_test_collect_user)

                acc_test_collect.append(acc_avg_all)
                loss_test_collect.append(loss_avg_all)

                acc_test_collect_user = []
                loss_test_collect_user = []

                print("\n" + "=" * 70)
                print(f"ROUND {global_round} SUMMARY (IMPROVED DGL Split Learning)")
                print("=" * 70)
                print(f"Train | Avg Accuracy: {acc_train_collect[-1]:.3f}% | "
                      f"Avg Loss: {loss_train_collect[-1]:.4f}")
                print(f"Test  | Avg Accuracy: {acc_avg_all:.3f}% | "
                      f"Avg Loss: {loss_avg_all:.4f}")
                print("=" * 70 + "\n")


# ============================================================================
# IMPROVED CLIENT CLASS
# ============================================================================

class ImprovedClient:
    """
    IMPROVED Client with:
    - Separate learning rates for main CNN and auxiliary network
    - Label smoothing
    - Dropout-enabled auxiliary network
    """

    def __init__(self, client_id, client_model, train_dataset, test_dataset,
                 train_idxs, test_idxs, config):
        self.client_id = client_id
        self.client_model = client_model
        self.config = config
        self.device = device

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

        # ✅ IMPROVED: Criterion with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    def train_dgl(self, server_model, server_optimizer, global_round):
        """
        IMPROVED DGL Training with separate learning rates
        """
        self.client_model.train()

        # ✅ IMPROVED: Separate parameter groups for different learning rates
        params_main = [p for n, p in self.client_model.named_parameters()
                       if 'auxiliary' not in n]
        params_aux = [p for n, p in self.client_model.named_parameters()
                      if 'auxiliary' in n]

        # ✅ IMPROVED: Different learning rates
        optimizer = torch.optim.SGD([
            {'params': params_main, 'lr': self.config.lr_client},
            {'params': params_aux, 'lr': self.config.lr_aux}  # 2x higher!
        ], momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        for local_epoch in range(self.config.local_epochs):
            len_batch = len(self.train_loader)

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward with auxiliary network
                smashed_data, aux_predictions = self.client_model(
                    images, return_aux=True
                )

                # ✅ Uses label smoothing criterion
                aux_loss = self.criterion(aux_predictions, labels)

                # Backpropagate with auxiliary loss
                aux_loss.backward()
                optimizer.step()

                # Send to server (detached)
                smashed_data_detached = smashed_data.clone().detach()

                train_server(
                    server_model,
                    smashed_data_detached,
                    labels,
                    self.criterion,  # Server also uses label smoothing
                    server_optimizer,
                    self.client_id,
                    self.config.local_epochs,
                    local_epoch,
                    len_batch
                )

        return self.client_model.state_dict()

    def evaluate(self, server_model, global_round):
        self.client_model.eval()

        with torch.no_grad():
            len_batch = len(self.test_loader)

            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                smashed_data = self.client_model(images, return_aux=False)

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
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("INITIALIZING IMPROVED DGL SPLIT LEARNING")
    print("=" * 70)

    # Load data with improved augmentation
    print("\n[1/5] Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, train_splits, test_splits = \
        load_cifar10_data_improved(
            config.num_clients,
            use_strong_augmentation=config.use_strong_augmentation
        )

    # Initialize models
    print("\n[2/5] Initializing models...")

    global_client_model = ImprovedClientSideModel(
        num_classes=config.num_classes,
        aux_type=config.aux_type,
        dropout_2d=config.dropout_2d,  # ✅ IMPROVED
        dropout_1d=config.dropout_1d  # ✅ IMPROVED
    ).to(device)

    global_server_model = ServerSideModel(
        num_classes=config.num_classes
    ).to(device)

    print(f"\nImproved Client Model:")
    print(f"  - Parameters: {sum(p.numel() for p in global_client_model.parameters()):,}")
    print(f"  - Auxiliary Network: {config.aux_type} (with dropout)")
    print(f"  - Dropout rates: 2D={config.dropout_2d}, 1D={config.dropout_1d}")
    print(f"\nServer Model:")
    print(f"  - Parameters: {sum(p.numel() for p in global_server_model.parameters()):,}")

    # Server optimizer
    server_optimizer = torch.optim.SGD(
        global_server_model.parameters(),
        lr=config.lr_server,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # Create clients
    print(f"\n[3/5] Creating {config.num_clients} clients...")
    clients = []
    for client_id in range(config.num_clients):
        client = ImprovedClient(
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

    # Training loop
    print("\n[4/5] Starting IMPROVED DGL Split Learning training...")
    print("=" * 70 + "\n")

    start_time = time.time()

    for global_round in range(1, config.epochs + 1):
        num_selected = max(int(config.client_frac * config.num_clients), 1)
        selected_clients = np.random.choice(
            range(config.num_clients), num_selected, replace=False
        )

        for client_idx in selected_clients:
            client = clients[client_idx]

            # Train with improved DGL
            updated_client_weights = client.train_dgl(
                server_model=global_server_model,
                server_optimizer=server_optimizer,
                global_round=global_round
            )

            # Update models
            global_client_model.load_state_dict(updated_client_weights)
            client.client_model.load_state_dict(updated_client_weights)

            # Evaluate
            client.evaluate(
                server_model=global_server_model,
                global_round=global_round
            )

    elapsed_time = time.time() - start_time

    # Save results
    print("\n[5/5] Saving results...")

    results = {
        'train_loss': loss_train_collect,
        'train_acc': acc_train_collect,
        'test_loss': loss_test_collect,
        'test_acc': acc_test_collect,
        'config': vars(config),
        'training_time': elapsed_time,
        'improvements': [
            'Dropout in auxiliary network',
            'Label smoothing',
            'Separate learning rates',
            'Reduced weight decay',
            'Increased local epochs',
            'Strong data augmentation'
        ]
    }

    torch.save(results, 'improved_dgl_split_learning_results.pt')

    # Create plots
    create_plots(loss_train_collect, acc_train_collect,
                 loss_test_collect, acc_test_collect)

    # Print final results
    print("\n" + "=" * 70)
    print("IMPROVED TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total Training Time: {elapsed_time / 60:.2f} minutes")
    print(f"Final Train Accuracy: {acc_train_collect[-1]:.2f}%")
    print(f"Final Test Accuracy: {acc_test_collect[-1]:.2f}%")
    print(f"\nImprovements Applied:")
    for improvement in results['improvements']:
        print(f"  ✅ {improvement}")
    print(f"\nResults saved to: improved_dgl_split_learning_results.pt")
    print("=" * 70 + "\n")

    return results


def create_plots(train_loss, train_acc, test_loss, test_acc):
    """Create training visualization plots"""

    rounds = list(range(1, len(train_loss) + 1))

    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(rounds, train_loss, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(rounds, test_loss, 'r-s', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('IMPROVED DGL Split Learning - Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(rounds, train_acc, 'b-o', label='Train Accuracy', linewidth=2)
    ax2.plot(rounds, test_acc, 'r-s', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('IMPROVED DGL Split Learning - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('improved_dgl_split_learning_plots.png', dpi=300, bbox_inches='tight')
    print(f"  - Plots saved to: improved_dgl_split_learning_plots.png")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("IMPROVED DGL SPLIT LEARNING")
    print("Backpropagation-Free Training with Priority 1 Enhancements")
    print("=" * 70)

    results = main()