"""
Standard Split Learning (Baseline for Comparison)
==================================================
This is the traditional Split Learning implementation where:
1. Client sends smashed data to server
2. Server sends gradients back to client
3. Client backpropagates using server gradients

This will be compared against DGL Split Learning to demonstrate the
benefits of backpropagation-free training.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prRed(text):
    print("\033[91m {}\033[00m".format(text))


def prGreen(text):
    print("\033[92m {}\033[00m".format(text))


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    num_clients = 10
    epochs = 100
    client_frac = 1.0
    local_epochs = 5
    batch_size = 128
    test_batch_size = 100
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_classes = 10


config = Config()


# ============================================================================
# MODEL ARCHITECTURES (Same as DGL for fair comparison)
# ============================================================================


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ClientSideModel_Baseline(nn.Module):
    """
    Standard Split Learning Client Model
    NO auxiliary network - relies on server gradients
    """

    def __init__(self):
        super(ClientSideModel_Baseline, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        return out


class ServerSideModel_Baseline(nn.Module):
    """Standard Split Learning Server Model"""

    def __init__(self, num_classes=10):
        super(ServerSideModel_Baseline, self).__init__()

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
# DATA LOADING
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


def load_cifar10_data(num_clients):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )

    train_splits = create_iid_data_split(train_dataset, num_clients)
    test_splits = create_iid_data_split(test_dataset, num_clients)

    return train_dataset, test_dataset, train_splits, test_splits


# ============================================================================
# SERVER FUNCTIONS (Standard Split Learning)
# ============================================================================

# Global metrics
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


def calculate_accuracy(predictions, labels):
    _, predicted = predictions.max(1)
    correct = predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy


def train_server_baseline(
    server_model,
    smashed_data,
    labels,
    criterion,
    optimizer,
    idx,
    local_epoch,
    local_epoch_count,
    len_batch,
):
    """
    Standard Split Learning Server Training
    RETURNS gradients to client (different from DGL)
    """
    global batch_acc_train, batch_loss_train, count_train
    global loss_train_collect_user, acc_train_collect_user
    global idx_collect, l_epoch_check, fed_check
    global loss_train_collect, acc_train_collect

    server_model.train()
    optimizer.zero_grad()

    smashed_data = smashed_data.to(device)
    labels = labels.to(device)

    # Forward pass
    predictions = server_model(smashed_data)

    # Compute loss
    loss = criterion(predictions, labels)
    accuracy = calculate_accuracy(predictions, labels)

    # Backward pass
    loss.backward()

    # KEY DIFFERENCE: Extract gradients for smashed data to send back to client
    dfx_client = smashed_data.grad.clone().detach()

    optimizer.step()

    # Tracking
    batch_loss_train.append(loss.item())
    batch_acc_train.append(accuracy)

    count_train += 1
    if count_train == len_batch:
        acc_avg = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count_train = 0

        prRed(
            f"Client{idx} Server Train => Epoch {local_epoch_count+1}/{local_epoch} | "
            f"Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}"
        )

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

    # Return gradients to client
    return dfx_client


def evaluate_server_baseline(
    server_model, smashed_data, labels, criterion, idx, len_batch, global_round
):
    """Standard Split Learning Evaluation"""
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

            prGreen(
                f"Client{idx} Server Test  =>                  | "
                f"Acc: {acc_avg:.3f}% | Loss: {loss_avg:.4f}"
            )

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
                print(f"ROUND {global_round} SUMMARY (Standard Split Learning)")
                print("=" * 70)
                print(
                    f"Train | Avg Accuracy: {acc_train_collect[-1]:.3f}% | "
                    f"Avg Loss: {loss_train_collect[-1]:.4f}"
                )
                print(
                    f"Test  | Avg Accuracy: {acc_avg_all:.3f}% | "
                    f"Avg Loss: {loss_avg_all:.4f}"
                )
                print("=" * 70 + "\n")


# ============================================================================
# CLIENT CLASS (Standard Split Learning)
# ============================================================================


class Client_Baseline:
    """
    Standard Split Learning Client
    KEY DIFFERENCE: Receives and uses gradients from server
    """

    def __init__(
        self,
        client_id,
        client_model,
        train_dataset,
        test_dataset,
        train_idxs,
        test_idxs,
        config,
    ):
        self.client_id = client_id
        self.client_model = client_model
        self.config = config
        self.device = device

        self.train_loader = DataLoader(
            DatasetSplit(train_dataset, train_idxs),
            batch_size=config.batch_size,
            shuffle=True,
        )
        self.test_loader = DataLoader(
            DatasetSplit(test_dataset, test_idxs),
            batch_size=config.test_batch_size,
            shuffle=False,
        )

        self.criterion = nn.CrossEntropyLoss()

    def train_standard(self, server_model, server_optimizer, global_round):
        """
        Standard Split Learning Training
        Uses gradients from server for backpropagation
        """
        self.client_model.train()

        optimizer = torch.optim.SGD(
            self.client_model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        for local_epoch in range(self.config.local_epochs):
            len_batch = len(self.train_loader)

            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward through client
                smashed_data = self.client_model(images)

                # IMPORTANT: Smashed data needs gradients for backprop
                smashed_data_with_grad = (
                    smashed_data.clone().detach().requires_grad_(True)
                )

                # Send to server and get gradients back
                dfx = train_server_baseline(
                    server_model,
                    smashed_data_with_grad,
                    labels,
                    self.criterion,
                    server_optimizer,
                    self.client_id,
                    self.config.local_epochs,
                    local_epoch,
                    len_batch,
                )

                # Backpropagate using gradients from server
                smashed_data.backward(dfx)
                optimizer.step()

        return self.client_model.state_dict()

    def evaluate(self, server_model, global_round):
        """Evaluate client model"""
        self.client_model.eval()

        with torch.no_grad():
            len_batch = len(self.test_loader)

            for batch_idx, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                smashed_data = self.client_model(images)

                evaluate_server_baseline(
                    server_model,
                    smashed_data,
                    labels,
                    self.criterion,
                    self.client_id,
                    len_batch,
                    global_round,
                )


# ============================================================================
# MAIN TRAINING
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("STANDARD SPLIT LEARNING (BASELINE)")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading CIFAR-10 dataset...")
    train_dataset, test_dataset, train_splits, test_splits = load_cifar10_data(
        config.num_clients
    )

    # Initialize models
    print("\n[2/4] Initializing models...")
    global_client_model = ClientSideModel_Baseline().to(device)
    global_server_model = ServerSideModel_Baseline(num_classes=config.num_classes).to(
        device
    )

    print(
        f"\nClient Model Parameters: {sum(p.numel() for p in global_client_model.parameters()):,}"
    )
    print(
        f"Server Model Parameters: {sum(p.numel() for p in global_server_model.parameters()):,}"
    )

    server_optimizer = torch.optim.SGD(
        global_server_model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    # Create clients
    print(f"\n[3/4] Creating {config.num_clients} clients...")
    clients = []
    for client_id in range(config.num_clients):
        client = Client_Baseline(
            client_id=client_id,
            client_model=copy.deepcopy(global_client_model).to(device),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            train_idxs=train_splits[client_id],
            test_idxs=test_splits[client_id],
            config=config,
        )
        clients.append(client)

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 70 + "\n")

    start_time = time.time()

    for global_round in range(1, config.epochs + 1):
        num_selected = max(int(config.client_frac * config.num_clients), 1)
        selected_clients = np.random.choice(
            range(config.num_clients), num_selected, replace=False
        )

        for client_idx in selected_clients:
            client = clients[client_idx]

            # Train with standard Split Learning (uses server gradients)
            updated_client_weights = client.train_standard(
                server_model=global_server_model,
                server_optimizer=server_optimizer,
                global_round=global_round,
            )

            # Update models
            global_client_model.load_state_dict(updated_client_weights)
            client.client_model.load_state_dict(updated_client_weights)

            # Evaluate
            client.evaluate(server_model=global_server_model, global_round=global_round)

    elapsed_time = time.time() - start_time

    # Save results
    results = {
        "train_loss": loss_train_collect,
        "train_acc": acc_train_collect,
        "test_loss": loss_test_collect,
        "test_acc": acc_test_collect,
        "config": vars(config),
        "training_time": elapsed_time,
    }

    torch.save(results, "baseline_split_learning_results.pt")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Total Training Time: {elapsed_time/60:.2f} minutes")
    print(f"Final Train Accuracy: {acc_train_collect[-1]:.2f}%")
    print(f"Final Test Accuracy: {acc_test_collect[-1]:.2f}%")
    print(f"Results saved to: baseline_split_learning_results.pt")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    results = main()
