import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import wandb

from golu.activation_utils import replace_activation_by_torch_module, get_activation_function



def random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

random_seed()

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, activation):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: 3x32x32, Output: 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # After pooling
        self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
        self.activation = activation
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))  # Conv1 -> Activation -> Pooling
        x = self.pool(self.activation(self.conv2(x)))  # Conv2 -> Activation -> Pooling
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = self.activation(self.fc1(x))  # Fully Connected Layer 1
        x = self.fc2(x)  # Fully Connected Layer 2
        return x

# Grid search function
def grid_search(activations, batch_sizes, learning_rates, args, epochs=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    results = {activation_name: {"loss": np.zeros((len(batch_sizes), len(learning_rates))),
                                 "accuracy": np.zeros((len(batch_sizes), len(learning_rates)))}
               for activation_name in activations.keys()}

    for activation_name, activation_fn in activations.items():
        print(f"Running grid search for activation: {activation_name}")

        for i, batch_size in enumerate(batch_sizes):
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

            for j, lr in enumerate(learning_rates):
                print(f"batch size: {batch_size}")
                print(f"learning rate: {lr}")

                if args.log_wandb:
                    wandb.init(project="cnn-grid-search",
                        name=f"{activation_name}_bs{batch_size}_lr{lr}",
                        config={"activation": activation_name, "batch_size": batch_size, "learning_rate": lr})

                model = SimpleCNN(activation_fn).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

                # Train the model
                model.train()
                for epoch in range(epochs):
                    running_loss = 0.0
                    train_loss = 0
                    for k, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item()
                        train_loss += loss.item() 
                        if k % 100 == 99:  # Print every 100 mini-batches
                            print(f"[{epoch + 1}, {k + 1}] loss: {running_loss / 100:.3f}")
                            running_loss = 0.0
                    average_epoch_loss = train_loss/(k+1)
                    if args.log_wandb:
                        wandb.log({"epoch": epoch, "train_loss": average_epoch_loss})


                gridpath = f"cnn_models/cnn_grid"
                if not os.path.exists(gridpath):
                    os.makedirs(gridpath, exist_ok=True)

                PATH = f'./{gridpath}/cnn_{activation_name}.pth'
                torch.save(model.state_dict(), PATH)

                # model = SimpleCNN(activation_fn)
                # model.load_state_dict(torch.load(PATH, weights_only=True))
                # model.to(device)

                # Evaluate the model
                model.eval()
                total_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in testloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        total_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                avg_loss = total_loss / len(testset)
                accuracy = correct / total

                results[activation_name]["loss"][i, j] = avg_loss
                results[activation_name]["accuracy"][i, j] = accuracy

                if args.log_wandb:
                    wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
                    wandb.finish()

                # if usewandb:
                #     wandb.log({
                #         'epoch': epoch,
                #         'train_loss': train_loss,
                #         'test_loss': avg_loss,
                #         'test_acc': accuracy,
                #         # 'best_val_acc': result['best_val_acc'],
                #         # 'best_epoch': result['best_epoch'],
                #         # 'epoch time': time.time() - start,
                #         # 'total time': time.time() - start_all,
                #         "id": args.id,
                #         "seed": args.seed,
                #         "run_name": args.wandb_name,
                #     })

    return results

# Plot heatmaps
def plot_heatmaps(results, batch_sizes, learning_rates, path):
    for activation_name, metrics in results.items():
        for metric_name, grid in metrics.items():
            plt.figure(figsize=(8, 6))
            plt.imshow(grid, cmap="viridis", aspect="auto", origin="lower",
                       extent=[learning_rates[0], learning_rates[-1], batch_sizes[0], batch_sizes[-1]])
            plt.colorbar(label=metric_name.capitalize())
            plt.title(f"{metric_name.capitalize()} Heatmap for {activation_name}")
            plt.xlabel("Learning Rate")
            plt.ylabel("Batch Size")
            plt.xticks(learning_rates)
            plt.yticks(batch_sizes)
            plt.savefig(f"{path}/{activation_name}_{metric_name}_heatmap.png")
            # plt.show()

# Find best hyperparameters
def find_best_hyperparameters(results, batch_sizes, learning_rates):
    best_hyperparams = {}
    for activation_name, metrics in results.items():
        best_idx = np.unravel_index(np.argmax(metrics["accuracy"]), metrics["accuracy"].shape)
        best_batch_size = batch_sizes[best_idx[0]]
        best_learning_rate = learning_rates[best_idx[1]]
        best_hyperparams[activation_name] = (best_batch_size, best_learning_rate)
        print(f"Best for {activation_name}: Batch Size={best_batch_size}, Learning Rate={best_learning_rate}, "
              f"Accuracy={metrics['accuracy'][best_idx]:.4f}")
    return best_hyperparams

# Main function
def main(args):

    # if args.log_wandb:
    #     import wandb

    activations = {
        # "ReLU": nn.ReLU(),
        # "Tanh": nn.Tanh(),
        # "LeakyReLU": nn.LeakyReLU(0.1),
        # "SiLU": nn.SiLU()
        # "GoLU": get_activation_function('MyGoLU'),
        "GoLU": get_activation_function('GoLUCUDA'),
        # "GELU": get_activation_function('GELU'),
        # "SiLU": get_activation_function('Swish'),  # Swish activation
        "ReLU": get_activation_function('ReLU'), 
    }

    # batch_sizes = [4, 8, 16, 32]
    # learning_rates = [0.0001, 0.001, 0.01, 0.1]

    batch_sizes = [4, 8, 16]
    learning_rates = [0.0001, 0.001, 0.01]

    # batch_sizes = [4, 8]
    # learning_rates = [0.0001, 0.001]

    figpath = f"activation_figs/{args.figfolder}"
    if not os.path.exists(figpath):
        os.makedirs(figpath, exist_ok=True)

    results = grid_search(activations, batch_sizes, learning_rates, args, epochs=10)
    plot_heatmaps(results, batch_sizes, learning_rates, figpath)
    best_hyperparams = find_best_hyperparameters(results, batch_sizes, learning_rates)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-figfolder', type=str, default='grid_heatmaps')
    parser.add_argument('--log-wandb', action='store_true', default=False, help='log training and validation metrics to wandb')
    parser.add_argument('--wandb_project', default='cnn_grid_exps', type=str, help="wandb project")
    # parser.add_argument('--wandb_name', default='eval', type=str, help="wandb name")
    args = parser.parse_args()
    main(args)
