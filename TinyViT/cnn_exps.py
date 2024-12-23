import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from golu.activation_utils import replace_activation_by_torch_module, get_activation_function

# # Define the CNN model
# class SimpleCNN(nn.Module):
#     def __init__(self, activation):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: 3x32x32, Output: 16x32x32
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
#         self.fc1 = nn.Linear(32 * 8 * 8, 128)  # After pooling
#         self.fc2 = nn.Linear(128, 10)  # 10 classes for CIFAR-10
#         self.pool = nn.MaxPool2d(2, 2)  # 2x2 max pooling

#         self.activation = activation

#         self.activations1 = []
#         self.activations2 = []
#         self.activations3 = []

#     def forward(self, x):
#         x = self.pool(self.activation(self.conv1(x)))
#         self.activations1.append(x.view(-1))  # Save activations
#         x = self.pool(self.activation(self.conv2(x)))
#         self.activations2.append(x.view(-1))  # Save activations
#         x = x.view(-1, 32 * 8 * 8)
#         x = self.activation(self.fc1(x))
#         self.activations3.append(x.view(-1))  # Save activations
#         x = self.fc2(x)
#         return x

#     def get_activations(self, x):
#         x = self.activation(self.conv1(x))
#         self.activations1.append(x.view(-1).detach().cpu().numpy())  # Collect activations
#         x = self.pool(x)
#         x = self.activation(self.conv2(x))
#         self.activations2.append(x.view(-1).detach().cpu().numpy())  # Collect activations
#         x = self.pool(x)
#         x = x.view(-1, 32 * 8 * 8)
#         x = self.activation(self.fc1(x))
#         self.activations3.append(x.view(-1).detach().cpu().numpy())  # Collect activations
#         # x = self.fc2(x)


# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = activation

        self.activations1 = []
        self.activations2 = []
        self.activations3 = []
        self.activations4 = []

    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_activations(self, x):
        x = self.activation(self.conv1(x))
        self.activations1.append(x.view(-1).detach().cpu().numpy())  # Collect activations
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        self.activations2.append(x.view(-1).detach().cpu().numpy())  # Collect activations
        x = self.pool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.activation(self.fc1(x))
        self.activations3.append(x.view(-1).detach().cpu().numpy())  # Collect activations
        x = self.activation(self.fc2(x))
        self.activations4.append(x.view(-1).detach().cpu().numpy())  # Collect activations
        # x = self.fc3(x)


def compute_layer_outputs(model, dataloader):
    # outputs_collections = {"conv1": [], "conv2": [], "fc1": []}

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # Forward pass
        # x = model.pool(model.activation(model.conv1(inputs)))
        # outputs_collections["conv1"].append(x.view(-1).detach().cpu().numpy())

        # x = model.pool(model.activation(model.conv2(x)))
        # outputs_collections["conv2"].append(x.view(-1).detach().cpu().numpy())

        # x = x.view(-1, 32 * 8 * 8)  # Flatten
        # x = model.activation(model.fc1(x))
        # outputs_collections["fc1"].append(x.view(-1).detach().cpu().numpy())
        model.get_activations(inputs)

    outputs_collections = {"conv1": model.activations1, "conv2": model.activations2, "fc1": model.activations3}

    # Aggregate and return outputs
    return {name: np.concatenate(values) for name, values in outputs_collections.items()}




def plot_output_distributions(output_data, activations, layer_name, path_save_plots):
    plt.figure(figsize=(10, 6))
    for activation_name, outputs in output_data.items():
        plt.hist(outputs, bins=50, alpha=0.6, label=activation_name, density=True)
    plt.title(f"Output Distribution: {layer_name}")
    plt.xlabel("Output Value")
    plt.ylabel("Density")
    plt.legend()

    # Sanitize layer_name for saving
    safe_layer_name = layer_name.replace(".", "_")
    plt.savefig(f"{path_save_plots}/{safe_layer_name}_outputs.png")
    # plt.show()


def plot_activation_distributions(activation_distributions, activation_name):
    num_layers = len(activation_distributions)
    fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

    for i, activations in enumerate(activation_distributions):
        axes[i].hist(activations.flatten(), bins=50, alpha=0.7)
        axes[i].set_title(f"Layer {i + 1} Activations")
        axes[i].set_xlabel("Activation Value")
        axes[i].set_ylabel("Frequency")

    plt.suptitle(f"Activation Output Distributions - {activation_name}")
    plt.tight_layout()
    plt.savefig(activation_name)

# Compute gradients w.r.t. layer weights
def compute_layerwise_gradients(model, criterion, dataloader):
    gradient_distributions = {name: [] for name, _ in model.named_parameters()}
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Collect gradients for each layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradient_distributions[name].append(param.grad.norm().item())

    # Aggregate and return distributions
    return {name: np.array(grads) for name, grads in gradient_distributions.items()}


# def plot_gradient_distributions(gradient_distributions, activation_name):
#     num_layers = len(gradient_distributions)
#     fig, axes = plt.subplots(1, num_layers, figsize=(15, 5))

#     for i, (layer_name, gradients) in enumerate(gradient_distributions.items()):
#         axes[i].hist(gradients, bins=30, alpha=0.7, color='blue')
#         axes[i].set_title(f"{layer_name} Gradients ({activation_name})")
#         axes[i].set_xlabel("Gradient Norm")
#         axes[i].set_ylabel("Frequency")

#     plt.tight_layout()
#     plt.savefig(activation_name)


def plot_gradient_distributions(gradient_data, activations, param_name, path_save_plots):
    plt.figure(figsize=(10, 6))
    for activation_name, gradients in gradient_data.items():
        plt.hist(gradients, bins=50, alpha=0.6, label=activation_name, density=True)
    plt.title(f"Gradient Distribution: {param_name}")
    plt.xlabel("Gradient Value")
    plt.ylabel("Density")
    plt.legend()
    # plt.show()
    # print('param_name =====', param_name)
    # print('param_name =', type(param_name))
    safe_param_name = param_name.replace(".", "_")
    plt.savefig(f"{path_save_plots}/{safe_param_name}_grads.png")




# Function to compute the loss landscape
def compute_loss_landscape(model, criterion, dataloader, grid_size=100):
    losses = []
    gradients = []

    for i, (inputs, targets) in enumerate(dataloader):
        if i >= grid_size:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        inputs.requires_grad = True  # Enable gradient computation w.r.t inputs

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Compute gradients of loss w.r.t. inputs
        grad = torch.autograd.grad(loss, inputs, retain_graph=False, create_graph=False)[0]
        losses.append(loss.item())
        gradients.append(grad.norm().item())

    return np.array(losses), np.array(gradients)

# Visualize loss landscape and gradient norms
def visualize_loss_landscape(losses, gradients, activation_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(losses, label="Loss")
    axes[0].set_title("Loss Landscape")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(gradients, label="Gradient Norm", color="orange")
    axes[1].set_title("Gradient Norms")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Gradient Norm")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(activation_name)

# Aggregate distributions of losses/gradients
def plot_distributions(losses, gradients, activation_name):
    plt.figure(figsize=(12, 6))

    # Loss distribution
    plt.subplot(1, 2, 1)
    plt.hist(losses, bins=20, color='blue', alpha=0.7, label="Loss")
    plt.title(f"{activation_name} Loss Distribution")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.legend()

    # Gradient norm distribution
    plt.subplot(1, 2, 2)
    plt.hist(gradients, bins=20, color='orange', alpha=0.7, label="Gradient Norm")
    plt.title(f"{activation_name} Gradient Norm Distribution")
    plt.xlabel("Gradient Norm")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(activation_name)


# Main function to train the model and analyze the loss landscape
def main(args):
    # CIFAR-10 Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 4

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Define activations to compare
    activations = {
        # "ReLU": nn.ReLU(),
        # "Tanh": nn.Tanh(),
        # "LeakyReLU": nn.LeakyReLU(0.1),
        # "SiLU": nn.SiLU()
        "GoLU": get_activation_function('MyGoLU'),
        # "GELU": get_activation_function('GELU'),
        "SiLU": get_activation_function('Swish'),  # Swish activation
        "ReLU": get_activation_function('ReLU'), 
    }

    output_collections = {name: {} for name in activations.keys()}
    gradient_collections = {name: {} for name in activations.keys()}

    for activation_name, activation_fn in activations.items():
        print(f"Analyzing activation: {activation_name}")

        # Initialize the model
        model = SimpleCNN(activation_fn).to(device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Train the model for a few epochs
        model.train()
        for epoch in range(2):  # Short training for demonstration
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 1000 == 999:  # Print every 1000 mini-batches
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 1000:.3f}")
                    running_loss = 0.0
        
        PATH = f'./cnn_models/cnn_{activation_name}.pth'
        torch.save(model.state_dict(), PATH)


        model = SimpleCNN(activation_fn)
        model.load_state_dict(torch.load(PATH, weights_only=True))
        model.to(device)





        # Analyze the loss landscape
        model.eval()
        # losses, gradients = compute_loss_landscape(model, criterion, trainloader)
        # visualize_loss_landscape(losses, gradients, activation_name)

        # plot_distributions(losses, gradients, activation_name)



        # activation_distributions = extract_activation_distributions(model, trainloader)
        # plot_activation_distributions(activation_distributions, activation_name)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    

        layer_outputs = compute_layer_outputs(model, trainloader)
        output_collections[activation_name] = layer_outputs

        gradient_distributions = compute_layerwise_gradients(model, criterion, trainloader)
        gradient_collections[activation_name] = gradient_distributions

    
    figpath = f"activation_figs/{args.figfolder}"
    
    if not os.path.exists(figpath):
        os.makedirs(figpath, exist_ok=True)

    # Compare output distributions across activations for each layer
    for layer_name in output_collections["GoLU"].keys():
        output_data = {activation_name: outputs[layer_name] for activation_name, outputs in output_collections.items()}
        plot_output_distributions(output_data, activations, layer_name, figpath)

        # Compare gradient distributions across activations for each parameter
    for param_name in gradient_collections["GoLU"].keys():
        gradient_data = {activation_name: gradients[param_name] for activation_name, gradients in gradient_collections.items()}
        plot_gradient_distributions(gradient_data, activations, param_name, figpath)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-figfolder', type=str, default='compare_dists')
    args = parser.parse_args()
    main(args)
