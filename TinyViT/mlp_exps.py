import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from golu.activation_utils import replace_activation_by_torch_module, get_activation_function

# Define the 2-layer MLP
class SimpleMLP(nn.Module):
    def __init__(self, activation):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(2, 32)  # Input: 2D grid, Hidden: 32 units
        self.fc2 = nn.Linear(32, 1)  # Output: 1D scalar
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# # Compute gradients of the output with respect to the input
# def compute_gradients(inputs, outputs):
#     gradients = []
#     for i in range(outputs.size(0)):
#         # Compute gradient of the output w.r.t input
#         grad = torch.autograd.grad(outputs[i], inputs, retain_graph=True, create_graph=False)[0]
#         # print('inputs.shape =', inputs.shape)
#         # print('outputs[i].shape =', outputs[i].shape)
#         # print('grad.shape =', grad.shape)
#         # exit()
#         gradients.append(grad)
#     # print('torch.stack(gradients).shape =', torch.stack(gradients).shape)
#     return torch.stack(gradients)

# Compute gradients of the output with respect to the input
def compute_gradients(inputs, outputs):
    grads = torch.autograd.grad(outputs.sum(), inputs, retain_graph=True, create_graph=False)[0]
    return grads

# Generate a grid of inputs
def generate_input_grid(grid_size=100):
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    xx, yy = torch.meshgrid(x, y)
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    # print('grid.shape =', grid.shape)
    # print('grid =', grid)
    return grid, xx, yy

# Visualize heatmap of outputs
def visualize_heatmap(xx, yy, outputs, title, ax):
    outputs = outputs.detach().cpu().numpy().reshape(xx.shape)
    heatmap = ax.contourf(xx, yy, outputs, levels=50, cmap='coolwarm')
    plt.colorbar(heatmap, ax=ax)
    ax.set_title(title)

# Main comparison code
def compare_activations(activations, grid_size=100):
    input_grid, xx, yy = generate_input_grid(grid_size)
    input_grid.requires_grad = True  # Required for gradient computation

    fig, axes = plt.subplots(len(activations), 2, figsize=(12, len(activations) * 5))
    axes = np.atleast_2d(axes)

    for i, (activation_name, activation_fn) in enumerate(activations.items()):
        # Initialize the model with the given activation
        model = SimpleMLP(activation_fn)
        model.eval()  # Ensure no randomness in the forward pass
        
        # Generate random weights for reproducibility
        # torch.manual_seed(42)
        # for layer in [model.fc1, model.fc2]:
        #     nn.init.normal_(layer.weight, mean=0, std=1)
        #     nn.init.constant_(layer.bias, 0)

        # Pass the input grid through the model
        outputs = model(input_grid)
        
        # Compute variance of the outputs
        variance = torch.var(outputs).item()

        # Compute gradients
        grads = compute_gradients(input_grid, outputs)
        grad_norms = grads.norm(dim=1).detach().cpu().numpy()
        avg_grad_norm = np.mean(grad_norms)

        # Plot the heatmap of outputs
        visualize_heatmap(xx, yy, outputs, f'{activation_name}: Var={variance:.4f}', axes[i, 0])

        # Plot the gradient norm heatmap
        grad_heatmap = grad_norms.reshape(xx.shape)
        heatmap = axes[i, 1].contourf(xx, yy, grad_heatmap, levels=50, cmap='viridis')
        plt.colorbar(heatmap, ax=axes[i, 1])
        axes[i, 1].set_title(f'{activation_name}: Avg Grad Norm={avg_grad_norm:.4f}')

    plt.tight_layout()
    # plt.show()
    plt.savefig('mlp')

# # Define the activations to compare
# activations = {
#     "ReLU": nn.ReLU(),
#     "Tanh": nn.Tanh(),
#     "Sigmoid": nn.Sigmoid(),
#     "LeakyReLU": nn.LeakyReLU(negative_slope=0.1),
#     "SiLU": nn.SiLU(),  # Swish activation
# }

# Define the activations to compare
activations = {
    "GoLU": get_activation_function('MyGoLU'),
    "GELU": get_activation_function('GELU'),
    "SiLU": get_activation_function('Swish'),  # Swish activation
}

# Compare activations
compare_activations(activations)
