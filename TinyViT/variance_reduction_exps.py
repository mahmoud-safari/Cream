import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import seaborn as sns
from golu.activation_utils import replace_activation_by_torch_module, get_activation_function
import random

# # Load and preprocess the image
# def load_image(image_path):
#     img = Image.open(image_path).convert("RGB")  # Ensure RGB format
#     img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#     img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW

#     mean = img_tensor.mean(dim=(1, 2))  # Compute mean across H, W for each channel
#     std = img_tensor.std(dim=(1, 2))    # Compute std across H, W for each channel

#     # Normalize image
#     normalized_img = (img_tensor - mean[:, None, None]) / std[:, None, None]
#     return normalized_img

# Load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")  # Ensure RGB format
    img = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # Convert to CxHxW

    # mean = img_tensor.mean(dim=(1, 2))  # Compute mean across H, W for each channel
    # std = img_tensor.std(dim=(1, 2))    # Compute std across H, W for each channel

    # # Normalize image
    # normalized_img = (img_tensor - mean[:, None, None]) / std[:, None, None]
    # return normalized_img
    return img_tensor


# Apply activation function to the image
def apply_activation(img_tensor, activation):
    # print('img_tensor.shape =', img_tensor.shape)

    # layer_norm = torch.nn.LayerNorm(img_tensor.shape)
    # img_tensor_norm = layer_norm(img_tensor)

    c = torch.nn.Conv2d(3, 3, 3)
    m = torch.nn.BatchNorm2d(3, affine=False)

    # img_tensor_conv = c(img_tensor)
    # img_tensor_norm = m(img_tensor_conv.unsqueeze(0)).squeeze(0)

    img_tensor = c(img_tensor)
    img_tensor = m(img_tensor.unsqueeze(0)).squeeze(0)


    # img_flat = img_tensor_norm.view(3, -1)  # Flatten each color channel (C, H*W)
    img_flat = img_tensor.view(3, -1)  # Flatten each color channel (C, H*W)
    activated = activation(img_flat)  # Apply activation
    # return activated.view_as(img_tensor_conv)  # Reshape back to original size
    return activated.view_as(img_tensor)  # Reshape back to original size


# Apply activation function to the image
def get_pre_activation(img_tensor):
    # print('img_tensor.shape =', img_tensor.shape)

    # layer_norm = torch.nn.LayerNorm(img_tensor.shape)
    # img_tensor_norm = layer_norm(img_tensor)

    c = torch.nn.Conv2d(3, 3, 3)
    m = torch.nn.BatchNorm2d(3, affine=False)

    # img_tensor_conv = c(img_tensor)
    # img_tensor_norm = m(img_tensor_conv.unsqueeze(0)).squeeze(0)

    img_tensor = c(img_tensor)
    img_tensor = m(img_tensor.unsqueeze(0)).squeeze(0)

    return img_tensor


# Plot distributions
def plot_distributions(preactiv, output1, output2, output3, act1_name, act2_name, act3_name, image_name, path_to_save='compare_dist'):
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    preactiv_var = torch.var(preactiv).item()
    output1_var = torch.var(output1).item()
    output2_var = torch.var(output2).item()
    output3_var = torch.var(output3).item()
    
    plt.figure(figsize=(5, 5))
    
    # Plot original image distribution
    sns.histplot(img_tensor.detach().numpy().flatten(), bins=50, kde=True, color="red", fill=False, element="step", label=f"Pre-activation - Var:{round(preactiv_var, 3)}", stat="density")
    sns.histplot(output1.detach().numpy().flatten(), bins=50, kde=True, color="orange", fill=False, element="step", label=f"{act1_name} - Var:{round(output1_var, 3)}", stat="density")
    sns.histplot(output2.detach().numpy().flatten(), bins=50, kde=True, color="blue", fill=False, element="step", label=f"{act2_name} - Var:{round(output2_var, 3)}", stat="density")
    sns.histplot(output3.detach().numpy().flatten(), bins=50, kde=True, color="green", fill=False, element="step", label=f"{act3_name} - Var:{round(output3_var, 3)}", stat="density")
    plt.title("Comparison of histograms of different activations")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    
    # # Plot first activation's output distribution
    # axes[1].hist(output1.numpy().flatten(), bins=100, color="green", alpha=0.7)
    # axes[1].set_title(f"After {act1_name}")
    # axes[1].set_xlabel("Pixel Value")
    
    # # Plot second activation's output distribution
    # axes[2].hist(output2.numpy().flatten(), bins=100, color="orange", alpha=0.7)
    # axes[2].set_title(f"After {act2_name}")
    # axes[2].set_xlabel("Pixel Value")
    
    # plt.tight_layout()
    # plt.show()
    plt.legend()
    path_to_folder = f"{path_to_save}/output_dists"
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder, exist_ok=True)
    save_path = os.path.join(path_to_folder, f"{image_name}.png")
    plt.savefig(save_path)

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def plot_image_outputs(original_image, outputs, activations, image_name, save_dir="output_images"):
    """
    Plots the original image and the outputs of different activation functions.
    Args:
        original_image (torch.Tensor): Original input image.
        outputs (list): List of output tensors from activation functions.
        activations (list): List of activation function names.
        save_dir (str): Directory to save the output images.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Convert tensors to numpy for plotting

    # for output in outputs:
    #     output = (output - output.min()) / (output.max() - output.min())



    original_image_np = normalize(original_image).permute(1, 2, 0).detach().numpy()  # (H, W, C)
    output_images_np = [normalize(output).permute(1, 2, 0).detach().numpy() for output in outputs]

    # Plot the images
    num_images = len(outputs) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # Plot the original image
    axes[0].imshow(original_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Save each output image
    path_to_folder = f"{save_dir}/output_images"
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder, exist_ok=True)

    save_path = os.path.join(path_to_folder, f"{image_name}_preactivation.png")
    plt.imsave(f'{save_path}', original_image_np, cmap="viridis")
    print(f"Saved preactivation at {save_path}")

    # Plot each activation's output
    for i, (output_np, activation) in enumerate(zip(output_images_np, activations)):
        axes[i + 1].imshow(output_np, cmap="viridis")
        axes[i + 1].set_title(f"Activation: {activation}")
        axes[i + 1].axis("off")

        # Save each output image
        path_to_folder = f"{save_dir}/output_images/{activation}"
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder, exist_ok=True)

        save_path = os.path.join(path_to_folder, f"{image_name}_{activation}.png")
        plt.imsave(f'{save_path}', output_np, cmap="viridis")
        print(f"Saved output for {activation} at {save_path}")

    # Display the plot
    plt.tight_layout()
    # plt.show()

def random_seed(seed=1337):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

# Main script
if __name__ == "__main__":

    random_seed(seed=42)

    # Load image (replace with your image path)
    # image_path = "/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/pexels-wojciech-kumpicki-1084687-2071882.jpg"
    # image_path = "/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/cat.png"

    input_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/analyse_images/sampled_images'
    output_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/analyse_images'  # Directory to save sampled images

    # Define two activation functions
    activation1 = get_activation_function('MyGoLU')
    activation2 = get_activation_function('GELU')
    activation3 = get_activation_function('Swish')

    # img_tensor = load_image(image_path)


    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_tensor = load_image(file_path)


        pre_activation = get_pre_activation(img_tensor)

        # Apply activations
        output1 = apply_activation(img_tensor, activation1)
        output2 = apply_activation(img_tensor, activation2)
        output3 = apply_activation(img_tensor, activation3)

        # output1 = (output1 - output1.min()) / (output1.max() - output1.min())
        # output2 = (output2 - output2.min()) / (output2.max() - output2.min())
        # output3 = (output3 - output3.min()) / (output3.max() - output3.min())


        # Plot and compare distributions
        plot_distributions(pre_activation, output1, output2, output3, "GoLU", "GELU", 'Swish', file_name, path_to_save=output_path)

        original_var = torch.var(pre_activation).item()
        output1_var = torch.var(output1).item()
        output2_var = torch.var(output2).item()
        output3_var = torch.var(output3).item()
        print(f"Original Variance for image {file_name}: {original_var}")
        print(f"Variance after GoLU for image {file_name}: {output1_var}")
        print(f"Variance after GELU for image {file_name}: {output2_var}")
        print(f"Variance after Swish for image {file_name}: {output3_var}")

        # output1 = (output1 - output1.min()) / (output1.max() - output1.min())
        # output2 = (output2 - output2.min()) / (output2.max() - output2.min())
        # output3 = (output3 - output3.min()) / (output3.max() - output3.min())

        plot_image_outputs(pre_activation, [output1, output2, output3], ["GoLU", "GELU", 'Swish'], file_name, save_dir=output_path)



