# --------------------------------------------------------
# TinyViT Utils (save/load checkpoints, etc.)
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted for TinyViT
# --------------------------------------------------------

# script to run code
# python checkpoint_exps.py --cfg configs/1k/tiny_vit_21m.yaml

import torch
import random
import argparse
import numpy as np

import torch.nn as nn
import torch.backends.cudnn as cudnn

from config import get_config
from models import build_model
from utils import add_common_args

import matplotlib.pyplot as plt
import seaborn as sns

# from custom_activations import replace_ac_function, get_activation_function, GoLU
# from golu.activation_utils import replace_activation_by_torch_module, get_activation_function
# from golu.golu_cuda_activation import GoLUCUDA

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

# try:
#     import wandb
# except ImportError:
#     wandb = None
# NORM_ITER_LEN = 100

# os.environ["WANDB_MODE"]="offline"


def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT training and evaluation script', add_help=False)
    add_common_args(parser)
    args = parser.parse_args()

    config = get_config(args)

    return args, config


args, config = parse_option()
config.defrost()
if config.DISTILL.TEACHER_LOGITS_PATH:
    config.DISTILL.ENABLED = True
config.freeze()


seed = config.SEED
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
cudnn.benchmark = True

config.defrost()
config.freeze()


        


# #############################################################################################
# # model = build_model(config)
# # model.eval()


# # # checkpoint = torch.load('output_{vit-golucuda-cream-horeka}_gelu_{2}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')
# # checkpoint = torch.load('output_{vit-golucuda-cream-horeka}_golu_cuda_{2}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')

# # model.load_state_dict(checkpoint['model'])

# # # model.eval()

# # # print('', model)

# # # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# # # print('number of params:', n_parameters)

# # # print('model:', model)

# # # for name, param in model.named_parameters():
# # #     print('name:', name)


# # ff_layer = model.layers[3].blocks[1].mlp.norm # Adjust the indexing for the specific layer
# # weights = ff_layer.weight.data.cpu().numpy()

# # # Plot the weight distribution
# # plt.figure(figsize=(8, 6))
# # sns.histplot(weights.flatten(), bins=50, kde=True, color='blue')
# # plt.title("Weight Distribution of Feedforward Layer")
# # plt.xlabel("Weight Value")
# # plt.ylabel("Frequency")
# # plt.savefig('fig_go')
# ################################################################################

# # model = build_model(config)

# # # Collect all normalization layers in a dictionary
# # norm_layers = {}

# # for name, module in model.named_modules():
# #     if "norm" in name: # and isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
# #         norm_layers[name] = module.weight.detach().cpu().numpy()

# # # Print the collected normalization layers
# # for name, module in norm_layers.items():
# #     print(f"{name}: {module}")
# ################################################################################

# # layers[1].blocks[1].mlp.fc1
# # layers[1].blocks[1].mlp.norm
# # layers[0].downsample.conv2.c
# # layers[2].blocks[5].mlp.norm
# # layers[2].blocks[5].attn.qkv
# # layers[2].blocks[5].attn.proj
# # ################################ SINGLE PLOT ################################################


# checkpoint1 = torch.load('output_{vit-golucuda-cream-horeka}_golu_cuda_{1}/TinyViT-21M-1k/default/ckpt_epoch_260.pth')
# # checkpoint2 = torch.load('output_{vit-golucuda-cream-horeka}_silu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')
# # checkpoint3 = torch.load('output_{vit-golucuda-cream-horeka}_gelu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')
# checkpoint4 = torch.load('output_{vit-golucuda-cream-horeka}_relu_{1}/TinyViT-21M-1k/default/ckpt_epoch_260.pth')
# # checkpoint5 = torch.load('output_{vit-golucuda-cream-horeka}_elu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')


# # checkpoints = [checkpoint1, checkpoint2, checkpoint3, checkpoint4, checkpoint5]
# # colors = ['orange', 'blue', 'red', 'brown', 'green']
# # labels = ['GoLU', 'SiLU', 'GELU', 'ReLU', 'ELU']

# checkpoints = [checkpoint1, checkpoint2, checkpoint3, checkpoint4]
# colors = ['orange', 'blue', 'red', 'green']
# labels = ['GoLU', 'SiLU', 'GELU', 'ReLU',]

# models = {}
# weights = {}


# for i, checkpoint in enumerate(checkpoints):
#     models[i] = build_model(config)
#     models[i].eval()

#     models[i].load_state_dict(checkpoint['model'])
#     ff_layer = models[i].layers[2].blocks[5].mlp.norm # Adjust the indexing for the specific layer
#     weights[i] = ff_layer.weights.data.cpu().numpy().flatten()


# # Plot the distributions
# plt.figure(figsize=(10, 6))
# for i in range(len(checkpoints)):
#     sns.histplot(weights[i], bins=50, kde=False, color=colors[i], label=labels[i], stat="density")

# # plt.title("Weight Distribution Comparison for Feedforward Layer")
# plt.title("Weight Distribution Comparison for LayerNorm")

# plt.xlabel("Weight Value")
# plt.ylabel("Density")
# plt.legend()
# plt.savefig('fig_norm_weights')



# ################################## GRID ##############################################


# checkpoint1 = torch.load('output_{vit-golucuda-cream-horeka}_golu_cuda_{1}/TinyViT-21M-1k/default/ckpt_epoch_260.pth')
# # checkpoint2 = torch.load('output_{vit-golucuda-cream-horeka}_silu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')
# checkpoint3 = torch.load('output_{vit-golucuda-cream-horeka}_gelu_{1}/TinyViT-21M-1k/default/ckpt_epoch_260.pth')
# # checkpoint4 = torch.load('output_{vit-golucuda-cream-horeka}_relu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')
# # checkpoint5 = torch.load('output_{vit-golucuda-cream-horeka}_elu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth')


# # checkpoints = [checkpoint1, checkpoint2, checkpoint3, checkpoint4, checkpoint5]
# # colors = ['orange', 'blue', 'red', 'brown', 'green']
# # labels = ['GoLU', 'SiLU', 'GELU', 'ReLU', 'ELU']

# checkpoints = [checkpoint1, checkpoint3]
# colors = ['orange', 'blue']
# labels = ['GoLU', 'GELU']

# models = {}
# norm_weights = {}
# norm_names = []

# for i, checkpoint in enumerate(checkpoints):
#     models[i] = build_model(config)
#     models[i].eval()

#     models[i].load_state_dict(checkpoint['model'])

#     # Collect all normalization layers in a dictionary
#     for name, module in models[i].named_modules():
#         if "norm" in name: # and isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)): # Choice1
#             if hasattr(module, 'weight') and module.weight is not None: # Choice1
#                 print(f"{name}: {module}")
#                 if i==0:
#                     norm_names.append(name)
#                 # norm_weights[(i, name)] = module.weight.data.cpu().numpy().flatten() # Choice2
#                 norm_weights[(i, name)] = module.bias.data.cpu().numpy().flatten() # Choice2





# k = len(checkpoints)
# # n = len(norm_weights) // k

# num_layers = len(norm_weights) // k
# print('num_layers =', num_layers)
# cols = 4  # Number of columns in the grid
# rows = (num_layers + cols - 1) // cols  # Calculate number of rows needed

# fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))  # Adjust figsize as needed
# axes = axes.flatten()  # Flatten the axes for easy indexing


# for idx, name in enumerate(norm_names):
#     ax = axes[idx]
#     for i in range(k):
#         # ax.hist(norm_weights[(i, name)], bins=50, alpha=0.5, label=labels[i])
#         sns.histplot(norm_weights[(i, name)], bins=50, kde=False, color=colors[i], label=labels[i], ax=ax, stat="density")
#         ax.set_title(f"{name}", fontsize=12)
#         ax.set_xlabel("Value")
#         ax.set_ylabel("Frequency")
#         ax.legend()

    

# # plt.title("Weight Distribution Comparison for Feedforward Layer")
# # plt.title("Weight Distribution Comparison for LayerNorm")

# plt.xlabel("Weight Value")
# plt.ylabel("Density")
# # plt.legend()

# for idx in range(len(norm_names), len(axes)):
#     axes[idx].axis('off')

# # plt.savefig('fig_norm_weights')
# plt.savefig('fig_norm_biases')


# # ########################################## SINGLE PLOT ALL WEIGHTS - SINGLE SEED #################################

# # torch.serialization.add_safe_globals([CfgNode])

# checkpoint1 = torch.load('output_{vit-golucuda-cream-horeka}_golu_cuda_{1}/TinyViT-21M-1k/default/ckpt_epoch_277.pth', weights_only=False)
# # checkpoint2 = torch.load('output_{vit-golucuda-cream-horeka}_silu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth', weights_only=False)
# checkpoint3 = torch.load('output_{vit-golucuda-cream-horeka}_gelu_{1}/TinyViT-21M-1k/default/ckpt_epoch_277.pth', weights_only=False)
# checkpoint4 = torch.load('output_{vit-golucuda-cream-horeka}_relu_{1}/TinyViT-21M-1k/default/ckpt_epoch_277.pth', weights_only=False)
# # checkpoint5 = torch.load('output_{vit-golucuda-cream-horeka}_elu_{1}/TinyViT-21M-1k/default/ckpt_epoch_299.pth', weights_only=False)


# checkpoints = [checkpoint1, checkpoint3, checkpoint4]
# colors = ['orange', 'blue', 'green']
# labels = ['GoLU', 'GELU', 'ReLU']


# models = {}
# weights = {}
# vars = {}

# # for i, checkpoint in enumerate(checkpoints):
# #     weights[i] = []

# # for i, checkpoint in enumerate(checkpoints):
# #     print(f'weights[{i}]=', weights[i]) 

# for i, checkpoint in enumerate(checkpoints):

#     ws = []
    
#     models[i] = build_model(config)
#     models[i].eval()
#     models[i].load_state_dict(checkpoint['model'])
    
#     # Collect all parameters in a list
#     for p in models[i].parameters():
#         ws.append(p.data.cpu().numpy().flatten())
        
#     # Concatenate the list to get a long flattened numpy array of all weights
#     weights[i] = np.concatenate(ws)

#     # # Collect all parameters in a dictionary
#     # for n, p in models[i].named_parameters():
#     #     if 'weights' in p:
#     #         weights[i] = p.data.cpu().numpy().flatten() # Choice2


# # Plot the distributions
# plt.figure(figsize=(10, 6))
# for i in range(len(checkpoints)):
#     w = weights[i]
#     var = np.var(w).item()
#     label = f"{labels[i]} - Var:{round(var, 5)}"
#     sns.histplot(w, bins=100, binrange=(-0.2, 0.2), kde=False, color=colors[i], label=label, stat="density",element="step", fill=False) #, log_scale=True)

# # plt.title("Weight Distribution Comparison for Feedforward Layer")
# plt.title("Weight Distribution Comparison for ImageNet-1K")

# plt.xlabel("Parameter Values")
# plt.ylabel("Density")
# plt.legend()
# plt.savefig('fig_all_weights')



# ########################################## SINGLE PLOT ALL WEIGHTS - ALL SEEDS #################################

# torch.serialization.add_safe_globals([CfgNode])

checkpoint1 = [torch.load(f"output_{{vit-golucuda-cream-horeka}}_golu_cuda_{{{i}}}/TinyViT-21M-1k/default/ckpt_epoch_299.pth", weights_only=False) for i in range(1,4)]
# checkpoint2 = [torch.load(f"output_{{vit-golucuda-cream-horeka}}_silu_{{{i}}}/TinyViT-21M-1k/default/ckpt_epoch_299.pth", weights_only=False) for i in range(1,4)]
checkpoint3 = [torch.load(f"output_{{vit-golucuda-cream-horeka}}_gelu_{{{i}}}/TinyViT-21M-1k/default/ckpt_epoch_299.pth", weights_only=False) for i in range(1,4)]
checkpoint4 = [torch.load(f"output_{{vit-golucuda-cream-horeka}}_relu_{{{i}}}/TinyViT-21M-1k/default/ckpt_epoch_299.pth", weights_only=False) for i in range(1,4)]
# checkpoint5 = [torch.load("output_{{vit-golucuda-cream-horeka}}_elu_{{{i}}}/TinyViT-21M-1k/default/ckpt_epoch_299.pth", weights_only=False) for i in range(1,4)]

# for n in range(1,4):
#     print('n =', n)

checkpoints = [checkpoint1, checkpoint3, checkpoint4]
colors = ['red', 'blue', 'green']
labels = ['GoLU', 'GELU', 'ReLU']


models = {}
weights = {}
vars = {}

# for i, checkpoint in enumerate(checkpoints):
#     weights[i] = []

# for i, checkpoint in enumerate(checkpoints):
#     print(f'weights[{i}]=', weights[i]) 

for i, checkpoint in enumerate(checkpoints):
    print(f'Collecting parameters for {labels[i]}')

    ws = []

    for j, checkpoint_one_seed in enumerate(checkpoint):
        print(f'Finished seed {j+1}')
    
        models[j] = build_model(config)
        models[j].eval()
        models[j].load_state_dict(checkpoint_one_seed['model'])
        
        # Collect all parameters in a list
        for p in models[j].parameters():
            ws.append(p.data.cpu().numpy().flatten())
        
    # Concatenate the list to get a long flattened numpy array of all weights
    weights[i] = np.concatenate(ws)

    # # Collect all parameters in a dictionary
    # for n, p in models[i].named_parameters():
    #     if 'weights' in p:
    #         weights[i] = p.data.cpu().numpy().flatten() # Choice2


# Plot the distributions
plt.figure(figsize=(10, 6))
for i in range(len(checkpoints)):
    print(f'Plotting histogram for {labels[i]}')
    w = weights[i]
    var = np.var(w).item()
    label = f"{labels[i]} - Var:{round(var, 5)}"
    sns.histplot(w, bins=50, binrange=(-0.2, 0.2), kde=False, color=colors[i], label=label, stat="density", element="step", fill=False) #, log_scale=True)

# plt.title("Weight Distribution Comparison for Feedforward Layer")
plt.title("Weight Distribution Comparison for ImageNet-1K")

plt.xlabel("Parameter Values")
plt.ylabel("Density")
plt.legend()
plt.savefig('fig_all_weights')

