############################################## FROM CIFAR10 #######################################
# import torch
# import torchvision
# import torchvision.transforms as transforms


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# batch_size = 1

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# import matplotlib.pyplot as plt
# import numpy as np


# # get some random training images
# def sample_images(trainloader, output_path, n):
#     dataiter = iter(trainloader)
#     for i in range(n):
#         images, labels = next(dataiter)
#         images = images / 2 + 0.5     # unnormalize
#         torchvision.utils.save_image(images, f"{output_path}/sample_{i}.png")


# output_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/analyse_images/sampled_images'  # Directory to save sampled images

# sample_images(trainloader, output_path, 4)


# ############################# FROM EXTRACTED TRAIN OR VAL FILES #########################################################

# import os
# import random
# from PIL import Image
# import matplotlib.pyplot as plt
# import shutil

# def sample_images_from_imagenet(dataset_path, output_dir=None, num_samples=5):
#     """
#     Sample random images from the ImageNet dataset and optionally save them.
    
#     Args:
#         dataset_path (str): Path to the ImageNet dataset (e.g., 'train' folder).
#         output_dir (str): Directory to save sampled images. If None, images are not saved.
#         num_samples (int): Number of random images to sample.

#     Returns:
#         None
#     """
#     # Get the list of all class directories
#     class_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
#     sampled_images = []  # List to store sampled image paths
    
#     for _ in range(num_samples):
#         # Randomly select a class directory
#         random_class_dir = random.choice(class_dirs)
#         # Get all images in the selected class directory
#         # print('len(random_class_dir) =', len(random_class_dir))
#         image_files = [os.path.join(random_class_dir, img) for img in os.listdir(random_class_dir) if img.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
#         # Randomly select an image from the class
#         # print('len(image_files) =', len(os.listdir(random_class_dir)))
#         sampled_image = random.choice(image_files)
#         sampled_images.append(sampled_image)
    
#     # Display and optionally save sampled images
#     plt.figure(figsize=(15, 5))
#     for i, img_path in enumerate(sampled_images):
#         img = Image.open(img_path)
#         plt.subplot(1, num_samples, i + 1)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(f"Class: {os.path.basename(os.path.dirname(img_path))}")
        
#         # Save image if output directory is specified
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
#             shutil.copy(img_path, os.path.join(output_dir, f"sample_{i + 1}.jpg"))
    
#     plt.show()

# # Example usage
# imagenet_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/train'  # Replace with your ImageNet dataset path
# output_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/analyse_images/sampled_images'  # Directory to save sampled images
# sample_images_from_imagenet(imagenet_path, output_dir=output_path, num_samples=5)



############################# FROM TAR FILES #########################################################

import tarfile
import random
import io
import os
import numpy as np

sampled_images = []  # List to store sampled image data
num_samples = 1  # Number of samples to pick

# def random_seed(seed=1337):
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.manual_seed(seed)
#     # if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.deterministic = True
#     torch.cuda.manual_seed_all(seed)

def sample_images_from_tar(tar_path, output_dir=None, num_samples=1):

    # random_seed(seed=42)
    random.seed(0)

    with tarfile.open(tar_path, "r") as train_tar:
        # Get a list of all class tar files
        class_tars = [member for member in train_tar.getmembers() if member.isfile() and member.name.endswith(".tar")]

        for _ in range(num_samples):
            # Randomly select one class tar file
            sampled_class_tar = random.choice(class_tars)
            print(f"Sampling from class tar: {sampled_class_tar.name}")
            
            # Extract the sampled class tar file into memory
            class_file_obj = train_tar.extractfile(sampled_class_tar)
            class_file_data = io.BytesIO(class_file_obj.read())  # Read into memory

            # Open the class tar file
            with tarfile.open(fileobj=class_file_data, mode="r") as class_tar:
                # Get a list of all image files in the class tar
                image_members = [member for member in class_tar.getmembers() if member.isfile()]
                
                # Randomly sample one image from this class tar
                sampled_image_member = random.choice(image_members)
                
                # Read the sampled image data (image file content)
                image_file_obj = class_tar.extractfile(sampled_image_member)
                image_data = image_file_obj.read()
                
                # Append the image data and info (you can modify this to save or process the images further)
                sampled_images.append((sampled_class_tar.name, sampled_image_member.name, image_data))

    # To visualize or save the images
    for class_tar_name, image_name, image_data in sampled_images:
        # Optionally, save the image data to disk or display it
        # Saving the image
        # output_dir = "sampled_images"
        os.makedirs(output_dir, exist_ok=True)
        # class_dir = os.path.join(output_dir, os.path.splitext(class_tar_name)[0])
        # os.makedirs(class_dir, exist_ok=True)

        output_path = os.path.join(output_dir, os.path.basename(image_name))
        with open(output_path, "wb") as f:
            f.write(image_data)

        # # To display the image (using PIL)
        # from PIL import Image
        # img = Image.open(io.BytesIO(image_data))
        # img.show()


# Example usage
train_tar_path = "/hkfs/work/workspace/scratch/fr_ms2108-data/data/imagenet_tar/ILSVRC2012_img_train.tar"
output_path = '/hkfs/work/workspace/scratch/fr_ms2108-data/Cream/TinyViT/analyse_images/sampled_images'  # Directory to save sampled images
sample_images_from_tar(train_tar_path, output_dir=output_path, num_samples=num_samples)
