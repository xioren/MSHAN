import os
import random

import torch
import numpy as np
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.image_pairs = self._load_image_pairs()

    def _load_image_pairs(self):
        image_pairs = []
        for hr_file in os.listdir(self.hr_dir):
            hr_path = os.path.join(self.hr_dir, hr_file)
            lr_path = os.path.join(self.lr_dir, hr_file)
            if os.path.exists(lr_path):
                image_pairs.append((lr_path, hr_path))
        return image_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lr_path, hr_path = self.image_pairs[idx]
        return load_and_transform_image(lr_path), load_and_transform_image(hr_path)


def extract_tensor_patches(tensor, patch_size=128):
    """
    Splits a tensor into chunks of specified size, padding if necessary, while preserving the channel dimension.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (B, C, H, W).
    patch_size (int): The size of each chunk along height and width.
    
    Returns:
    List[torch.Tensor]: List containing chunks of size (B, C, patch_size, patch_size).
    """
    B, C, H, W = tensor.shape
    
    # Calculate padding needed to make H and W multiples of patch_size
    pad_height = (patch_size - H % patch_size) % patch_size
    pad_width = (patch_size - W % patch_size) % patch_size
    
    # Pad the tensor
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height), mode="constant", value=0)
    
    # Calculate new height and width after padding
    new_H, new_W = padded_tensor.shape[2], padded_tensor.shape[3]
    
    # Creating a list to store chunks
    chunks = []
    
    # Split the tensor into chunks
    for i in range(0, new_H, patch_size):
        for j in range(0, new_W, patch_size):
            chunk = padded_tensor[:, :, i:i+patch_size, j:j+patch_size]
            chunks.append(chunk)
    
    return chunks


def recompile_tensor_patches(patches, original_width, original_height, patch_size=128):
    """
    Recompiles patches (as tensors) into a full image tensor, considering potential misalignments and
    the need for cropping patches on the image edges.
    
    Args:
    patches (list of torch.Tensor): List of tensor patches, each of shape (C, patch_size, patch_size).
    original_width (int): The width of the scaled original image.
    original_height (int): The height of the scaled original image.
    patch_size (int): The size of each patch (assumed square).
    
    Returns:
    torch.Tensor: The recompiled full image tensor of shape (C, original_height, original_width).
    """
    # Calculate how many patches fit horizontally and vertically
    num_patches_per_row = (original_width + patch_size - 1) // patch_size
    num_patches_per_col = (original_height + patch_size - 1) // patch_size

    # Create a new blank tensor for the full image
    C = patches[0].size(0)  # Assuming all patches have the same number of channels
    full_image = torch.zeros((C, original_height, original_width))
    
    # Iterate over each patch and place it in the correct position within the full_image tensor
    for idx, patch in enumerate(patches):
        x = (idx % num_patches_per_row) * patch_size
        y = (idx // num_patches_per_row) * patch_size
        
        # Determine the region in the full image where the patch should be placed
        x_end = min(x + patch_size, original_width)
        y_end = min(y + patch_size, original_height)
        
        # Calculate the region of the patch that fits within the bounds (crop if necessary)
        patch_width = x_end - x
        patch_height = y_end - y
        
        # Place the cropped patch into the full image
        full_image[:, y:y_end, x:x_end] = patch[:, :patch_height, :patch_width]

    return full_image


def extract_pil_patches(image, patch_size=128, every=16):
    patches = []
    rows = image.height // patch_size
    cols = image.width // patch_size
    for i in range(rows):
        for j in range(cols):
            if j % every == 0:
                top = i * patch_size
                left = j * patch_size
                right = left + patch_size
                bottom = top + patch_size
                patch = TF.crop(image, top, left, patch_size, patch_size)  # Use top, left, height, width
                patches.append(patch)
    return patches


def recompile_pil_patches(patches, original_width, original_height, patch_size=128):
    """
    Recompiles patches into a full image, considering potential misalignments and
    the need for cropping patches on the image edges.
    
    Args:
    patches (list of PIL.Image.Image): List of image patches.
    original_width (int): The width of the scaled original image.
    original_height (int): The height of the scaled original image.
    patch_size (int): The size of each patch (assumed square).
    
    Returns:
    PIL.Image.Image: The recompiled full image.
    """
    # Calculate how many patches fit horizontally and vertically
    num_patches_per_row = (original_width + patch_size - 1) // patch_size
    num_patches_per_col = (original_height + patch_size - 1) // patch_size
    
    # Create a new blank image with the same dimensions as the original scaled image
    full_image = Image.new("RGB", (original_width, original_height))
    
    # Paste each patch in the appropriate location, adjusting for edge cases
    for idx, patch in enumerate(patches):
        x = (idx % num_patches_per_row) * patch_size
        y = (idx // num_patches_per_row) * patch_size
        # Determine the exact region where the patch should be pasted
        box = (x, y, min(x + patch_size, original_width), min(y + patch_size, original_height))
        
        # Crop the patch if it extends beyond the expected box dimensions
        if patch.width > box[2] - box[0] or patch.height > box[3] - box[1]:
            patch = patch.crop((0, 0, box[2] - box[0], box[3] - box[1]))
        
        full_image.paste(patch, box)
    
    return full_image


def save_model(epoch, generator, discriminator, g_optimizer, d_optimizer, path="checkpoint.pth"):
    state = {
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict()
    }
    torch.save(state, path)


def load_model(path, generator, discriminator, g_optimizer, d_optimizer):
    checkpoint = torch.load(path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    return checkpoint["epoch"]


def reassemble_image_tensor(chunks, image_shape, patch_size=128):
    """Reassembles image chunks back to the original image shape."""
    _, h, w = image_shape
    assembled_image = torch.zeros((3, h, w))
    idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if idx < len(chunks):
                assembled_image[:, i:i + patch_size, j:j + patch_size] = chunks[idx].squeeze(0)
                idx += 1
    return assembled_image


def unnormalize(normalized_tensor):
    # NOTE: unnormalized_tensor * 255  # scale from [0, 1] to [0, 255]
    return (normalized_tensor + 1) / 2 # scale from -1, 1 to 0, 1


def tensor_to_pil(tensor, normalize=True):
    if normalize:
        return TF.to_pil_image(unnormalize(tensor))
    else:
        return TF.to_pil_image(tensor)


def pil_to_tensor(img, normalize=True):
    if normalize:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    return transform(img)


def load_and_transform_image(image_path):
    """Loads and transforms the input image."""
    with Image.open(image_path) as image:
        return pil_to_tensor(image.convert("RGB"))


def load_resize_and_transform_image(image_path, dims=(64, 64)):
    """Loads and transforms the input image."""
    transform = transforms.Compose([
        transforms.Resize(dims),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    with Image.open(image_path).convert("RGB") as image:
        return transform(image)


def save_lr_hr_grid(gen_imgs, real_imgs, filename, num_pairs=8, nrow=4, normalize=True):
    total_pairs = min(num_pairs, gen_imgs.size(0), real_imgs.size(0))
    indices = random.sample(range(gen_imgs.size(0)), total_pairs)

    images = []
    for idx in indices:
        # Normalize or unnormalize as needed
        gen_img = tensor_to_pil(gen_imgs[idx].cpu(), normalize=normalize)
        real_img = tensor_to_pil(real_imgs[idx].cpu(), normalize=normalize)
        images.append((gen_img, real_img))

    # Calculate dimensions for the grid
    img_width, img_height = images[0][0].size
    ncol = (total_pairs + nrow - 1) // nrow  # Calculate columns needed based on the number of rows
    total_width = img_width * 2 * ncol  # Two images (gen and real) side by side per column
    total_height = img_height * nrow

    # Create a new blank image for the grid
    new_image = Image.new("RGB", (total_width, total_height))

    # Place images in the grid
    for idx, (gen_img, real_img) in enumerate(images):
        x_offset = (idx % ncol) * img_width * 2  # Calculate the horizontal offset for the current image
        y_offset = (idx // ncol) * img_height   # Calculate the vertical offset for the current image
        new_image.paste(gen_img, (x_offset, y_offset))
        new_image.paste(real_img, (x_offset + img_width, y_offset))

    # Save the final composed image
    new_image.save(filename)


def save_pil_image(image, image_path):
    if image_path.endswith((".jpg", ".jpeg")):
        image.save(image_path, quality=100)
    else:
        image.save(image_path)


def extract_patches_from_dir(root, patch_size=128, every=16):
    """extract patches from source images in a directory to be used as hr training images"""
    output_dir = os.path.join(root, "hr_images")
    os.makedirs(output_dir, exist_ok=True)

    for img_idx, filename in enumerate(os.listdir(root)):
        if filename.lower().endswith((IMAGE_EXTENSIONS)):
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    original_image = img.convert("RGB")
            except Image.UnidentifiedImageError:
                print("skipping bad image file", file)
                continue
            
            patches = extract_pil_patches(original_image, patch_size=patch_size, every=every)

            for patch_idx, patch in enumerate(patches):
                patch.save(os.path.join(output_dir, f"{str(img_idx).zfill(4)}-{str(patch_idx).zfill(4)}.png"))
