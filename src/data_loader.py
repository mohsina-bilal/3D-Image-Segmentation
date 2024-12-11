# src/data_loader.py

import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_directory(image_dir, label_dir):
    """
    Load CT and mask images from the specified directories.

    Args:
    - image_dir (str): Directory containing CT images (.nii.gz files).
    - label_dir (str): Directory containing mask labels (.nii.gz files).

    Returns:
    - images (list of np.array): List of 3D CT image arrays.
    - masks (list of np.array): List of 3D mask arrays.
    """
    # Get all .nii.gz files in the image and label directories
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

    images = []
    masks = []

    for img_file, label_file in zip(image_files, label_files):
        # Construct full file paths
        ct_path = os.path.join(image_dir, img_file)
        ct_label_path = os.path.join(label_dir, label_file)

        # CT: Load and convert to array
        img_sitk = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(img_sitk)

        # Mask: Load and convert to array
        mask_sitk = sitk.ReadImage(ct_label_path, sitk.sitkInt32)
        mask = sitk.GetArrayFromImage(mask_sitk)

        # Append to list
        images.append(image)
        masks.append(mask)

        # Display shapes for debug purposes
        print(f'Processing {img_file} and {label_file}')
        print(f'CT Shape={image.shape}')
        print(f'CT Mask Shape={mask.shape}')

    return images, masks

def visualize_image_and_mask(image, mask):
    """
    Visualize a single slice of the image and mask.

    Args:
    - image (np.array): 3D CT image array.
    - mask (np.array): 3D mask array.
    """
    slice_idx = image.shape[0] // 2  # Choose the middle slice for visualization
    f, axarr = plt.subplots(1, 3, figsize=(15, 15))

    axarr[0].imshow(np.squeeze(image[slice_idx, :, :]), cmap='gray', origin='lower')
    axarr[0].set_ylabel('Axial View', fontsize=14)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[0].set_title('CT', fontsize=14)

    axarr[1].imshow(np.squeeze(mask[slice_idx, :, :]), cmap='jet', origin='lower')
    axarr[1].axis('off')
    axarr[1].set_title('Mask', fontsize=14)

    axarr[2].imshow(np.squeeze(image[slice_idx, :, :]), cmap='gray', alpha=1, origin='lower')
    axarr[2].imshow(np.squeeze(mask[slice_idx, :, :]), cmap='jet', alpha=0.5, origin='lower')
    axarr[2].axis('off')
    axarr[2].set_title('Overlay', fontsize=14)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

# Function to load, visualize, and return images
def load_and_visualize(image_dir, label_dir):
    """
    Load images and visualize a slice from each.

    Args:
    - image_dir (str): Directory containing CT images.
    - label_dir (str): Directory containing mask labels.

    Returns:
    - images (list): List of 3D CT images.
    - masks (list): List of 3D mask images.
    """
    images, masks = load_images_from_directory(image_dir, label_dir)

    # Visualize the first image and mask
    if images and masks:
        visualize_image_and_mask(images[0], masks[0])

    return images, masks


