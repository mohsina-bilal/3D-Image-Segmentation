import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Normalization functions
def normalise(image):
    np_img = np.clip(image, -1000., 800.).astype(np.float32)
    return np_img

def whitening(image):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image * 0.

def normalise_zero_one(image):
    image = image.astype(np.float32)
    minimum = np.min(image)
    maximum = np.max(image)
    if maximum > minimum:
        return (image - minimum) / (maximum - minimum)
    else:
        return image * 0.

def normalize_images_in_directory(image_directory, label_directory):
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_directory) if f.endswith('.nii.gz')])

    processed_images = []
    processed_labels = []

    for img_file, lbl_file in zip(image_files, label_files):
        img_path = os.path.join(image_directory, img_file)
        lbl_path = os.path.join(label_directory, lbl_file)

        img_sitk = sitk.ReadImage(img_path)
        lbl_sitk = sitk.ReadImage(lbl_path)

        image = sitk.GetArrayFromImage(img_sitk)
        label = sitk.GetArrayFromImage(lbl_sitk)

        # Apply different normalization techniques
        Normalize_minun100_to_800hu = normalise(image)
        Normalize_0mean_UnitVr = whitening(image)
        Normalize_0to1 = normalise_zero_one(image)

        # Visualize the normalizations
        visualize_normalizations(image, Normalize_minun100_to_800hu, Normalize_0mean_UnitVr, Normalize_0to1)

        # Append the processed images and labels to their respective lists
        processed_images.append(Normalize_0to1)  # Or whichever normalized version you prefer
        processed_labels.append(label)

    return processed_images, processed_labels

def visualize_normalizations(image, norm_min_max, norm_zero_mean, norm_zero_one):
    slice_idx = image.shape[0] // 2  # Take the middle slice for visualization
    f, axarr = plt.subplots(1, 4, figsize=(20, 5))
    f.suptitle('Different intensity normalisation methods on a CT image')

    img = axarr[0].imshow(np.squeeze(image[slice_idx, :, :]), cmap='gray', origin='lower')
    axarr[0].axis('off')
    axarr[0].set_title('Original image')
    f.colorbar(img, ax=axarr[0])

    img = axarr[1].imshow(np.squeeze(norm_min_max[slice_idx, :, :]), cmap='gray', origin='lower')
    axarr[1].axis('off')
    axarr[1].set_title('-1000 to 800')
    f.colorbar(img, ax=axarr[1])

    img = axarr[2].imshow(np.squeeze(norm_zero_mean[slice_idx, :, :]), cmap='gray', origin='lower')
    axarr[2].axis('off')
    axarr[2].set_title('Zero mean/unit stdev')
    f.colorbar(img, ax=axarr[2])

    img = axarr[3].imshow(np.squeeze(norm_zero_one[slice_idx, :, :]), cmap='gray', origin='lower')
    axarr[3].axis('off')
    axarr[3].set_title('[0,1] rescaling')
    f.colorbar(img, ax=axarr[3])

    f.subplots_adjust(wspace=0.05, hspace=0, top=0.8)
    plt.show()

# Call the function to normalize images
# processed_images, processed_labels = normalize_images_in_directory(image_dir, label_dir)

# Resample images
def resample_img(itk_image, out_spacing=[2.0, 2.0, 2.0], is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# Function to resize images
def resize_image_with_crop_or_pad(image, img_size=(128, 128, 128), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension to fit the specified size."""
    assert isinstance(image, (np.ndarray, np.generic))
    assert image.ndim == len(img_size), 'Image dimensionality must match length of img_size.'

    # Initialize slicer and padding
    rank = len(img_size)
    slicer = [slice(None)] * rank
    to_padding = [[0, 0] for _ in range(rank)]

    for i in range(rank):
        if image.shape[i] < img_size[i]:
            padding_needed = img_size[i] - image.shape[i]
            padding_before = padding_needed // 2
            padding_after = padding_needed - padding_before
            to_padding[i] = (padding_before, padding_after)
        else:
            crop_needed = image.shape[i] - img_size[i]
            start_crop = crop_needed // 2
            end_crop = start_crop + img_size[i]
            slicer[i] = slice(start_crop, end_crop)

    cropped_image = image[tuple(slicer)]
    padded_image = np.pad(cropped_image, to_padding, **kwargs)

    return padded_image

# Function to process and resize images from the resample function
def process_and_resize_images_in_directory(processed_images, processed_labels, output_dir, target_size=(128, 128, 128)):
    preprocessed_image_dir = os.path.join(output_dir, "preprocessed_images")
    preprocessed_label_dir = os.path.join(output_dir, "preprocessed_labels")

    os.makedirs(preprocessed_image_dir, exist_ok=True)
    os.makedirs(preprocessed_label_dir, exist_ok=True)

    for idx, (image, label) in enumerate(zip(processed_images, processed_labels)):
        # Apply resizing
        resized_image = resize_image_with_crop_or_pad(image, img_size=target_size, mode='constant', constant_values=-1024)
        resized_label = resize_image_with_crop_or_pad(label, img_size=target_size, mode='constant', constant_values=0)

        # Convert back to SimpleITK image format
        resized_img_sitk = sitk.GetImageFromArray(resized_image)
        resized_lbl_sitk = sitk.GetImageFromArray(resized_label)

        resized_img_sitk.SetSpacing([1.0, 1.0, 1.0])
        resized_lbl_sitk.SetSpacing([1.0, 1.0, 1.0])

        img_filename = os.path.join(preprocessed_image_dir, f'resized_image_{idx}.nii.gz')
        label_filename = os.path.join(preprocessed_label_dir, f'resized_label_{idx}.nii.gz')

        sitk.WriteImage(resized_img_sitk, img_filename)
        sitk.WriteImage(resized_lbl_sitk, label_filename)

        print(f"Processed and saved: {img_filename}, {label_filename}")

