import os
import shutil
from sklearn.model_selection import train_test_split


def split_data_into_train_val_test(preprocessed_image_dir, preprocessed_label_dir, output_dir, test_size=0.2,
                                   val_size=0.1):
    """
    Split preprocessed images and labels into train, validation, and test sets.

    Parameters:
    - preprocessed_image_dir: Path to the directory containing preprocessed images.
    - preprocessed_label_dir: Path to the directory containing one-hot encoded labels.
    - output_dir: Path to save the split train, validation, and test data.
    - test_size: Fraction of data to be reserved for testing.
    - val_size: Fraction of remaining data to be reserved for validation after splitting test data.
    """

    # Create directories for train, validation, and test sets
    train_image_dir = os.path.join(output_dir, 'train', 'images')
    train_label_dir = os.path.join(output_dir, 'train', 'labels')
    val_image_dir = os.path.join(output_dir, 'validation', 'images')
    val_label_dir = os.path.join(output_dir, 'validation', 'labels')
    test_image_dir = os.path.join(output_dir, 'test', 'images')
    test_label_dir = os.path.join(output_dir, 'test', 'labels')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # Load image and label file paths
    image_files = sorted(
        [os.path.join(preprocessed_image_dir, f) for f in os.listdir(preprocessed_image_dir) if f.endswith('.nii.gz')])
    label_files = sorted(
        [os.path.join(preprocessed_label_dir, f) for f in os.listdir(preprocessed_label_dir) if f.endswith('.nii.gz')])

    # Ensure the number of images and labels match
    assert len(image_files) == len(label_files), "Mismatch between the number of images and labels."

    # Split the data into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(image_files, label_files, test_size=test_size,
                                                                random_state=42)

    # Further split the train+val set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1 - test_size),
                                                      random_state=42)

    # Function to save files to the respective directories
    def save_files(file_list, label_list, image_dir, label_dir):
        for img_file, lbl_file in zip(file_list, label_list):
            # Get the base filename
            img_filename = os.path.basename(img_file)
            lbl_filename = os.path.basename(lbl_file)

            # Copy the files to the target directories
            shutil.copy(img_file, os.path.join(image_dir, img_filename))
            shutil.copy(lbl_file, os.path.join(label_dir, lbl_filename))

    # Save train, validation, and test sets
    save_files(X_train, y_train, train_image_dir, train_label_dir)
    save_files(X_val, y_val, val_image_dir, val_label_dir)
    save_files(X_test, y_test, test_image_dir, test_label_dir)

    print("Train, validation, and test sets have been split and saved in the respective directories.")