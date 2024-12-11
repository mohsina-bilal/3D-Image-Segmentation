from src.data_loader import load_and_visualize
import os
from src.preprocess import (
    normalize_images_in_directory,
    resample_img,
    process_and_resize_images_in_directory,
    load_preprocessed_data,
    process_labels_for_desired_classes
)
from src.data_split import split_data_into_train_val_test
from src.model import vnet, compile_model, load_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.save_video import render_video_from_predictions

# Directories where the images and labels are stored
image_dir = 'C:/Users/mohsi/PycharmProjects/3D_image_segmentation/pythonProject/data/raw/FLARE22Train/images'
label_dir = 'C:/Users/mohsi/PycharmProjects/3D_image_segmentation/pythonProject/data/raw/FLARE22Train/labels'
output_dir = 'C:/Users/mohsi/PycharmProjects/3D_image_segmentation/pythonProject/data/processed'  # Update with your actual directory to save processed data
split_data_output_dir = os.path.join(output_dir, 'split_data_output')  # Output directory for train, val, test splits

def main():
    # Load and visualize images and masks
    images, masks = load_and_visualize(image_dir, label_dir)

    # Step 1: Normalize the images and labels
    print("Step 1: Normalizing images and labels...")
    processed_images, processed_labels = normalize_images_in_directory(image_dir, label_dir)

    # Step 2: Resample the images and labels
    print("Step 2: Resampling images and labels...")
    resampled_images, resampled_labels = resample_img(processed_images, processed_labels, out_spacing=[1, 1, 1], save_dir=output_dir)

    # Step 3: Resize the resampled images and labels
    print("Step 3: Resizing images and labels...")
    process_and_resize_images_in_directory(resampled_images, resampled_labels, output_dir, target_size=(128, 128, 128))

    # Step 4: Process and one-hot encode the labels
    print("Step 4: One-hot encoding the labels...")
    desired_classes = {1: 0, 2: 1, 3: 2, 13: 3}  # Example of desired classes
    num_classes = len(desired_classes)
    preprocessed_label_dir = os.path.join(output_dir, "preprocessed_labels")
    one_hot_label_output_dir = os.path.join(output_dir, "one_hot_encoded_labels")

    process_labels_for_desired_classes(preprocessed_label_dir, one_hot_label_output_dir, desired_classes, num_classes)

    # Step 5: Loading the preprocessed data (optional)
    print("Step 5: Loading preprocessed images and labels for verification...")
    preprocessed_image_dir = os.path.join(output_dir, "preprocessed_images")
    preprocessed_label_dir = os.path.join(output_dir, "preprocessed_labels")

    # Load preprocessed data to check the shape
    preprocessed_images, preprocessed_labels = load_preprocessed_data(preprocessed_image_dir, preprocessed_label_dir)

    # Check shapes
    print(f"Loaded preprocessed image shape: {preprocessed_images[0].shape}")
    print(f"Loaded preprocessed label shape: {preprocessed_labels[0].shape}")

    # Optionally, print all image and label shapes
    for i in range(len(preprocessed_images)):
        print(f"Image {i} shape: {preprocessed_images[i].shape}")
        print(f"Label {i} shape: {preprocessed_labels[i].shape}")

    # Step 6: Split the data into train, validation, and test sets
    print("Step 6: Splitting data into train, validation, and test sets...")
    split_data_into_train_val_test(
        preprocessed_image_dir=os.path.join(output_dir, 'preprocessed_images_with_channel'),
        preprocessed_label_dir=os.path.join(output_dir, 'one_hot_encoded_labels'),
        output_dir=split_data_output_dir,
        test_size=0.2,
        val_size=0.1
    )

    # Step 7: Train the model using the training and validation data
    print("Step 7: Training the VNet model...")
    # Set directories for train, validation, and test data
    train_dir = os.path.join(split_data_output_dir, 'train')
    val_dir = os.path.join(split_data_output_dir, 'validation')
    test_dir = os.path.join(split_data_output_dir, 'test')

    # Define where to save the model
    output_model_dir = 'C:/Users/mohsi/PycharmProjects/3D_image_segmentation/pythonProject/model_output'

    # Train the model (this function will call model.fit)
    train_model(train_dir, val_dir, test_dir, output_model_dir, epochs=20, batch_size=4)

    # Load the test data
    X_test, y_test = load_data(os.path.join(test_dir, 'images'), os.path.join(test_dir, 'labels'))

    # Assuming model is saved under this path
    model_path = 'C:/Users/mohsi/PycharmProjects/3D_image_segmentation/pythonProject/model_output/vnet_model.h5'

    # Evaluate the model
    test_loss, dice_coefficients = evaluate_model(model_path, X_test, y_test)

    y_pred = model.predict(X_test[:1])[0]  # Predict on the first test sample

    # Call the function to render and save the video
    render_video_from_predictions(y_pred, X_test[0, ..., 0], output_dir='output/saved_video',
                                  output_file='segmentation_video.mp4')
if __name__ == "__main__":
    main()
