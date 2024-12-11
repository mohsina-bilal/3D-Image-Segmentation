import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import os
import SimpleITK as sitk
import numpy as np

# VNet model definition
def vnet(input_shape=(128, 128, 128, 1), num_classes=4):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(inputs)
    conv1 = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(conv1)
    down1 = layers.MaxPooling3D((2, 2, 2))(conv1)

    conv2 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(down1)
    conv2 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(conv2)
    down2 = layers.MaxPooling3D((2, 2, 2))(conv2)

    conv3 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(down2)
    conv3 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(conv3)
    down3 = layers.MaxPooling3D((2, 2, 2))(conv3)

    # Bottleneck
    bottleneck = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')(down3)
    bottleneck = layers.Conv3D(128, (3, 3, 3), padding='same', activation='relu')(bottleneck)

    # Decoder
    up3 = layers.UpSampling3D((2, 2, 2))(bottleneck)
    up3 = layers.concatenate([up3, conv3])
    conv4 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(up3)
    conv4 = layers.Conv3D(64, (3, 3, 3), padding='same', activation='relu')(conv4)

    up2 = layers.UpSampling3D((2, 2, 2))(conv4)
    up2 = layers.concatenate([up2, conv2])
    conv5 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(up2)
    conv5 = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(conv5)

    up1 = layers.UpSampling3D((2, 2, 2))(conv5)
    up1 = layers.concatenate([up1, conv1])
    conv6 = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(up1)
    conv6 = layers.Conv3D(16, (3, 3, 3), padding='same', activation='relu')(conv6)

    # Output
    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(conv6)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Dice coefficient and loss functions
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def dice_coefficient_per_class(y_true, y_pred, smooth=1):
    y_pred = tf.nn.softmax(y_pred, axis=-1)
    y_true_f = K.reshape(y_true, (-1, y_true.shape[-1]))
    y_pred_f = K.reshape(y_pred, (-1, y_pred.shape[-1]))

    intersection = K.sum(y_true_f * y_pred_f, axis=0)
    denominator = K.sum(y_true_f + y_pred_f, axis=0)
    dice_per_class = (2. * intersection + smooth) / (denominator + smooth)

    return dice_per_class

# Functions to calculate Dice coefficient for each class
def dice_coefficient_class0(y_true, y_pred):
    return dice_coefficient_per_class(y_true, y_pred)[0]

def dice_coefficient_class1(y_true, y_pred):
    return dice_coefficient_per_class(y_true, y_pred)[1]

def dice_coefficient_class2(y_true, y_pred):
    return dice_coefficient_per_class(y_true, y_pred)[2]

def dice_coefficient_class3(y_true, y_pred):
    return dice_coefficient_per_class(y_true, y_pred)[3]

# Compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss=dice_loss,
                  metrics=[dice_coefficient_class0,
                           dice_coefficient_class1,
                           dice_coefficient_class2,
                           dice_coefficient_class3])
    model.summary()

# Function to load train, validation, and test data
def load_images_and_labels(image_dir, label_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

    images = []
    labels = []

    for img_file, lbl_file in zip(image_files, label_files):
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, lbl_file)

        img_sitk = sitk.ReadImage(img_path)
        lbl_sitk = sitk.ReadImage(lbl_path)

        image = sitk.GetArrayFromImage(img_sitk)
        label = sitk.GetArrayFromImage(lbl_sitk)

        image = np.expand_dims(image, axis=-1)  # Add channel dimension

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

# Load data for training, validation, and testing
def load_data(train_dir, val_dir, test_dir):
    X_train, y_train = load_images_and_labels(
        os.path.join(train_dir, 'images'),
        os.path.join(train_dir, 'labels')
    )

    X_val, y_val = load_images_and_labels(
        os.path.join(val_dir, 'images'),
        os.path.join(val_dir, 'labels')
    )

    X_test, y_test = load_images_and_labels(
        os.path.join(test_dir, 'images'),
        os.path.join(test_dir, 'labels')
    )

    print(f"Train data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
