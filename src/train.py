import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import vnet, compile_model, load_data

def train_model(train_dir, val_dir, test_dir, output_model_dir, epochs=10, batch_size=2):
    """
    Trains the VNet model with the given data.

    Parameters:
    - train_dir: Directory containing the training images and labels
    - val_dir: Directory containing the validation images and labels
    - test_dir: Directory containing the test images and labels
    - output_model_dir: Directory where the trained model will be saved
    - epochs: Number of training epochs (default=10)
    - batch_size: Batch size for training (default=2)
    """
    # Load the data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(train_dir, val_dir, test_dir)

    # Initialize the VNet model
    model = vnet(input_shape=(128, 128, 128, 1), num_classes=4)

    # Compile the model
    compile_model(model)

    # Create output directory if not exists
    os.makedirs(output_model_dir, exist_ok=True)

    # Define model checkpoint callback to save the best model during training
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(output_model_dir, 'vnet_best_model.h5'),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )

    # Define early stopping callback to stop training when the model stops improving
    early_stopping_cb = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint_cb, early_stopping_cb]
    )

    # Save the final trained model
    model.save(os.path.join(output_model_dir, 'vnet_final_model.h5'))

    print("Training completed and model saved.")

    return history, model


