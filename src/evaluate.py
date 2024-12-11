import numpy as np
from tensorflow.keras.models import load_model

def evaluate_model(model_path, X_test, y_test):
    # Load the pre-trained model
    model = load_model(model_path, custom_objects={
        'dice_loss': dice_loss,  # Ensure to include your custom loss if used during training
        'dice_coefficient_class0': dice_coefficient_class0,
        'dice_coefficient_class1': dice_coefficient_class1,
        'dice_coefficient_class2': dice_coefficient_class2,
        'dice_coefficient_class3': dice_coefficient_class3
    })

    # Evaluate the model on the test set
    test_metrics = model.evaluate(X_test, y_test)

    # Extract the loss and dice coefficients for each class
    test_loss = test_metrics[0]  # The first value is always the loss
    dice_coefficients = test_metrics[1:]  # Remaining values are dice coefficients for each class

    # Print the test loss
    print(f"Test Loss: {test_loss}")

    # Print the dice coefficient for each class
    for i, dice in enumerate(dice_coefficients):
        print(f"Dice Coefficient for Class {i}: {dice}")

    return test_loss, dice_coefficients
