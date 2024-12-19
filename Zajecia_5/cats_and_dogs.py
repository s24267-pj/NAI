import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt


def load_and_preprocess_data(base_dir):
    """
    Loads and preprocesses the Cats and Dogs dataset from the specified directory.

    Parameters:
    - base_dir: str, Path to the dataset directory.

    Returns:
    - train_generator: ImageDataGenerator, Training data generator.
    - validation_generator: ImageDataGenerator, Validation data generator.
    """
    # Normalize the data and split into training and validation sets
    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator


def build_model():
    """
    Builds a Convolutional Neural Network (CNN) model for binary classification.

    Returns:
    - model: Sequential, Compiled CNN model.
    """
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, train_generator, validation_generator):
    """
    Trains the CNN model on the provided training data.

    Parameters:
    - model: Sequential, Compiled CNN model.
    - train_generator: ImageDataGenerator, Training data generator.
    - validation_generator: ImageDataGenerator, Validation data generator.
    - epochs: int, Number of epochs to train the model.

    Returns:
    - history: History, Training history of the model.
    """
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size
    )

    return history


def visualize_predictions(model, validation_generator):
    """
    Visualizes predictions on a batch of validation data.

    Parameters:
    - model: Sequential, The trained CNN model.
    - validation_generator: ImageDataGenerator, Validation data generator.
    """
    # Get a batch of validation data
    X_val, y_val = next(validation_generator)
    predictions = model.predict(X_val)
    predicted_classes = (predictions > 0.5).astype(int)

    # Visualize the results
    plt.figure(figsize=(12, 12))
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_val[i])
        plt.title(f"Pred: {predicted_classes[i][0]}, True: {int(y_val[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('cats_dogs.png')
    plt.show()


# Main function
def main(base_dir):
    """
    Main function to load data, build, train, and evaluate the model.

    Parameters:
    - base_dir: str, Path to the dataset directory.
    """
    # Load and preprocess data
    train_generator, validation_generator = load_and_preprocess_data(base_dir)

    # Build the model
    model = build_model()

    # Train the model
    train_model(model, train_generator, validation_generator)

    # Visualize predictions
    visualize_predictions(model, validation_generator)


base_dir = 'PetImages'

main(base_dir)
