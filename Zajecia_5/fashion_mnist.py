import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_and_preprocess_fashion_mnist():
    """
    Loads and preprocesses the Fashion-MNIST dataset.

    Returns:
    - X_train: Array of training feature data, normalized and reshaped.
    - X_test: Array of test feature data, normalized and reshaped.
    - y_train: Array of one-hot encoded training labels.
    - y_test: Array of one-hot encoded test labels.
    """
    # Load Fashion-MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize the data
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Reshape data to include channel dimension
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # One-hot encode the labels
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train, X_test, y_test):
    """
    Builds, compiles, and trains the CNN model on the Fashion-MNIST dataset.

    Parameters:
    - X_train: Array of training feature data.
    - y_train: Array of training labels.
    - X_test: Array of test feature data.
    - y_test: Array of test labels.

    Returns:
    - model: The trained CNN model.
    """
    # Define the model
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained CNN model on the test dataset.

    Parameters:
    - model: The trained CNN model.
    - X_test: Array of test feature data.
    - y_test: Array of test labels.

    Returns:
    - test_acc: Accuracy of the model on the test dataset.
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.2f}")

    return test_acc

def test_model(model, X_test, y_test):
    """
    Tests the trained CNN model and visualizes predictions on the test dataset.

    Parameters:
    - model: The trained CNN model.
    - X_test: Array of test feature data.
    - y_test: Array of test labels.
    """
    # Make predictions
    predictions = model.predict(X_test)

    # Convert predictions to class labels
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Visualize predictions
    indices = np.random.choice(np.arange(len(X_test)), size=10, replace=False)
    plt.figure(figsize=(12, 6))
    for i, index in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predicted_classes[index]}, True: {true_classes[index]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    """
    Plots the confusion matrix for the CNN model.

    Parameters:
    - model: The trained CNN model.
    - X_test: Array of test feature data.
    - y_test: Array of test labels.
    """
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Load and preprocess Fashion-MNIST data
X_train, X_test, y_train, y_test = load_and_preprocess_fashion_mnist()

# Build, train, and evaluate the model
model = build_and_train_model(X_train, y_train, X_test, y_test)
evaluate_model(model, X_test, y_test)

# Test the model with predictions
test_model(model, X_test, y_test)

# Plot confusion matrix
plot_confusion_matrix(model, X_test, y_test)
