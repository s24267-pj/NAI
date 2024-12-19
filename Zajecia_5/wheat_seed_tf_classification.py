import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

def load_and_preprocess_wheat_data(path):
    """
    Loads and preprocesses the Wheat Seed Dataset.

    Parameters:
    - path: str, Path to the dataset file.

    Returns:
    - X: DataFrame, Features of the dataset.
    - y: Series, Target labels of the dataset.
    """
    columns = [
        'area', 'perimeter', 'compactness', 'length_of_kernel',
        'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove', 'class'
    ]

    try:
        data = pd.read_csv(path, header=None, names=columns, sep='\\s+')
    except Exception:
        data = pd.read_csv(path, header=None, names=columns, delim_whitespace=True)

    X = data.drop('class', axis=1)
    y = data['class'] - 1  # Ensure labels are zero-indexed

    return X, y

def train_and_evaluate_wheat_models(X_train, X_test, y_train, y_test):
    """
    Trains Decision Tree, SVM, and TensorFlow models and evaluates their performance.

    Parameters:
    - X_train: DataFrame, Training feature data.
    - X_test: DataFrame, Test feature data.
    - y_train: Series, Training labels.
    - y_test: Series, Test labels.

    Returns:
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.
    - tf_model: Trained TensorFlow model.
    - accuracies: List of float, Accuracies of the models.
    """
    # Train models
    dt_model = DecisionTreeClassifier(random_state=42)
    svm_model = SVC(kernel='linear', random_state=42)

    dt_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # TensorFlow model
    tf_model = Sequential([
        InputLayer(input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(8, activation='relu'),
        Dense(3, activation='softmax')
    ])
    tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tf_model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    # Predictions
    y_pred_dt = dt_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)
    y_pred_tf = tf_model.predict(X_test).argmax(axis=1)

    # Classification report for Decision Tree
    print("Classification Report - Decision Tree:")
    print(classification_report(y_test, y_pred_dt))

    # Classification report for SVM
    print("\nClassification Report - SVM:")
    print(classification_report(y_test, y_pred_svm))

    # Classification report for TensorFlow
    print("\nClassification Report - TensorFlow:")
    print(classification_report(y_test, y_pred_tf))

    # Model accuracies
    accuracy_dt = dt_model.score(X_test, y_test)
    accuracy_svm = svm_model.score(X_test, y_test)
    accuracy_tf = tf_model.evaluate(X_test, y_test, verbose=0)[1]

    return dt_model, svm_model, tf_model, [accuracy_svm, accuracy_dt, accuracy_tf]

def plot_accuracy_comparison(models, accuracies):
    """
    Creates a bar plot comparing model accuracies.

    Parameters:
    - models: List of str, Names of the models.
    - accuracies: List of float, Accuracies of the models.
    """
    plt.figure(figsize=(8, 6))
    plt.bar(models, accuracies, color=['lightblue', 'orange', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison of Models")
    plt.ylabel("Accuracy")

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', fontsize=12, color='black')


    plt.show()

def predict_wheat_class(new_data, dt_model, svm_model, tf_model):
    """
    Predicts class labels for new data using trained models.

    Parameters:
    - new_data: DataFrame, New examples to classify.
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.
    - tf_model: Trained TensorFlow model.

    Returns:
    - predictions_dt: Array, Predictions from the Decision Tree model.
    - predictions_svm: Array, Predictions from the SVM model.
    - predictions_tf: Array, Predictions from the TensorFlow model.
    """
    predictions_dt = dt_model.predict(new_data)
    predictions_svm = svm_model.predict(new_data)
    predictions_tf = tf_model.predict(new_data).argmax(axis=1)

    return predictions_dt, predictions_svm, predictions_tf

path = 'wheat_seed.data'
X, y = load_and_preprocess_wheat_data(path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate models
dt_model, svm_model, tf_model, accuracies = train_and_evaluate_wheat_models(X_train, X_test, y_train, y_test)

# Compare model accuracies
plot_accuracy_comparison(['SVM', 'Decision Tree', 'TensorFlow'], accuracies)

# Predictions for new data
new_data = pd.DataFrame([
    [14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956],
    [14.29, 14.09, 0.905, 5.291, 3.337, 2.699, 4.825]
], columns=['area', 'perimeter', 'compactness', 'length_of_kernel',
            'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove'])

print("\nNew Data:")
print(new_data)

predictions_dt, predictions_svm, predictions_tf = predict_wheat_class(new_data, dt_model, svm_model, tf_model)

print("\nDecision Tree Predictions:")
for i, pred in enumerate(predictions_dt):
    print(f"Example {i + 1}: Class {pred}")

print("\nSVM Predictions:")
for i, pred in enumerate(predictions_svm):
    print(f"Example {i + 1}: Class {pred}")

print("\nTensorFlow Predictions:")
for i, pred in enumerate(predictions_tf):
    print(f"Example {i + 1}: Class {pred}")
