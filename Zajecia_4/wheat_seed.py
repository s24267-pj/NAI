import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def load_and_preprocess_wheat_data(path):
    """
    Loads and preprocesses the Wheat Seed Dataset.

    Parameters:
    - path: Path to the dataset file.

    Returns:
    - X: Features of the dataset.
    - y: Target labels of the dataset.
    """
    columns = [
        'area', 'perimeter', 'compactness', 'length_of_kernel',
        'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove', 'class'
    ]

    try:
        data = pd.read_csv(path, header=None, names=columns, sep='\t')
    except Exception:
        data = pd.read_csv(path, header=None, names=columns, delim_whitespace=True)

    X = data.drop('class', axis=1)
    y = data['class']

    return X, y


def train_and_evaluate_wheat_models(X_train, X_test, y_train, y_test):
    """
    Trains Decision Tree and SVM models and evaluates their performance.

    Parameters:
    - X_train: Training feature data.
    - X_test: Test feature data.
    - y_train: Training labels.
    - y_test: Test labels.

    Returns:
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.
    - accuracies: List of model accuracies.
    """
    # Train models
    dt_model = DecisionTreeClassifier(random_state=42)
    svm_model = SVC(kernel='linear', random_state=42)

    dt_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Predictions
    y_pred_dt = dt_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)

    # Classification report for Decision Tree
    print("Classification Report - Decision Tree:")
    print(classification_report(y_test, y_pred_dt))

    # Classification report for SVM
    print("\nClassification Report - SVM:")
    print(classification_report(y_test, y_pred_svm))

    # Confusion Matrix for Decision Tree
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=np.unique(y_train)).plot(cmap=plt.cm.OrRd)
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    # Confusion Matrix for SVM
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=np.unique(y_train)).plot(cmap=plt.cm.OrRd)
    plt.title("Confusion Matrix - SVM")
    plt.show()

    # Model accuracies
    accuracy_dt = dt_model.score(X_test, y_test)
    accuracy_svm = svm_model.score(X_test, y_test)

    return dt_model, svm_model, [accuracy_svm, accuracy_dt]


def plot_accuracy_comparison(models, accuracies):
    """
    Creates a bar plot comparing model accuracies.

    Parameters:
    - models: List of model names.
    - accuracies: List of model accuracies.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies, color=['lightblue', 'orange'])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison of Models")
    plt.ylabel("Accuracy")

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', fontsize=12, color='black')

    plt.show()


def predict_wheat_class(new_data, dt_model, svm_model):
    """
    Predicts class labels for new data using trained models.

    Parameters:
    - new_data: Dataframe of new examples to classify.
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.

    Returns:
    - predictions_dt: Predictions from the Decision Tree model.
    - predictions_svm: Predictions from the SVM model.
    """
    predictions_dt = dt_model.predict(new_data)
    predictions_svm = svm_model.predict(new_data)

    return predictions_dt, predictions_svm


path = 'wheat_seed.data'
X, y = load_and_preprocess_wheat_data(path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate models
dt_model, svm_model, accuracies = train_and_evaluate_wheat_models(X_train, X_test, y_train, y_test)

# Compare model accuracies
plot_accuracy_comparison(['SVM', 'Decision Tree'], accuracies)

# Predictions for new data
new_data = pd.DataFrame([
    [14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956],
    [14.29, 14.09, 0.905, 5.291, 3.337, 2.699, 4.825]
], columns=['area', 'perimeter', 'compactness', 'length_of_kernel',
            'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove'])

print("\nNew Data:")
print(new_data)

predictions_dt, predictions_svm = predict_wheat_class(new_data, dt_model, svm_model)

print("\nDecision Tree Predictions:")
for i, pred in enumerate(predictions_dt):
    print(f"Example {i + 1}: Class {pred}")

print("\nSVM Predictions:")
for i, pred in enumerate(predictions_svm):
    print(f"Example {i + 1}: Class {pred}")
