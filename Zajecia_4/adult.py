import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_and_preprocess_data(path):
    """
    Loads and preprocesses data from a CSV file.

    Parameters:
    - path: Path to the dataset file.

    Returns:
    - X: Input features.
    - y: Target label (Income).
    - label_encoders: Dictionary of label encoders for categorical variables.
    - scaler: StandardScaler object used for numerical scaling.
    """
    columns = [
        'Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-Num',
        'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex',
        'Capital Gain', 'Capital Loss', 'Hours per week', 'Native Country', 'Income'
    ]
    data = pd.read_csv(path, header=None, names=columns, skipinitialspace=True)

    # Removing missing values
    data = data.replace('?', np.nan).dropna()

    # Encoding categorical variables
    label_encoders = {}
    for col in ['Workclass', 'Education', 'Marital Status', 'Occupation',
                'Relationship', 'Race', 'Sex', 'Native Country', 'Income']:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # Splitting features and target labels
    X = data.drop('Income', axis=1)
    y = data['Income']

    # Standardizing numerical variables
    scaler = StandardScaler()
    X[['Age', 'Fnlwgt', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']] = scaler.fit_transform(
        X[['Age', 'Fnlwgt', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']]
    )

    return X, y, label_encoders, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Trains Decision Tree and SVM models and evaluates their performance.

    Parameters:
    - X_train: Training feature data.
    - X_test: Testing feature data.
    - y_train: Training labels.
    - y_test: Testing labels.

    Returns:
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.
    - accuracies: List of model accuracies.
    """
    # Training models
    dt_model = DecisionTreeClassifier(random_state=42)
    svm_model = SVC(kernel='linear', random_state=42)

    dt_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Model evaluation
    y_pred_dt = dt_model.predict(X_test)
    y_pred_svm = svm_model.predict(X_test)

    print("Decision Tree:")
    print(f"Accuracy: {dt_model.score(X_test, y_test):.2f}")
    print(classification_report(y_test, y_pred_dt))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_dt, cmap=plt.cm.OrRd)
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()

    print("SVM:")
    print(f"Accuracy: {svm_model.score(X_test, y_test):.2f}")
    print(classification_report(y_test, y_pred_svm))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap=plt.cm.OrRd)
    plt.title("Confusion Matrix - SVM")
    plt.show()

    # Return accuracies
    accuracies = [dt_model.score(X_test, y_test), svm_model.score(X_test, y_test)]
    return dt_model, svm_model, accuracies


def plot_accuracy_comparison(models, accuracies):
    """
    Plots a bar chart comparing model accuracies.

    Parameters:
    - models: List of model names.
    - accuracies: List of model accuracies.
    """
    plt.bar(models, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison')

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc * 100:.2f}%", ha='center', fontsize=12, color='black')

    plt.show()


def predict_new_data(new_data, label_encoders, scaler, dt_model, svm_model):
    """
    Predicts class labels for new data using trained models.

    Parameters:
    - new_data: Dataframe containing new examples to classify.
    - label_encoders: Dictionary of label encoders used for categorical variables.
    - scaler: StandardScaler object used for numerical scaling.
    - dt_model: Trained Decision Tree model.
    - svm_model: Trained SVM model.

    Returns:
    - Predictions from Decision Tree and SVM models.
    """
    # Process new data
    for col in label_encoders:
        if col in new_data:
            new_data[col] = label_encoders[col].transform(new_data[col])

    new_data[['Age', 'Fnlwgt', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']] = scaler.transform(
        new_data[['Age', 'Fnlwgt', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']]
    )

    # Predictions
    prediction_dt = dt_model.predict(new_data)
    prediction_svm = svm_model.predict(new_data)

    return prediction_dt, prediction_svm


path = 'adult.data'
X, y, label_encoders, scaler = load_and_preprocess_data(path)

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and evaluate models
dt_model, svm_model, accuracies = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Compare model accuracies
plot_accuracy_comparison(['Decision Tree', 'SVM'], accuracies)

# Predictions for new data
new_data = pd.DataFrame([[
    35, 'Private', 120000, 'Bachelors', 13, 'Married-civ-spouse',
    'Exec-managerial', 'Husband', 'White', 'Male', 0, 0, 40, 'United-States'
]], columns=['Age', 'Workclass', 'Fnlwgt', 'Education', 'Education-Num',
            'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex',
            'Capital Gain', 'Capital Loss', 'Hours per week', 'Native Country'])

prediction_dt, prediction_svm = predict_new_data(new_data, label_encoders, scaler, dt_model, svm_model)

print("\nNew Data Predictions:")
print(f"Decision Tree: {prediction_dt[0]}")
print(f"SVM: {prediction_svm[0]}")
