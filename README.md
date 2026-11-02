# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required Python libraries and load the Iris dataset.



Separate the input features and target species.



Split the dataset into training and testing data.



Train the SGD Classifier model using the training dataset.



Predict the Iris flower species and calculate the model accuracy.

## Program:
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()

# Input features
X = iris.data

# Target labels
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SGD Classifier
model = SGDClassifier(
    loss='log_loss',
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# Predict a new Iris flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nFlower Measurements:")
print("Sepal Length:", new_flower[0][0])
print("Sepal Width:", new_flower[0][1])
print("Petal Length:", new_flower[0][2])
print("Petal Width:", new_flower[0][3])

print("\nPredicted Iris Species:",
      iris.target_names[prediction[0]])

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))
```

## Output:
```
Model Accuracy: 90.0 %

Flower Measurements:
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2

Predicted Iris Species: setosa

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.75      1.00      0.86         9
   virginica       1.00      0.73      0.84        11

    accuracy                           0.90        30
   macro avg       0.92      0.91      0.90        30
weighted avg       0.93      0.90      0.90        30
```

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
