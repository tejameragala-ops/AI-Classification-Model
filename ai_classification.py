#ai_classification.py
# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['target'] = iris.target

# Show dataset
print("===== DATASET PREVIEW =====")
print(df.head())

# Features and labels
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create AI model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL RESULT =====")
print("Accuracy:", accuracy * 100, "%")

# Test sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\n===== PREDICTION =====")
print("Predicted Flower:", iris.target_names[prediction][0])