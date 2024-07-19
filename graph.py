import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import PassiveAggressiveClassifier

# Generate some training data
X_train, y_train = make_classification(n_samples=100, n_features=61278, random_state=42)

# Train a PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Generate some dummy test data for prediction
X_test, _ = make_classification(n_samples=100, n_features=61278, random_state=42)

# Make predictions
predictions = model.predict(X_test)

# Assuming the accuracy is known and static
accuracy = 94.71

# For visualization purposes, let's assume we're interested in the first two features of the input
# and how they relate to the predicted class
X_test_2d = X_test[:, :2]  # Taking the first two features for plotting

# Plot the results
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=predictions, cmap='viridis', label='Predictions')
plt.title(f'Model Predictions (Accuracy: {accuracy}%)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(scatter, label='Predicted Class')
plt.legend()
plt.grid(True)
plt.show()
