import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
boston_dataset = load_boston()

# Create a DataFrame
boston = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
boston['target'] = boston_dataset.target

# Display the first few rows and the dataset description
print(boston.head())
print(boston.describe())
# Selecting one or more features
X = boston[['RM', 'LSTAT']]  # Example with two features: average number of rooms per dwelling (RM) and % lower status of the population (LSTAT)
y = boston['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create a linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2 Score): {r2}')
plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Actual vs. Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid()
plt.show()
# Example prediction with new data
new_data = [[6.5, 15]]  # Example features: RM = 6.5, LSTAT = 15
predicted_price = model.predict(new_data)
print(f'Predicted Price: ${predicted_price[0]:.2f}')
