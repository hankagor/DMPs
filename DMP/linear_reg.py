import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and fit the linear regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# Predict values
X_new = np.linspace(0, 2, 100).reshape(-1, 1)
y_linear_predict = linear_model.predict(X_new)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_linear_predict, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Print the model parameters
print("Linear Regression Intercept:", linear_model.intercept_)
print("Linear Regression Coefficient:", linear_model.coef_)
