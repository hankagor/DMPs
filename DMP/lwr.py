import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

def locally_weighted_regression(X, y, tau, n_neighbors):
    def kernel(distances):
        return np.exp(-0.5 * (distances / tau) ** 2)

    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=kernel)
    knn.fit(X, y)
    return knn

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Set tau and number of neighbors
tau = 0.1
n_neighbors = 5  # Number of neighbors for LWR

# Create and fit the locally weighted regression model
lwr_model = locally_weighted_regression(X, y, tau, n_neighbors)

# Predict values
X_new = np.linspace(0, 2, 100).reshape(-1, 1)
y_lwr_predict = lwr_model.predict(X_new)

# Plot the results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_new, y_lwr_predict, color='green', linewidth=2, label=f'Locally Weighted Regression ({n_neighbors} neighbors)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
