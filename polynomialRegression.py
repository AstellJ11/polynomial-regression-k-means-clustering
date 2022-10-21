import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection as ms


# Method to create polynomial data matrix
def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1, degree + 1):
        X = np.column_stack((X, x ** i))  # Feature expansion upto max degree
    return X


# Find weights for each degree of polynomial
def pol_regression(features_train, y_train, degree):
    if degree <= 0:
        return np.mean(y_train) # For the 0th degree, return the mean
    else:
        X = getPolynomialDataMatrix(features_train, degree)

        XX = X.transpose().dot(X)
        parameters = np.linalg.solve(XX, X.transpose().dot(y_train))  # Weight for each degree value
        return parameters


# Plotting the polynomial functions
def plotPolynomial(weights, degree):
    if degree <= 0:
        plt.plot(npRange,
                 getPolynomialDataMatrix(npRange, degree) * weights)  # For the 0th degree, the y-intercept line
    else:
        plt.plot(npRange, getPolynomialDataMatrix(npRange, degree).dot(weights))  # Plot using custom range


# Calculate RMSE value for each degree of polynomial
def eval_pol_regression(parameters, x, y, degree):
    M = getPolynomialDataMatrix(x, degree)
    if degree <= 0:
        rmse = np.sqrt(((M * (parameters) - y) ** 2).mean())
    else:
        rmse = np.sqrt(((M.dot(parameters) - y) ** 2).mean())
    return rmse


data_train = pd.read_csv('Task1 - dataset - pol_regression.csv')  # Read in the data

sorted_data = data_train.sort_values('x')  # Sorts the data in terms of x values from smallest to largest

# Get the training data as numpy arrays
x_train = sorted_data['x']
y_train = sorted_data['y']

npRange = np.linspace(-5, 5)  # Define the custom range

axes = plt.gca()
axes.set_ylim([-250, 100])  # Set the y-axis to a custom limit

# Find the weights ready for plotting
weight_0 = pol_regression(x_train, y_train, 0)
weight_1 = pol_regression(x_train, y_train, 1)
weight_2 = pol_regression(x_train, y_train, 2)
weight_3 = pol_regression(x_train, y_train, 3)
weight_6 = pol_regression(x_train, y_train, 6)
weight_10 = pol_regression(x_train, y_train, 10)

# Plot the lines with degree 0, 1, 2, 3, 6, 10 alongside training points
plt.plot(x_train, y_train, 'bo')  # Plot the original training points
plotPolynomial(weight_0, 0)
plotPolynomial(weight_1, 1)
plotPolynomial(weight_2, 2)
plotPolynomial(weight_3, 3)
plotPolynomial(weight_6, 6)
plotPolynomial(weight_10, 10)

# Styling and saving figure
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(('Training Points', '$x^0$', '$x$', '$x^2$', '$x^3$', '$x^6$', '$x^{10}$'),
           loc='lower right')
plt.savefig('polynomialRegression.png')

# New variable defined before splitting
x_split = data_train['x']
y_split = data_train['y']

x_train, x_test, y_train, y_test = ms.train_test_split(x_split, y_split, test_size=0.3)  # Split the data 70:30

# RMSE values for training data
train_rmse_0 = eval_pol_regression(weight_0, x_train, y_train, 0)
train_rmse_1 = eval_pol_regression(weight_1, x_train, y_train, 1)
train_rmse_2 = eval_pol_regression(weight_2, x_train, y_train, 2)
train_rmse_3 = eval_pol_regression(weight_3, x_train, y_train, 3)
train_rmse_6 = eval_pol_regression(weight_6, x_train, y_train, 6)
train_rmse_10 = eval_pol_regression(weight_10, x_train, y_train, 10)

# RMSE values for training data
test_rmse_0 = eval_pol_regression(weight_0, x_test, y_test, 0)
test_rmse_1 = eval_pol_regression(weight_1, x_test, y_test, 1)
test_rmse_2 = eval_pol_regression(weight_2, x_test, y_test, 2)
test_rmse_3 = eval_pol_regression(weight_3, x_test, y_test, 3)
test_rmse_6 = eval_pol_regression(weight_6, x_test, y_test, 6)
test_rmse_10 = eval_pol_regression(weight_10, x_test, y_test, 10)

# Styling and saving figure
plt.figure()
plt.plot([0, 1, 2, 3, 6, 10], [train_rmse_0, train_rmse_1, train_rmse_2, train_rmse_3, train_rmse_6, train_rmse_10])
plt.plot([0, 1, 2, 3, 6, 10], [test_rmse_0, test_rmse_1, test_rmse_2, test_rmse_3, test_rmse_6, test_rmse_10])
plt.xlabel("Degree of Polynomial Regression")
plt.ylabel("Root Mean Square Error")
plt.legend(('RMSE on Training Data', 'RMSE on Testing Data'), loc='upper right')
plt.savefig('polynomialEvaluation.png')

plt.show()
