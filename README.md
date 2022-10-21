
# Machine Learning - Polynomial Regression Model & K-Means Clustering Algorithm (CMP3751M_Assessment_01)


## Dataset Information

For the polynomial regression model, the 'Task1-dataset-pol_regression.csv' file contains a simple 1-dimensional dataset with inputs x (1 input dimension) and output y (1 output dimension). The dataset has been generated from an unknown polynomial function plus a noise term.

For the K-means clustering algorithm, the 'Task2-dataset-dog_breeds.csv' file contains the data of multiple dog breeds with four unique features, which are height, tail length, leg length, and nose circumference.
   
   
## Implementation of Polynomial Regression
The first step in implementing a polynomial regression algorithm is to set up the data in a matrix in the required form. Firstly, a blank array matching the size of the input array is creating containing all ones. These are then re-written in the following for loop where the feature expansion is calculated. This can be repeated up to the max degree specified when called.

Once the feature expansion function has been created, the primary polynomial regression function can be implemented. The function ‘pol_regression’ has three arguments required; these are features_train, y_train, and degree. When called, feature_train corresponds to the x column of the data, y_train is the y column, and degree will be the degree of polynomial regression being called. The function begins with an if statement checking if the degree value is zero. This check is performed in almost every function as the case for degree equalling zero will be different. In this case, when the degree is zero, the y parameter is simply the mean value of the y column.

For every other case, the feature expansion of the x column is calculated by calling the ‘getPolynomialDataMatrix’ function and assigning the result to the variable X. Next, a new variable, XX, is defined, which is equal to the dot product of the new matrix (X) and is transposed afterward. Finally, the parameters are calculated by using the numpy.linalg function which easily solves the matrix equation of the XX value against the transposed, dot product of the y_train. This transposed value is found using the same method used to find XX but instead does not require a feature expansion to be performed beforehand. The pol_regression function is then called later in the code several times for each required degree (0,1,2,3,6,10). These values are now the parameters, sometimes called weights, for the regression to be plotted.

In order to plot the polynomial regression, a plot function was created to make this process easier. Again, as seen previously, an exclusive case for degree equalling zero is created, which multiples the weights/parameters rather than taking the dot product of the matrix. The resulting polynomials are plotted within the range of -5 to 5 as required, as well as the feature expansion of the range and whichever degree value will be called. The resulting matrix is then taken as the dot product alongside the weight being selected from the previous figure.

The plotting function is then called six times with each weight value (weight_0, weight_1...) and the appropriate degree which matches this weight. After these plots have been created, the original data is plotted on the same figure as points in blue to show the regression goes through the points as expected. Some styling is used to label the axis as well as create a legend, labelling each of the lines to the appropriate polynomial. Finally, the created graph is saved in the workspace using the plt function and shown in the plots view.


## Evaluation of Polynomial Regression
For an accurate evaluation of a regression model to exist, a numerical value for each different regression degree needs to be found. RMSE is the chosen method to perform the evaluation and compare the different lines. RMSE is a useful metric to do this as it is directly interpretable to the data when compared to other evaluation methods.

Before any calculations can occur, the data must first be split into training and testing data. This is done on a 70:30 ratio, with training making up 70% of the data. The ‘train_test_split’ function from the sklearn library is used to perform this split, taking the inputs from the x and y columns and the desired test size. Four input variables are automatically assigned this data, in the order asked for by the function. Before these four arrays are created, the function randomly shuffles the rows of the data, to ensure the data is split randomly.

The primary function created to deal with evaluating the polynomial regressions is called ‘eval_pol_regression’. It takes the parameters calculated from the previous functions as well as the appropriate x and y values and again the degree regression. For calculating the RMSE value, the feature expansion function is called, and the variable M is created. The degree equals 0 case is dealt with separately by substituting the dot product for multiplication. RMSE is found by taking the dot product of the different weights and taking away the y column values. In the same line, this is then squared, and afterward, the mean is taken. Finally, the square root can be taken, leaving the final RMSE value to be returned.

Once the evaluation function has been initialised, and the data has been successfully assigned into training and testing sets, the function can be called for each individual RMSE value that needs to be calculated. The parameters are the weights from the previously found regressions. Finally, the degree is given, which matches that of the parameter for each regression. This leaves six training RMSE values and six testing RMSE values. These values on then plotted and results can be deriven from them.


## Implementation of the K-Means Clustering
To begin implementing k-means clustering, the data was read in using the ‘get_data’ created function, this uses the pandas library to read in the file. Next, the data is shuffled randomly while maintaining the row structure to ensure the data is not completely randomised.

The first step for the actual clustering algorithm is to find the Euclidean distance. This will be the distance between each data point and the centroid as defined later in the code. Calculating this involves finding the distance between two vectors (vec_1 and vec_2), which is done by taking the first from the second and taking the square of this value. This is then summed and square rooted using the numpy library to make this more efficient.

The next step was to randomly initialise the centroids, which was done by creating a function that takes the arguments of the dataset and the number of clusters that will be used. It uses the built-in ‘randint’ function to select a value from the dataset randomly and assigns it to the centroid’s value to
be returned.

The following is still under progress...
