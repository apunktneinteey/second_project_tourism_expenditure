# Calculate metric
import numpy as np
from numpy import mean, std, absolute
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def calculate_metrics(y_train, y_pred_train, y_test, y_pred_test):
    """Calculate and print out RMSE and R2 for train and test data
    Args:
        y_train (array): true values of y_train
        y_pred_train (array): predicted values of model for y_train
        y_test (array): true values of y_test
        y_pred_test (array): predicted values of model for y_test
    """

    print("Metrics on training data") 
    rmse_train = np.sqrt(mean_squared_error(y_train,y_pred_train))
    r2 = r2_score(y_train,y_pred_train)
    print("RMSE:", round(rmse_train, 3))
    print("RMSE Antilog", np.exp(rmse_train) )
    print("R2:", round(r2, 3))
    print("Mean of the y train set", (y_train.mean()))
    print("Antilog of the mean of the y train set", (np.exp(y_train.mean())))
    print("Mean absolute error", mean_absolute_error(y_train, y_pred_train))
    print("Mean absolute error antilog", np.exp(mean_absolute_error(y_train, y_pred_train)) )
    print("---"*10)
    
    # Calculate metric
    print("Metrics on test data")  
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    # you can get the same result with this line:
    # rmse = np.sqrt(mean_squared_error(y_test,y_pred_test))

    r2 = r2_score(y_test,y_pred_test)
    print("RMSE:", round(rmse_test, 3))
    print("RMSE Antilog", np.exp(rmse_test))
    print("R2:", round(r2, 3))
    print("Mean of the y test set", (y_test.mean()))
    print("Antilog of the mean of the y test set", (np.exp(y_test.mean())))
    print("Mean absolute error", mean_absolute_error(y_test, y_pred_test))
    print("Mean absolute error antilog", np.exp(mean_absolute_error(y_test, y_pred_test)) )
    print("---"*10)