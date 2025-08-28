import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def linearRegression():
    data = pd.read_csv("Machine-Learning-Proj\Pages\California_Houses.csv")
    data = pd.DataFrame(data)
    x = data.drop(columns=["Median_House_Value"])
    y = data["Median_House_Value"]
    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    lr = LinearRegression()
    lr.fit(x_train , y_train)
    y_pred = lr.predict(x_test)
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return lr.intercept_, lr.coef_, [mse, r2]

def lassoRegression():
    data = pd.read_csv("Machine-Learning-Proj\Pages\California_Houses.csv")
    data = pd.DataFrame(data)
    x = data.drop(columns=["Median_House_Value"])
    y = data["Median_House_Value"]
    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    lr = Lasso()
    lr.fit(x_train , y_train)
    y_pred = lr.predict(x_test)
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return lr.intercept_, lr.coef_, [mse, r2]

def ridgeRegression():
    data = pd.read_csv("Machine-Learning-Proj\Pages\California_Houses.csv")
    data = pd.DataFrame(data)
    x = data.drop(columns=["Median_House_Value"])
    y = data["Median_House_Value"]
    x_train , x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    lr = Ridge()
    lr.fit(x_train , y_train)
    y_pred = lr.predict(x_test)
    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return lr.intercept_, lr.coef_, [mse, r2]