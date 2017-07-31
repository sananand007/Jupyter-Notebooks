'''
author: Sandeep Anand
version: Python 3.6
- Illustrating the usage of numpy
- Usage of Pandas
- Usage of sklearn
'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('max_columns',50)
%matplotlib inline

# %%
if True:
    array_1 = np.array([[1,3],[5,7]], float)
    array_2 = np.array([[6,9],[1,8]], float)

    print(array_1+array_2)
    print(array_1*array_2)

# %%
s=pd.Series([7, "Heisenberg", 3.14, -434524354, "Journey of Life!"], index=["A","B","C","D","E"])
s

# %%
# Getting to use api's for pandas
data = {
'year' : [2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
'team' : ['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions',
                     'Lions', 'Lions'],
'wins' : [11, 8, 10, 15, 11, 6, 10, 4],
'losses' : [5, 8, 6, 1, 5, 10, 6, 12]
}
football = pd.DataFrame(data)
print(football.dtypes)
print(football.describe())
print(football.head())
print(football.tail())

# %%
# Using sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Load the datasets
housing_data = datasets.load_boston()
linear_regression_model = LinearRegression()

linear_regression_model.fit(housing_data.data, housing_data.target)

predictions = linear_regression_model.predict(housing_data.data)

score = metrics.r2_score(housing_data.target, predictions)

print(score)

# %%
# Using sklearn and Gaussian Naive Bayes

from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabled points after the model fit : %d out of total points %d" %((iris.target!=y_pred).sum(), iris.data.shape[0]))
iris.data.shape
