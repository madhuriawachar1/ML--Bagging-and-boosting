import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from sklearn.metrics import mean_squared_error
from ensemble.gradientBoosted import GradientBoostedRegressor
from tree.base import WeightedDecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import sklearn.ensemble
from metrics import rmse
X, y= make_regression(
       n_features=3,
       n_informative=3,
       noise=10,
       tail_strength=10,
       random_state=42,
   )

X = pd.DataFrame(X)
y = pd.Series(y)

tree = DecisionTreeRegressor(max_depth=2)
GradBoost = GradientBoostedRegressor(tree, n_estimators=10, learning_rate =0.1)
GradBoost.fit(X,y)
y_hat = GradBoost.predict(X)
print(mean_squared_error(y,y_hat))
#print(y_hat)
GradBoost = sklearn.ensemble.GradientBoostingRegressor(max_depth=2,learning_rate=0.1,n_estimators=10,loss="squared_error")
GradBoost.fit(X,y)
y_hat = GradBoost.predict(X)
print(mean_squared_error(y,y_hat))
# Or use sklearn decision tree
#print(y_hat)
########### GradientBoostedClassifier ###################
