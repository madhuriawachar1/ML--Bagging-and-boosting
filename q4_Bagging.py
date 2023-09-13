import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from metrics import *

from ensemble.bagging import BaggingClassifier
from tree.base import WeightedDecisionTree
import multiprocessing
# Or use sklearn decision tree
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
########### BaggingClassifier ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
criteria = "entropy"
if y.dtype=="category":
    tree = DecisionTreeClassifier(criterion=criteria)
else:
    tree = DecisionTreeRegressor(criterion=criteria)

start_time = time.perf_counter()
Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators)
Classifier_B.fit(X, y)
y_hat = Classifier_B.predict(X)
[fig1, fig2] = Classifier_B.plot(X,y)
print("Criteria :", criteria)
print("Accuracy: ", accuracy(y_hat, y))
for cls in y.unique():
    print("Precision: ", precision(y_hat, y, cls))
    print("Recall: ", recall(y_hat, y, cls))
finish_time = time.perf_counter()
'''if i==1:
    print('Regular Implementation(time): ',finish_time-start_time)
else:
    print('Parallel Implementation(time): ',finish_time-start_time)'''
