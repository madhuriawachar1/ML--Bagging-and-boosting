from tree.base import WeightedDecisionTree
from sklearn.metrics import accuracy_score, mean_squared_error
from metrics import accuracy, rmse
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from metrics import accuracy
#print(np.random.uniform(0,1,size=10))
np.random.seed(42)
## compare both the trees
N = 200
P = 3
NUM_OP_CLASSES = 3

X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N),dtype='category')
weights=pd.Series(np.random.uniform(0,1,size=y.size))
X = X.sample(frac=1)
y = y[X.index]
weights = weights[X.index]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
weights =weights.reset_index(drop=True)
train_test_split = int(y.size*0.7)
X_train = X.iloc[:train_test_split]
y_train = y.iloc[:train_test_split]
X_test = X.iloc[train_test_split:]
y_test = y.iloc[train_test_split:]
weights_test = weights.iloc[train_test_split:]
weights=weights.iloc[:train_test_split]

clf = DecisionTreeClassifier(criterion='gini',max_depth=3)
clf.fit(X_train,y_train,sample_weight=weights)
y_hat1 = clf.predict(X_test)
print('Sklearn Accuracy:',accuracy_score(y_test,y_hat1))

clf2 = WeightedDecisionTree(criterion='gini',max_depth = 3)
clf2.fit(X_train,y_train,sample_weights=weights)
y_hat2 = clf2.predict(X_test)
print('DecisionTree Accuracy:',accuracy(y_test,y_hat2))

