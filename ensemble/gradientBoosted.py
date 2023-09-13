import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn
class GradientBoostedRegressor:
    def __init__(
        self, base_estimator, n_estimators=3, learning_rate=0.1
    ):  # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        pass

    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        h0 = np.mean(y)
        self.models.append(h0)
        residual = y-h0
        #print(residual)
        y_hat = h0
        for estimator in range(self.n_estimators):
            model = self.base_estimator
            model.fit(X,residual)
            self.models.append(model)
            y_hat += self.learning_rate*model.predict(X)
            residual-=self.learning_rate*model.predict(X)
            #print(residual)
        #for i in self.models[1:]:
         #   print(sklearn.tree.export_text(i))

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = pd.Series([self.models[0]]*X.shape[0])
        for i in range(1,len(self.models)):
            #print(y_hat[0])
            if y_hat is None:
                y_hat = self.learning_rate*self.models[i].predict(X)
                
            else:
                y_hat += self.learning_rate*self.models[i].predict(X)
            #print(self.models[i].predict(X)[0])
            #print(self.learning_rate*self.models[i].predict(X)[0])
        #y_hat += self.models[0]
        return pd.Series(y_hat)
        pass
