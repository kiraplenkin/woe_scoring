import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.svm import l1_min_c
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from typing import Union, List



class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self):
        ...

    def fit(self, data, target):
        """
        Fits a logistic regression model to the given data and target.

        Args:
            data: pandas.DataFrame or numpy.ndarray
                The input data matrix of shape (n_samples, n_features).
            target: pandas.Series or numpy.ndarray
                The target vector of shape (n_samples,).

        Returns:
            self: LogisticRegression
                The fitted model.
        """

        self.model_ = sm.Logit(target, sm.add_constant(data)).fit()
        return self

    def predict(self, data):
        """
        Predict the binary outcome of a dataset using the fitted logistic regression model.

        Args:
            data (array-like): The dataset to predict, with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The predicted binary outcomes, with shape (n_samples,).

        """

        decision = self.model_.predict(sm.add_constant(data)) > 0.5
        return np.array(decision.astype(int), dtype=np.int64)

    def predict_proba(self, data):
        """
        Predict class probabilities for input data.

        Args:
            data (array-like): Input data to predict probabilities for.

        Returns:
            array: An array of shape (n_samples, 2) containing the predicted 
            probabilities for each class, where n_samples is the number of 
            samples in the input data. The first column contains the probability 
            of the negative class and the second column contains the probability 
            of the positive class.
        """

        decision = self.model_.predict(sm.add_constant(data))
        decision_2d = np.c_[1 - decision, decision]
        return np.array(decision_2d, dtype=np.float64)


class Model:
    """
    Initialize a new model.

    Args:
    - model_type (str): the type of the model to be used.
    - l1_exp_scale (int): the exponent scale to be used in the L1 regularization.
    - l1_grid_size (int): the number of grid points to be used in the L1 regularization.
    - cv (int): the number of folds in the cross-validation. Default is None.
    - class_weight (str): the weight to be assigned to each class. Default is None.
    - random_state (int): the seed to be used in the random number generator. Default is None.
    - n_jobs (int): the number of CPU cores to be used in parallel computation. Default is None.
    - scoring (str): the scoring metric to be used. Default is None.

    Returns:
    - None
    """

    def __init__(self, model_type: str, l1_exp_scale: int, l1_grid_size: int, cv: int = None, class_weight: str = None, random_state: int = None, n_jobs: int = None, scoring: str = None) -> None:
        self.model_type = model_type
        self.cv = cv
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.l1_exp_scale = l1_exp_scale
        self.l1_grid_size = l1_grid_size

        self.model = self._get_model(model_type)
        self.coef_ = List[float]
        self.intercept_ = float
        self.feature_names_ = List[str]
        self.model_score_ = float
        self.pvalues_ = List[float]


    def get_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """
        Returns a callable object that can be used to make predictions based on the provided data and target.

        :param data: The input data to use for making predictions.
        :type data: pd.DataFrame
        :param target: The target values to use for making predictions.
        :type target: Union[pd.Series, np.ndarray]
        :return: A callable object that can be used to make predictions.
        :rtype: callable
        """

        return self.model(data, target)

    
    def _get_model(self, model_type: str) -> callable:
        if model_type == 'sklearn':
            return self._get_sklearn_model
        elif model_type == 'statsmodels':
            return self._get_statsmodels_model
        else:
            raise ValueError(f'Unknown model type: {model_type}. Should be either "sklearn" or "statsmodels"')

    def _get_sklearn_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """
        Returns a callable sklearn logistic regression model trained on the given data and target.

        Args:
            data (pd.DataFrame): The input data to train the model on.
            target (Union[pd.Series, np.ndarray]): The target values to train the model on.

        Returns:
            callable: The trained logistic regression model.
        """

        Cs = l1_min_c(data, target, loss="log", fit_intercept=True) * np.logspace(0, self.l1_exp_scale, self.l1_grid_size)
        model = LogisticRegressionCV(
            Cs=Cs,
            cv=self.cv,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            solver='saga',
            tol=1e-5,
            intercept_scaling=10000.0,
            penalty="l1",
            max_iter=1000,
            scoring=self.scoring
        ).fit(data, target)
        self.coef_ = list(model.coef_[0])
        self.intercept_ = model.intercept_[0]
        self.feature_names_ = data.columns.to_list()
        self.model_score_ = cross_val_score(model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring).mean()
        self.pvalues_ = list(self._calc_pvalues(model, data))
        return model

    
    def _get_statsmodels_model(self, data: pd.DataFrame, target: Union[pd.Series, np.ndarray]) -> callable:
        """ Fits a statsmodels model to the given data and target, and returns the trained model.

        Args: 
            data (pd.DataFrame): The input data to be used for training the model.
            target (Union[pd.Series, np.ndarray]): The target variable to be used for training the model.

        Returns:
            callable: The trained statsmodels model.
        """

        model = SMWrapper().fit(data, target)
        self.coef_ = list(model.model_.params[1:])
        self.intercept_ = model.model_.params[0]
        self.feature_names_ = data.columns.to_list()
        self.model_score_ = cross_val_score(model, data, target, cv=self.cv, n_jobs=self.n_jobs, scoring=self.scoring).mean()
        self.pvalues_ = list(model.model_.pvalues)[1:]
        return model


    def _calc_pvalues(self, model: callable, data: pd.DataFrame) -> np.ndarray:
        """
        Calculates p-values for a logistic regression model.

        Args:
            model: A logistic regression model fit using scikit-learn.
            data: A Pandas DataFrame of features.

        Returns:
            A NumPy array of p-values for each feature.
        """

        p = model.predict_proba(data)[:, 1]
        coefs = np.concatenate([model.intercept_, model.coef_[0]])
        x_full = np.insert(np.array(data), 0, 1, axis=1)
        ans = np.einsum('ij,ik,i->jk', x_full, x_full, p*(1-p))
        vcov = np.linalg.inv(ans)
        se = np.sqrt(np.diag(vcov))
        t = coefs/se
        p = (1 - norm.cdf(abs(t))) * 2
        return p[1:]
