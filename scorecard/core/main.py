import pandas as pd
import numpy as np
from time import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from typing import Union, List, Tuple
from .functions import cat_bining
from .functions import num_bining


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_finale: int = 10,
                 min_pcnt_group: float = 0.05,
                 verbose: bool = False,
                 cat_features: List = None,
                 cat_features_threshold: int = 0,
                 missing_mask: str = 'NaN',
                 cat_features_temperature: float = 0.005,
                 specials: List = []):
        """
        Performs the Weight Of Evidence transformation over the input X features using information from y vector.
        :param verbose: boolean flag to add verbose output

        TODO: add n_jobs
        """
        self.n_finale = n_finale
        self.min_pcnt_group = min_pcnt_group
        self.cat_features = cat_features if cat_features else []
        self.cat_features_threshold = cat_features_threshold
        self.missing_mask = missing_mask
        self.cat_features_temperature = cat_features_temperature
        self.specials = specials if specials else []
        self.verbose = verbose

        self.WOE_IV_dict = []  # self.transformers
        self.feature_names = []
        self.num_features  = []

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]):
        """
        Fits the input data
        :param X: data matrix
        :param y: target vector
        :return: self
        """
        if isinstance(X, pd.DataFrame):
            if self.specials:
                self.feature_names = X.drop(self.specials, axis=1).columns
            else:
                self.feature_names = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names = ['X%i' for i in range(X.shape[-1])]
        else:
            raise TypeError('X vector is not np array neither data frame')

        X, y = self._check_inputs(X, y)
        
        start_time = time()
        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            for i in range(len(self.feature_names)):
                if type(X[0, i]) == np.dtype('object') \
                    or type(X[0, i]) == np.dtype('str') \
                    or len(np.unique(X[:, i][~np.isin(X[:, i], self.missing_mask)])) < self.cat_features_threshold:
                    self.cat_features.append(self.feature_names[i])
        if len(self.cat_features) > 0:
            self.num_features = [feature for feature in self.feature_names if feature not in self.cat_features]
            for feature in self.cat_features:
                feature_idx = list(self.feature_names).index(feature)
                self._print(f'Preparing {feature} feature')
                self.WOE_IV_dict.append(
                    {feature: cat_bining(X=X[:, feature_idx],
                                         y=y,
                                         min_pcnt_group=self.min_pcnt_group,
                                         n_finale=self.n_finale,
                                         temperature=self.cat_features_temperature,
                                         mask=self.missing_mask),
                     'type': 'categotical'}
                )
        else:
            self.num_features = self.feature_names

        for feature in self.num_features:
            feature_idx = list(self.feature_names).index(feature)
            self._print(f'Preparing {feature} feature')
            self.WOE_IV_dict.append(
                {feature: num_bining(X=X[:, feature_idx],
                                     y=y,
                                     min_pcnt_group=self.min_pcnt_group,
                                     n_finale=self.n_finale,
                                     mask=self.missing_mask),
                 'type': 'numeric'}
            )
        print(round(time() - start_time, 4))


    def _check_inputs(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check input data
        :param X: data matrix
        :param y: target vector
        :return: X, y
        """
        if type_of_target(y) != 'binary':
            raise ValueError('y vector should be binary')
        
        X, y = check_X_y(X, y,
                         accept_sparse=False,
                         force_all_finite=False,
                         dtype=None,
                         y_numeric=True)
        return X, y


    def _print(self,
               msg: str):
        if self.verbose:
            print(msg)
