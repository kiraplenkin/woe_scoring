import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from typing import Union, List, Tuple
from .functions import cat_feature_bining
from .functions import calc_WOE
from .functions import chi_merge
from .functions import assign_bin
from .functions import bad_rate_monotone


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_finale: int = 15,
                 verbose: bool = False,
                 cat_features: List = None,
                 cat_features_threshold: int = 0,
                 specials: List = []):
        """
        Performs the Weight Of Evidence transformation over the input X features using information from y vector.
        :param verbose: boolean flag to add verbose output

        TODO: add n_jobs
        """
        self.n_finale = n_finale
        self.cat_features = cat_features if cat_features else []
        self.cat_features_threshold = cat_features_threshold
        self.specials = specials if specials else []
        self.verbose = verbose

        self.WOE_IV_dict = {}  # self.transformers
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

        # X, y = self._check_inputs(X, y)
        X_prep = X.copy(deep=True)

        if len(self.cat_features) == 0 and self.cat_features_threshold > 0:
            for feature in X_prep.columns:
                if X_prep[feature].nunique() <= self.cat_features_threshold \
                    or X_prep[feature].dtype == 'object' \
                    or X_prep[feature].dtype == 'str':
                    self.cat_features.append(feature)
        
        if len(self.cat_features) > 0:
            self.num_features = [feature for feature in X_prep.drop(self.cat_features, axis=1).columns]
            X_prep['target'] = y
            for feature in self.cat_features:
                X_prep[feature] = X_prep[feature].fillna('Missing')
                cat_feature_bining(df=X_prep,
                                   var=feature,
                                   target='target',
                                   max_bin=5,
                                   verbose=False)  # TODO verbose
                self.WOE_IV_dict[feature] = calc_WOE(df=X_prep,
                                                     col=feature,
                                                     target='target',
                                                     cat=True)
        else:
            self.num_features = [feature for feature in X_prep.columns]
        
        X_prep['target'] = y
        for feature in self.num_features:
            print(f'Preparing {feature}')

            bad_rate_min_value = X_prep[X_prep[feature] == X_prep[feature].min()]['target'].sum() * 1.0 / X_prep.shape[0]
            bad_rate_max_value = X_prep[X_prep[feature] == X_prep[feature].max()]['target'].sum() * 1.0 / X_prep.shape[0]

            if bad_rate_min_value > bad_rate_max_value:
                X_prep[feature] = X_prep[feature].fillna(np.NINF)
            else:
                X_prep[feature] = X_prep[feature].fillna(np.Inf)
                
            bin_num = self.cat_features_threshold
            new_bin = feature + '_Bin'
            bin, group_intervals = chi_merge(df=X_prep,
                                            col=feature,
                                            target='target',
                                            max_interval=bin_num,
                                            min_bin_pcnt=0.05)
            X_prep[new_bin] = X_prep[feature].apply(lambda x: assign_bin(x=x,
                                                               cut_off_points=bin,
                                                               group_intervals=group_intervals))
            while not bad_rate_monotone(df=X_prep,
                                        sort_by_var=new_bin,
                                        target='target'):
                bin_num -= 1
                bin, group_intervals = chi_merge(df=X_prep,
                                                 col=feature,
                                                 target='target',
                                                 max_interval=bin_num,
                                                 min_bin_pcnt=0.05)
                X_prep[new_bin] = X_prep[feature].apply(lambda x: assign_bin(x=x,
                                                                   cut_off_points=bin,
                                                                   group_intervals=group_intervals))
            self.WOE_IV_dict[feature] = calc_WOE(df=X_prep,
                                                 col=new_bin,
                                                 target='target')
        return self


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


    def transform(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray] = None,
                  hight_IV_threshold: float = 0.02,
                  suffix: str = '_WOE'):
        """
        Checks and transforms input arrays
        :param X: X data array
        :param y: target array
        :return: transformed data
        """
        X['target'] = y  # TODO delete y. Give y from fit dict
        hight_IV = [(k, v['IV']) for k, v in self.WOE_IV_dict.items() if v['IV'] >= hight_IV_threshold]
        # hight_IV_sorted = sorted(hight_IV, key=lambda x: x[1], reverse=True)

        if len(self.cat_features) > 0:
            for feature in self.cat_features:
                X[feature] = X[feature].fillna('Missing')
        
        for feature in self.num_features:
            bad_rate_min_value = X[X[feature] == X[feature].min()]['target'].sum() * 1.0 / X.shape[0]
            bad_rate_max_value = X[X[feature] == X[feature].max()]['target'].sum() * 1.0 / X.shape[0]

            if bad_rate_min_value > bad_rate_max_value:
                X[feature] = X[feature].fillna(np.NINF)
            else:
                X[feature] = X[feature].fillna(np.Inf)

        for (feature, _) in hight_IV:
            new_feature = feature + suffix
            X[new_feature] = X[feature].map(lambda x: self.WOE_IV_dict[feature]['WOE'][x])
        return X
