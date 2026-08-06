"""Stage 2 of the combinatorial-sputtering -> device-physics -> ML pipeline.

ProcessCorrelationModel learns the mapping from sputtering process conditions
(RF power, pressure, O2:Ar ratio, thickness, ...) to whatever targets you give it --
typically the Stage 1 drift-diffusion fitted physical parameters (mobility, doping,
trap density, ...) and/or the measured device performance (Voc, Jsc, FF, PCE). This
is deliberately a classical regressor (Random Forest / Gradient Boosting by default,
with a Gaussian Process option for small datasets with calibrated uncertainty) rather
than a deep net: combinatorial-sputtering campaigns typically produce tens to a few
hundred devices, not the thousands a neural network needs, and tree-based models give
directly interpretable feature importances -- i.e. an actual, ranked answer to "which
knob controls what".
"""
######### Package Imports #########################################################################

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

######### Class Definition #####################################################################


class ProcessCorrelationModel:
    """Fit process-conditions -> target(s) regressors and expose feature importance /
    partial dependence for interpreting which process parameter drives which physical
    parameter or performance metric.

    Parameters
    ----------
    model : str, optional
        One of 'rf' (RandomForestRegressor), 'gbr' (GradientBoostingRegressor), or
        'gp' (GaussianProcessRegressor, gives predictive uncertainty -- best for
        small datasets, e.g. < ~50 devices), by default 'rf'.
    model_kwargs : dict, optional
        Extra keyword arguments passed to the underlying sklearn estimator.
    random_state : int, optional
        Random seed, by default 0.
    """

    def __init__(self, model='rf', model_kwargs=None, random_state=0):
        self.model_name = model
        self.model_kwargs = model_kwargs or {}
        self.random_state = random_state
        self.feature_cols = None
        self.target_cols = None
        self.scaler_X = None
        self.estimator = None

    def _build_estimator(self):
        if self.model_name == 'rf':
            base = RandomForestRegressor(n_estimators=300, random_state=self.random_state, **self.model_kwargs)
        elif self.model_name == 'gbr':
            base = GradientBoostingRegressor(random_state=self.random_state, **self.model_kwargs)
        elif self.model_name == 'gp':
            kernel = Matern(nu=2.5) + WhiteKernel()
            base = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.random_state,
                                             **self.model_kwargs)
        else:
            raise ValueError("model must be 'rf', 'gbr' or 'gp'")
        return base

    def fit(self, df, feature_cols, target_cols):
        """Fit the model.

        Parameters
        ----------
        df : DataFrame
            One row per device, with process-condition columns and target columns
            (e.g. the Stage 1 output of scripts/fit_combinatorial_devices.py).
        feature_cols : list of str
            Process-condition columns to use as inputs (e.g. ['rf_power_W',
            'pressure_mTorr', 'o2_ar_ratio', 'thickness_nm']).
        target_cols : list of str
            Target columns to predict (e.g. ['l1.mu_n', 'l1.N_t_bulk', 'PCE']).

        Returns
        -------
        self
        """
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)

        X = df[self.feature_cols].to_numpy(dtype=float)
        Y = df[self.target_cols].to_numpy(dtype=float)

        self.scaler_X = StandardScaler().fit(X)
        Xs = self.scaler_X.transform(X)

        base = self._build_estimator()
        self.estimator = MultiOutputRegressor(base) if len(self.target_cols) > 1 else base
        self.estimator.fit(Xs, Y if len(self.target_cols) > 1 else Y.ravel())
        self._X_train, self._Y_train = Xs, Y
        return self

    def predict(self, df):
        """Predict target(s) for new process conditions.

        Parameters
        ----------
        df : DataFrame
            Must contain self.feature_cols.

        Returns
        -------
        DataFrame
            Predictions, one column per target.
        """
        X = df[self.feature_cols].to_numpy(dtype=float)
        Xs = self.scaler_X.transform(X)
        preds = self.estimator.predict(Xs)
        preds = np.atleast_2d(preds.T).T if preds.ndim == 1 else preds
        return pd.DataFrame(preds, columns=self.target_cols, index=df.index)

    def cross_val_r2(self, cv=5):
        """Leave-some-out cross-validated R^2 per target, to sanity-check the fit
        before trusting feature importances or using the model as a BO surrogate.

        Returns
        -------
        dict
            {target_name: mean_cv_r2}
        """
        scores = {}
        for i, target in enumerate(self.target_cols):
            y = self._Y_train[:, i] if self._Y_train.ndim > 1 else self._Y_train
            est = self._build_estimator()
            cv_n = min(cv, len(y))
            scores[target] = float(np.mean(cross_val_score(est, self._X_train, y, cv=cv_n, scoring='r2')))
        return scores

    def feature_importance(self, n_repeats=20):
        """Permutation feature importance per target -- answers "which process knob
        controls which physical parameter / performance metric", ranked.

        Returns
        -------
        DataFrame
            Rows = feature_cols, columns = target_cols, values = mean importance.
        """
        importances = pd.DataFrame(index=self.feature_cols, columns=self.target_cols, dtype=float)
        estimators = self.estimator.estimators_ if hasattr(self.estimator, 'estimators_') else [self.estimator]
        for i, target in enumerate(self.target_cols):
            est = estimators[i] if len(self.target_cols) > 1 else self.estimator
            y = self._Y_train[:, i] if self._Y_train.ndim > 1 else self._Y_train
            r = permutation_importance(est, self._X_train, y, n_repeats=n_repeats, random_state=self.random_state)
            importances[target] = r.importances_mean
        return importances

    def plot_feature_importance(self, ax=None):
        """Bar plot of feature_importance(), one group of bars per target."""
        import matplotlib.pyplot as plt
        importances = self.feature_importance()
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(self.feature_cols)), 4))
        importances.plot(kind='bar', ax=ax)
        ax.set_ylabel('Permutation importance')
        ax.set_title('Process condition -> target importance')
        ax.legend(title='target')
        return ax

    def plot_partial_dependence(self, feature, target, ax=None):
        """1D partial dependence of `target` on process-condition `feature`, holding
        other features at their observed distribution -- shows the shape (not just
        strength) of a process/physics relationship, e.g. "does mu_n increase or
        decrease with O2:Ar ratio, and is it monotonic".
        """
        import matplotlib.pyplot as plt
        target_idx = self.target_cols.index(target)
        estimators = self.estimator.estimators_ if hasattr(self.estimator, 'estimators_') else [self.estimator]
        est = estimators[target_idx] if len(self.target_cols) > 1 else self.estimator
        feat_idx = self.feature_cols.index(feature)

        pd_result = partial_dependence(est, self._X_train, [feat_idx], kind='average')
        xs_scaled = pd_result['grid_values'][0]
        ys = pd_result['average'][0]
        xs = xs_scaled * self.scaler_X.scale_[feat_idx] + self.scaler_X.mean_[feat_idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xs, ys)
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f'Partial dependence of {target} on {feature}')
        ax.grid(alpha=0.3)
        return ax
