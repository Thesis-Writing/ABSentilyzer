import numpy as np
import warnings
from scipy import sparse
from scipy.special import logsumexp
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted,check_non_negative, _check_sample_weight

class _BaseNB(ClassifierMixin, BaseEstimator):
	"""Abstract base class for naive Bayes estimators"""

	def _joint_log_likelihood(self, X):
		"""Compute the unnormalized posterior log probability of X

		I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
		shape (n_classes, n_samples).

		Input is passed to _joint_log_likelihood as-is by predict,
		predict_proba and predict_log_proba.
		"""

	def _check_X(self, X):
		"""To be overridden in subclasses with the actual checks.

		Only used in predict* methods.
		"""

	def predict(self, X):
		"""
		Perform classification on an array of test vectors X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
				The input samples.

		Returns
		-------
		C : ndarray of shape (n_samples,)
				Predicted target values for X.
		"""
		check_is_fitted(self)
		X = self._check_X(X)
		jll = self._joint_log_likelihood(X)
		return self.classes_[np.argmax(jll, axis=1)]

	def predict_log_proba(self, X):
		"""
		Return log-probability estimates for the test vector X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
				The input samples.

		Returns
		-------
		C : array-like of shape (n_samples, n_classes)
				Returns the log-probability of the samples for each class in
				the model. The columns correspond to the classes in sorted
				order, as they appear in the attribute :term:`classes_`.
		"""
		check_is_fitted(self)
		X = self._check_X(X)
		jll = self._joint_log_likelihood(X)
		# normalize by P(x) = P(f_1, ..., f_n)
		log_prob_x = logsumexp(jll, axis=1)
		return jll - np.atleast_2d(log_prob_x).T

	def predict_proba(self, X):
		"""
		Return probability estimates for the test vector X.

		Parameters
		----------
		X : array-like of shape (n_samples, n_features)
				The input samples.

		Returns
		-------
		C : array-like of shape (n_samples, n_classes)
				Returns the probability of the samples for each class in
				the model. The columns correspond to the classes in sorted
				order, as they appear in the attribute :term:`classes_`.
		"""
		return np.exp(self.predict_log_proba(X))

_ALPHA_MIN = 1e-10

class _BaseDiscreteNB(_BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per _BaseNB
    """

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y in fit methods."""
        return self._validate_data(X, y, accept_sparse="csr", reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        if np.min(self.alpha) < 0:
            raise ValueError(
                "Smoothing parameter alpha = %.1e. alpha should be > 0."
                % np.min(self.alpha)
            )
        if isinstance(self.alpha, np.ndarray):
            if not self.alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "alpha should be a scalar or a numpy array with shape [n_features]"
                )
        if np.min(self.alpha) < _ALPHA_MIN:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha = %.1e"
                % _ALPHA_MIN
            )
            return np.maximum(self.alpha, _ALPHA_MIN)
        return self.alpha

    def fit(self, X, y, sample_weight=None):
        """Fit Naive Bayes classifier according to X, y.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights applied to individual samples (1. for unweighted).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def coef_(self):
        return (
            self.feature_log_prob_[1:]
            if len(self.classes_) == 2
            else self.feature_log_prob_
        )
    
    def intercept_(self):
        return (
            self.class_log_prior_[1:]
            if len(self.classes_) == 2
            else self.class_log_prior_
        )

    def _more_tags(self):
        return {"poor_score": True}

    def n_features_(self):
        return self.n_features_in_

def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly.

    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()
    return ret

class MultinomialNB(_BaseDiscreteNB):
	def __init__(self, *, alpha=1.0, fit_prior=True, class_prior=None):
		self.alpha = alpha
		self.fit_prior = fit_prior
		self.class_prior = class_prior

	def _more_tags(self):
		return {"requires_positive_X": True}

	def _count(self, X, Y):
		"""Count and smooth feature occurrences."""
		check_non_negative(X, "MultinomialNB (input X)")
		self.feature_count_ += safe_sparse_dot(Y.T, X)
		self.class_count_ += Y.sum(axis=0)

	def _update_feature_log_prob(self, alpha):
		"""Apply smoothing to raw counts and recompute log probabilities"""
		smoothed_fc = self.feature_count_ + alpha
		smoothed_cc = smoothed_fc.sum(axis=1)

		self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
				smoothed_cc.reshape(-1, 1)
		)

	def _joint_log_likelihood(self, X):
		"""Calculate the posterior log probability of the samples X"""
		return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_

	def _init_counters(self, n_classes, n_features):
		self.class_count_ = np.zeros(n_classes, dtype=np.float64)
		self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)