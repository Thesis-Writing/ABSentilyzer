from sklearn.base import clone, is_regressor, is_classifier, BaseEstimator, ClassifierMixin, MultiOutputMixin, MetaEstimatorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted, _num_samples
from sklearn.utils.fixes import delayed
from joblib import Parallel
import numpy as np
import array
import scipy.sparse as sp
import warnings


def _fit_binary(estimator, X, y, classes=None):
  """Fit a single binary estimator."""
  unique_y = np.unique(y)
  if len(unique_y) == 1:
      if classes is not None:
          if y[0] == -1:
              c = 0
          else:
              c = y[0]
          warnings.warn(
              "Label %s is present in all training examples." % str(classes[c])
          )
      estimator = _ConstantPredictor().fit(X, unique_y)
  else:
      estimator = clone(estimator)
      estimator.fit(X, y)
  return estimator

def _predict_binary(estimator, X):
  """Make predictions using a single binary estimator."""
  if is_regressor(estimator):
      return estimator.predict(X)
  try:
      score = np.ravel(estimator.decision_function(X))
  except (AttributeError, NotImplementedError):
      # probabilities of the positive class
      score = estimator.predict_proba(X)[:, 1]
  return score

class _ConstantPredictor(BaseEstimator):
  def fit(self, X, y):
      check_params = dict(
          force_all_finite=False, dtype=None, ensure_2d=False, accept_sparse=True
      )
      self._validate_data(
          X, y, reset=True, validate_separately=(check_params, check_params)
      )
      self.y_ = y
      return self

  def predict(self, X):
      check_is_fitted(self)
      self._validate_data(
          X,
          force_all_finite=False,
          dtype=None,
          accept_sparse=True,
          ensure_2d=False,
          reset=False,
      )

      return np.repeat(self.y_, _num_samples(X))

  def decision_function(self, X):
      check_is_fitted(self)
      self._validate_data(
          X,
          force_all_finite=False,
          dtype=None,
          accept_sparse=True,
          ensure_2d=False,
          reset=False,
      )

      return np.repeat(self.y_, _num_samples(X))

  def predict_proba(self, X):
      check_is_fitted(self)
      self._validate_data(
          X,
          force_all_finite=False,
          dtype=None,
          accept_sparse=True,
          ensure_2d=False,
          reset=False,
      )

      return np.repeat([np.hstack([1 - self.y_, self.y_])], _num_samples(X), axis=0)

class OneVsRestClassifier(
    MultiOutputMixin, ClassifierMixin, MetaEstimatorMixin, BaseEstimator
):
  
  def __init__(self, estimator, *, n_jobs=None):
    self.estimator = estimator
    self.n_jobs = n_jobs
  
  def fit(self, X, y):
    # A sparse LabelBinarizer, with sparse_output=True, has been shown to
    # outperform or match a dense label binarizer in all cases and has also
    # resulted in less or equal memory consumption in the fit_ovr function
    # overall.
    
    self.label_binarizer_ = LabelBinarizer(sparse_output=True)
    Y = self.label_binarizer_.fit_transform(y)
    Y = Y.tocsc()
    self.classes_ = self.label_binarizer_.classes_
    columns = (col.toarray().ravel() for col in Y.T)
    
    # In cases where individual estimators are very fast to train setting
    # n_jobs > 1 in can results in slower performance due to the overhead
    # of spawning threads.  See joblib issue #112.
    
    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
      delayed(_fit_binary)(
          self.estimator,
          X,
          column,
          classes=[
              "not %s" % self.label_binarizer_.classes_[i],
              self.label_binarizer_.classes_[i],
          ],
      )
      for i, column in enumerate(columns)
    )
    
    if hasattr(self.estimators_[0], "n_features_in_"):
        self.n_features_in_ = self.estimators_[0].n_features_in_
    if hasattr(self.estimators_[0], "feature_names_in_"):
        self.feature_names_in_ = self.estimators_[0].feature_names_in_

    return self
  
  def predict(self, X):
    check_is_fitted(self)
    
    n_samples = _num_samples(X)
    if self.label_binarizer_.y_type_ == "multiclass":
      maxima = np.empty(n_samples, dtype=float)
      maxima.fill(-np.inf)
      argmaxima = np.zeros(n_samples, dtype=int)
      for i, e in enumerate(self.estimators_):
          pred = _predict_binary(e, X)
          np.maximum(maxima, pred, out=maxima)
          argmaxima[maxima == pred] = i
      return self.classes_[argmaxima]
    else:
      if hasattr(self.estimators_[0], "decision_function") and is_classifier(
          self.estimators_[0]
      ):
          thresh = 0
      else:
          thresh = 0.5
      indices = array.array("i")
      indptr = array.array("i", [0])
      for e in self.estimators_:
          indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
          indptr.append(len(indices))
      data = np.ones(len(indices), dtype=int)
      indicator = sp.csc_matrix(
          (data, indices, indptr), shape=(n_samples, len(self.estimators_))
      )
      return self.label_binarizer_.inverse_transform(indicator)
    
  def predict_proba(self, X):
    check_is_fitted(self)
    # Y[i, j] gives the probability that sample i has the label j.
    # In the multi-label case, these are not disjoint.
    Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T

    if len(self.estimators_) == 1:
        # Only one estimator, but we still want to return probabilities
        # for two classes.
        Y = np.concatenate(((1 - Y), Y), axis=1)

    if not self.multilabel_:
        # Then, probabilities should be normalized to 1.
        Y /= np.sum(Y, axis=1)[:, np.newaxis]
    return Y
  
  def decision_function(self, X):
    """Decision function for the OneVsRestClassifier.

    Return the distance of each sample from the decision boundary for each
    class. This can only be used with estimators which implement the
    `decision_function` method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    T : array-like of shape (n_samples, n_classes) or (n_samples,) for \
        binary classification.
        Result of calling `decision_function` on the final estimator.

        .. versionchanged:: 0.19
            output shape changed to ``(n_samples,)`` to conform to
            scikit-learn conventions for binary classification.
    """
    check_is_fitted(self)
    if len(self.estimators_) == 1:
        return self.estimators_[0].decision_function(X)
    return np.array(
        [est.decision_function(X).ravel() for est in self.estimators_]
    ).T
  
  def multilabel_(self):
    """Whether this is a multilabel classifier."""
    return self.label_binarizer_.y_type_.startswith("multilabel")
  
  def n_classes_(self):
    """Number of classes."""
    return len(self.classes_)