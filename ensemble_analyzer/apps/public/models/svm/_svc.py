import scipy.sparse as sp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import _libsvm_sparse as libsvm_sparse
from sklearn.utils import column_or_1d, compute_class_weight,check_random_state
from sklearn.utils.validation import check_is_fitted,check_consistent_length,_num_samples
from sklearn.utils.multiclass import check_classification_targets
from sklearn.exceptions import NotFittedError

def _ovr_decision_function(predictions, confidences, n_classes):
	n_samples = predictions.shape[0]
	votes = np.zeros((n_samples, n_classes))
	sum_of_confidences = np.zeros((n_samples, n_classes))

	k = 0
	for i in range(n_classes):
		for j in range(i + 1, n_classes):
			sum_of_confidences[:, i] -= confidences[:, k]
			sum_of_confidences[:, j] += confidences[:, k]
			votes[predictions[:, k] == 0, i] += 1
			votes[predictions[:, k] == 1, j] += 1
			k += 1

  # Monotonically transform the sum_of_confidences to (-1/3, 1/3)
  # and add it with votes. The monotonic transformation  is
  # f: x -> x / (3 * (|x| + 1)), it uses 1/3 instead of 1/2
  # to ensure that we won't reach the limits and change vote order.
  # The motivation is to use confidence levels as a way to break ties in
  # the votes without switching any decision made based on a difference
  # of 1 vote.
	transformed_confidences = sum_of_confidences / (
			3 * (np.abs(sum_of_confidences) + 1)
	)
	return votes + transformed_confidences

LIBSVM_IMPL = ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]

class BaseSVC(BaseEstimator,ClassifierMixin):
	def __init__(
		self,
		kernel,
		degree,
		gamma,
		coef0,
		tol,
		C,
		nu,
		shrinking,
		probability,
		cache_size,
		class_weight,
		verbose,
		max_iter,
		decision_function_shape,
		random_state,
		break_ties,
	):
		self.decision_function_shape = decision_function_shape
		self.break_ties = break_ties
		self.kernel = kernel
		self.degree = degree
		self.gamma = gamma
		self.coef0 = coef0
		self.tol = tol
		self.C = C
		self.nu = nu
		self.shrinking = shrinking
		self.probability = probability
		self.cache_size = cache_size
		self.class_weight = class_weight
		self.verbose = verbose
		self.max_iter = max_iter
		self.random_state = random_state
		
	def _validate_targets(self, y):
		y_ = column_or_1d(y, warn=True)
		check_classification_targets(y)
		cls, y = np.unique(y_, return_inverse=True)
		self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
		if len(cls) < 2:
			raise ValueError(
					"The number of classes has to be greater than one; got %d class"
					% len(cls)
			)

		self.classes_ = cls

		return np.asarray(y, dtype=np.float64, order="C")

	def decision_function(self, X):
		dec = self._decision_function(X)
		if self.decision_function_shape == "ovr" and len(self.classes_) > 2:
			return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
		return dec
  
	def fit(self, X, y, sample_weight=None):
		rnd = check_random_state(self.random_state)
		sparse = sp.isspmatrix(X)
		
		if sparse and self.kernel == "precomputed":
					raise TypeError("Sparse precomputed kernels are not supported.")
		self._sparse = sparse and not callable(self.kernel)
  
		if callable(self.kernel):
			check_consistent_length(X, y)
		else:
			X, y = self._validate_data(
				X,
				y,
				dtype=np.float64,
				order="C",
				accept_sparse="csr",
				accept_large_sparse=False,
			)

		y = self._validate_targets(y)
    
		sample_weight = np.asarray(
			[] if sample_weight is None else sample_weight, dtype=np.float64
		)
		solver_type = LIBSVM_IMPL.index(self._impl)
    
    # input validation
		n_samples = _num_samples(X)
		if solver_type != 2 and n_samples != y.shape[0]:
			raise ValueError(
				"X and y have incompatible shapes.\n"
				+ "X has %s samples, but y has %s." % (n_samples, y.shape[0])
			)

		if self.kernel == "precomputed" and n_samples != X.shape[1]:
			raise ValueError(
				"Precomputed matrix must be a square matrix."
				" Input is a {}x{} matrix.".format(X.shape[0], X.shape[1])
			)

		if sample_weight.shape[0] > 0 and sample_weight.shape[0] != n_samples:
			raise ValueError(
				"sample_weight and X have incompatible shapes: "
				"%r vs %r\n"
				"Note: Sparse matrices cannot be indexed w/"
				"boolean masks (use `indices=True` in CV)."
				% (sample_weight.shape, X.shape)
			)
    
		kernel = "precomputed" if callable(self.kernel) else self.kernel
    
		if kernel == "precomputed":
			# unused but needs to be a float for cython code that ignores
			# it anyway
			self._gamma = 0.0
		elif isinstance(self.gamma, str):
			if self.gamma == "scale":
				# var = E[X^2] - E[X]^2 if sparse
				X_var = (X.multiply(X)).mean() - (X.mean()) ** 2 if sparse else X.var()
				self._gamma = 1.0 / (X.shape[1] * X_var) if X_var != 0 else 1.0
			elif self.gamma == "auto":
				self._gamma = 1.0 / X.shape[1]
			else:
				raise ValueError(
					"When 'gamma' is a string, it should be either 'scale' or "
					"'auto'. Got '{}' instead.".format(self.gamma)
				)
		else:
			self._gamma = self.gamma
		
		fit = self._sparse_fit if self._sparse else self._dense_fit
		if self.verbose:
			print("[LibSVM]", end="")
    
		seed = rnd.randint(np.iinfo("i").max)
		fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)
    
		self.shape_fit_ = X.shape if hasattr(X, "shape") else (n_samples,)
    
    # In binary case, we need to flip the sign of coef, intercept and
    # decision function. Use self._intercept_ and self._dual_coef_
    # internally.
		self._intercept_ = self.intercept_.copy()
		self._dual_coef_ = self.dual_coef_
		if self._impl in ["c_svc", "nu_svc"] and len(self.classes_) == 2:
			self.intercept_ *= -1
			self.dual_coef_ = -self.dual_coef_

		return self
  
	def _sparse_fit(self, X, y, sample_weight, solver_type, kernel, random_seed):
		X.data = np.asarray(X.data, dtype=np.float64, order="C")
		X.sort_indices()

		kernel_type = self._sparse_kernels.index(kernel)

		libsvm_sparse.set_verbosity_wrap(self.verbose)

		(
			self.support_,
			self.support_vectors_,
			dual_coef_data,
			self.intercept_,
			self._n_support,
			self._probA,
			self._probB,
			self.fit_status_,
		) = libsvm_sparse.libsvm_sparse_train(
			X.shape[1],
			X.data,
			X.indices,
			X.indptr,
			y,
			solver_type,
			kernel_type,
			self.degree,
			self._gamma,
			self.coef0,
			self.tol,
			self.C,
			self.class_weight_,
			sample_weight,
			self.nu,
			self.cache_size,
			self.epsilon,
			int(self.shrinking),
			int(self.probability),
			self.max_iter,
			random_seed,
		)

		self._warn_from_fit_status()

		if hasattr(self, "classes_"):
				n_class = len(self.classes_) - 1
		else:  # regression
				n_class = 1
		n_SV = self.support_vectors_.shape[0]

		dual_coef_indices = np.tile(np.arange(n_SV), n_class)
		if not n_SV:
				self.dual_coef_ = sp.csr_matrix([])
		else:
			dual_coef_indptr = np.arange(
					0, dual_coef_indices.size + 1, dual_coef_indices.size / n_class
			)
			self.dual_coef_ = sp.csr_matrix(
					(dual_coef_data, dual_coef_indices, dual_coef_indptr), (n_class, n_SV)
			)

	def predict(self, X):
		check_is_fitted(self)
		if self.break_ties and self.decision_function_shape == "ovo":
			raise ValueError(
					"break_ties must be False when decision_function_shape is 'ovo'"
			)

		if (
			self.break_ties
			and self.decision_function_shape == "ovr"
			and len(self.classes_) > 2
		):
			y = np.argmax(self.decision_function(X), axis=1)
		else:
			y = super().predict(X)
		return self.classes_.take(np.asarray(y, dtype=np.intp))
  
	def predict_proba(self, X):
		X = self._validate_for_predict(X)
		if self.probA_.size == 0 or self.probB_.size == 0:
				raise NotFittedError(
						"predict_proba is not available when fitted with probability=False"
				)
		pred_proba = (
				self._sparse_predict_proba if self._sparse else self._dense_predict_proba
		)
		return pred_proba(X)

class SVC(BaseSVC):
  _impl = "c_svc"

  def __init__(
		self,
		*,
		C=1.0,
		kernel="rbf",
		degree=3,
		gamma="scale",
		coef0=0.0,
		shrinking=True,
		probability=False,
		tol=1e-3,
		cache_size=200,
		class_weight=None,
		verbose=False,
		max_iter=-1,
		decision_function_shape="ovr",
		break_ties=False,
		random_state=None,
  ):
			super().__init__(
				kernel=kernel,
				degree=degree,
				gamma=gamma,
				coef0=coef0,
				tol=tol,
				C=C,
				nu=0.0,
				shrinking=shrinking,
				probability=probability,
				cache_size=cache_size,
				class_weight=class_weight,
				verbose=verbose,
				max_iter=max_iter,
				decision_function_shape=decision_function_shape,
				break_ties=break_ties,
				random_state=random_state,
	)
	
def _more_tags(self):
	return {
		"_xfail_checks": {
			"check_sample_weights_invariance": (
				"zero sample_weight is not equivalent to removing samples"
			),
		}
	}