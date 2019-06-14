import warnings
import math
import numpy as np
import spams
import time

from scipy import sparse
from sklearn import cluster
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import sparse_encode
from sklearn.linear_model import orthogonal_mp
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.utils import check_random_state, check_array, check_symmetric


class SelfRepresentation(BaseEstimator, ClusterMixin):
    """Base class for self-representation based subspace clustering

    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``
    """

    def __init__(self, n_clusters=8, random_state=None, n_init=20, n_jobs=1):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute representation matrix, then apply spectral clustering
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        self._representation_to_affinity()
        self._spectral_clustering()
        self.timer_time_ = time.time() - time_base

        return self
	
    def fit_self_representation(self, X, y=None):
        """Compute representation matrix without apply spectral clustering
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'], dtype=np.float64)
        time_base = time.time()
        
        self._self_representation(X)
        self.timer_self_representation_ = time.time() - time_base
        
        return self

    def _representation_to_affinity(self):
        """Compute affinity matrix from representation matrix as A = (|C|+|C'|)/2
        """
        self.affinity = (np.absolute(self.representation_matrix_) + np.absolute(self.representation_matrix_.T)) / 2.0

        # neighbors_graph = kneighbors_graph(self.representation_matrix_, 5, mode='connectivity', include_self=False)
        # self.affinity = neighbors_graph + neighbors_graph.T

    def _spectral_clustering(self):
        self.labels_ = spectral_clustering(self.affinity, self.n_clusters, norm_laplacian=True, random_state=self.random_state, n_init=self.n_init)


def active_support_elastic_net(X, y, alpha, tau=1.0, algorithm='spams', support_init='knn', support_size=100, maxiter=40):
    """An active support based algorithm for solving the elastic net optimization problem
        min_{c} tau ||c||_1 + (1-tau)/2 ||c||_2^2 + alpha / 2 ||y - c X ||_2^2
		
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (1, n_features)
    alpha : float
    tau : float, default 1.0
    algorithm : string, default ``spams``
        Algorithm for computing solving the subproblems. Either lasso_lars or lasso_cd or spams 
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    support_init: string, default ``knn``
        This determines how the active support is initialized
        It can be either ``knn`` or ``L2``
    support_size: int, default 100
        This determines the size of working set
        A small support_size decreases the runtime per iteration while increase the number of iterations
    maxiter: int default 40
        Termination condition for active support update
		
    Returns
    -------
    c : shape n_samples
        The optimal solution to the optimization problem
	"""
    n_samples = X.shape[0]

    if n_samples <= support_size:  # skip active support search for small scale data
        supp = np.arange(n_samples, dtype=int)  # this results in the following iteration to converge in 1 iteration
    else:    
        if support_init == 'L2':
            L2sol = np.linalg.solve(np.identity(y.shape[1]) * alpha + np.dot(X.T, X), y.T)
            c0 = np.dot(X, L2sol)[:, 0]
            supp = np.argpartition(-np.abs(c0), support_size)[0:support_size]
        elif support_init == 'knn':
            supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]

    curr_obj = float("inf")
    for _ in range(maxiter):
        Xs = X[supp, :]
        if algorithm == 'spams':
            cs = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(Xs.T), lambda1=tau*alpha, lambda2=(1.0-tau)*alpha)
            cs = np.asarray(cs.todense()).T
        else:
            cs = sparse_encode(y, Xs, algorithm=algorithm, alpha=alpha)
      
        delta = (y - np.dot(cs, Xs)) / alpha
		
        obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau)/2.0 * np.sum(np.power(cs[0], 2.0)) + alpha/2.0 * np.sum(np.power(delta, 2.0))
        if curr_obj - obj < 1.0e-10 * curr_obj:
            break
        curr_obj = obj
			
        coherence = np.abs(np.dot(delta, X.T))[0]
        coherence[supp] = 0
        addedsupp = np.nonzero(coherence > tau + 1.0e-10)[0]
        
        if addedsupp.size == 0:  # converged
            break

        # Find the set of nonzero entries of cs.
        activesupp = supp[np.abs(cs[0]) > 1.0e-10]  
        
        if activesupp.size > 0.8 * support_size:  # this suggests that support_size is too small and needs to be increased
            support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])
        
        if addedsupp.size + activesupp.size > support_size:
            ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[0:support_size - activesupp.size]
            addedsupp = addedsupp[ord]
        
        supp = np.concatenate([activesupp, addedsupp])
    
    c = np.zeros(n_samples)
    c[supp] = cs
    return c

  
def spectral_clustering(affinity, n_clusters, norm_laplacian=True, random_state=None, n_init=20):
    """Spectral clustering.
    This is a simplified version of spectral_clustering in sklearn. 
    """
    affinity = check_symmetric(affinity)
    random_state = check_random_state(random_state)
    
    laplacian = sparse.csgraph.laplacian(affinity, normed=norm_laplacian)
    _, vec = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, k=n_clusters, sigma=None, which='LA')
    embedding = normalize(vec)
    _, labels, _ = cluster.k_means(embedding, n_clusters, random_state=random_state, n_init=n_init)
    return  labels


def elastic_net_subspace_clustering(X, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars', n_nonzero=50, active_support=True, support_init='knn', support_size=100):
    """Elastic net subspace clustering (EnSC) [1]. 
    Compute self-representation matrix C from solving the following optimization problem
    min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,
    where c_j and x_j are the j-th rows of C and X, respectively.
	
	Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.
	``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, 
    ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in 
	spams package available at http://spams-devel.gforge.inria.fr/ (installation required)
    In principle, all three algorithms give the same result.	
    For large scale data (e.g. with > 5000 data points), use any of these algorithms in
	conjunction with ``active_support=True``. It adopts an efficient active support 
	strategy that solves the optimization problem by breaking it into a sequence of 
    small scale optimization problems as described in [1].

    If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
    If tau = 0.0, the method reduces to least squares regression (LSR) [3].
	Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    gamma : float
    gamma_nz : boolean, default True
        gamma and gamma_nz together determines the parameter alpha. If gamma_nz = False, then
        alpha = gamma. If gamma_nz = True, then alpha = gamma * alpha0, where alpha0 is the largest 
        number that the solution to the optimization problem with alpha = alpha0 is zero vector
        (see Proposition 1 in [1]). 
    tau : float, default 1.0
        Parameter for elastic net penalty term. 
        When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
        When tau = 0.0, the method reduces to least squares regression (LSR) [3].
    algorithm : string, default ``lasso_lars``
        Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams 
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    n_nonzero : int, default 50
        This is an upper bound on the number of nonzero entries of each representation vector. 
        If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of
        entries with largest absolute value are kept.
    active_support: boolean, default True
        Set to True to use the active support algorithm in [1] for solving the optimization problem.
        This should significantly reduce the running time when n_samples is large.
    support_init: string, default ``knn``
        This determines how the active support is initialized for the active support algorithm. 
        It can be either ``knn`` or ``L2``. 
        Ignored when active_support is set to False
    support_size: int, default 100
        This determines the size of working set in the active support algorithm. 
        A small support_size decreases the runtime per iteration while increase the number of iterations.
        Ignored when active_support is set to False
	
    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------	
	[1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
	[2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013
    [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    if algorithm in ('lasso_lars', 'lasso_cd') and tau < 1.0 - 1.0e-10:  
        warnings.warn('algorithm cannot handle tau smaller than 1. Using tau = 1')
        tau = 1.0

    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero)
    cols = np.zeros(n_samples * n_nonzero)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0
 
    for i in range(n_samples):
    #    if i % 1000 == 999:
    #        print('SSC: sparse coding finished {i} in {n_samples}'.format(i=i, n_samples=n_samples))
        y = X[i, :].copy().reshape(1, -1)
        X[i, :] = 0
        
        if algorithm in ('lasso_lars', 'lasso_cd', 'spams'):
            if gamma_nz == True:
                coh = np.delete(np.absolute(np.dot(X, y.T)), i)
                alpha0 = np.amax(coh) / tau  # value for which the solution is zero
                alpha = alpha0 / gamma
            else:
                alpha = 1.0 / gamma

            if active_support == True:
                c = active_support_elastic_net(X, y, alpha, tau, algorithm, support_init, support_size)
            else:
                if algorithm == 'spams':
                    c = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(X.T), lambda1=tau * alpha, lambda2=(1.0-tau) * alpha)
                    c = np.asarray(c.todense()).T[0]
                else:
                    c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha)[0]
        else:
          warnings.warn("algorithm not found")
	    	  
        index = np.flatnonzero(c)
        if index.size > n_nonzero:
        #  warnings.warn("The number of nonzero entries in sparse subspace clustering exceeds n_nonzero")
          index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
        rows[curr_pos:curr_pos + len(index)] = i
        cols[curr_pos:curr_pos + len(index)] = index
        vals[curr_pos:curr_pos + len(index)] = c[index]
        curr_pos += len(index)
        
        X[i, :] = y

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))


class ElasticNetSubspaceClustering(SelfRepresentation):
    """Elastic net subspace clustering (EnSC) [1]. 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following elastic net problem
    min_{c_j} tau ||c_j||_1 + (1-tau)/2 ||c_j||_2^2 + alpha / 2 ||x_j - c_j X ||_2^2 s.t. c_jj = 0,
    where c_j and x_j are the j-th rows of C and X, respectively.
	
	Parameter ``algorithm`` specifies the algorithm for solving the optimization problem.
	``lasso_lars`` and ``lasso_cd`` are algorithms implemented in sklearn, 
    ``spams`` refers to the same algorithm as ``lasso_lars`` but is implemented in 
	spams package available at http://spams-devel.gforge.inria.fr/ (installation required)
    In principle, all three algorithms give the same result.	
    For large scale data (e.g. with > 5000 data points), use any of these algorithms in
	conjunction with ``active_support=True``. It adopts an efficient active support 
	strategy that solves the optimization problem by breaking it into a sequence of 
    small scale optimization problems as described in [1].

    If tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
    If tau = 0.0, the method reduces to least squares regression (LSR) [3].
	Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.

    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    gamma : float
    gamma_nz : boolean, default True
        gamma and gamma_nz together determines the parameter alpha. If gamma_nz = False, then
        alpha = gamma. If gamma_nz = True, then alpha = gamma * alpha0, where alpha0 is the largest 
        number that the solution to the optimization problem with alpha = alpha0 is zero vector
        (see Proposition 1 in [1]). 
    tau : float, default 1.0
        Parameter for elastic net penalty term. 
        When tau = 1.0, the method reduces to sparse subspace clustering with basis pursuit (SSC-BP) [2].
        When tau = 0.0, the method reduces to least squares regression (LSR) [3].
    algorithm : string, default ``lasso_lars``
        Algorithm for computing the representation. Either lasso_lars or lasso_cd or spams 
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    n_nonzero : int, default 50
        This is an upper bound on the number of nonzero entries of each representation vector. 
        If there are more than n_nonzero nonzero entries,  only the top n_nonzero number of
        entries with largest absolute value are kept.
    active_support: boolean, default True
        Set to True to use the active support algorithm in [1] for solving the optimization problem.
        This should significantly reduce the running time when n_samples is large.
    support_init: string, default ``knn``
        This determines how the active support is initialized for the active support algorithm. 
        It can be either ``knn`` or ``L2``. 
        Ignored when active_support is set to False
    support_size: int, default 100
        This determines the size of working set in the active support algorithm. 
        A small support_size decreases the runtime per iteration while increase the number of iterations.
        Ignored when active_support is set to False
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``

    References
    -----------	
	[1] C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
	[2] E. Elhaifar, R. Vidal, Sparse Subspace Clustering: Algorithm, Theory, and Applications, TPAMI 2013
    [3] C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    def __init__(self, n_clusters=8, random_state=None, n_init=20, n_jobs=1, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars', n_nonzero=50, active_support=True, support_init='knn', support_size=100):
        self.gamma = gamma
        self.gamma_nz = gamma_nz
        self.tau = tau
        self.algorithm = algorithm
        self.n_nonzero = n_nonzero
        self.active_support = active_support
        self.support_size = support_size
        self.support_init = support_init
        SelfRepresentation.__init__(self, n_clusters, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = elastic_net_subspace_clustering(X, self.gamma, self.gamma_nz, self.tau, self.algorithm, self.n_nonzero, 
		                                                              self.active_support, self.support_init, self.support_size)
					

def sparse_subspace_clustering_orthogonal_matching_pursuit(X, n_nonzero=10, thr=1.0e-6):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)
    Compute self-representation matrix C by solving the following optimization problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	

    Returns
    -------
    representation_matrix_ : csr matrix, shape: n_samples by n_samples
        The self-representation matrix.
	
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """	
    n_samples = X.shape[0]
    rows = np.zeros(n_samples * n_nonzero, dtype = int)
    cols = np.zeros(n_samples * n_nonzero, dtype = int)
    vals = np.zeros(n_samples * n_nonzero)
    curr_pos = 0
 
    for i in range(n_samples):
        residual = X[i, :].copy()  # initialize residual
        supp = np.empty(shape=(0), dtype = int)  # initialize support
        residual_norm_thr = np.linalg.norm(X[i, :]) * thr
        for t in range(n_nonzero):  # for each iteration of OMP  
            # compute coherence between residuals and X     
            coherence = abs( np.matmul(residual, X.T) )
            coherence[i] = 0.0
            # update support
            supp = np.append(supp, np.argmax(coherence))
            # compute coefficients
            c = np.linalg.lstsq( X[supp, :].T, X[i, :].T, rcond=None)[0]
            # compute residual
            residual = X[i, :] - np.matmul(c.T, X[supp, :])
            # check termination
            if np.sum(residual **2) < residual_norm_thr:
                break

        rows[curr_pos:curr_pos + len(supp)] = i
        cols[curr_pos:curr_pos + len(supp)] = supp
        vals[curr_pos:curr_pos + len(supp)] = c
        curr_pos += len(supp)

#   affinity = sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples)) + sparse.csr_matrix((vals, (cols, rows)), shape=(n_samples, n_samples))
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
					

class SparseSubspaceClusteringOMP(SelfRepresentation):
    """Sparse subspace clustering by orthogonal matching pursuit (SSC-OMP). 
    This is a self-representation based subspace clustering method that computes
    the self-representation matrix C via solving the following problem
    min_{c_j} ||x_j - c_j X ||_2^2 s.t. c_jj = 0, ||c_j||_0 <= n_nonzero
    via OMP, where c_j and x_j are the j-th rows of C and X, respectively

    Parameters
    -----------
    n_clusters : integer, optional, default: 8
        Number of clusters in the dataset.
    random_state : int, RandomState instance or None, optional, default: None
        This is the random_state parameter for k-means. 
    n_init : int, optional, default: 10
        This is the n_init parameter for k-means. 
    n_nonzero : int, default 10
        Termination condition for omp.
    thr : float, default 1.0e-5
        Termination condition for omp.	
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``

    References
    -----------	
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    def __init__(self, n_clusters=8, random_state=None, n_init=10, n_jobs=1, n_nonzero=10, thr=1.0e-6):
        self.n_nonzero = n_nonzero
        self.thr = thr
        SelfRepresentation.__init__(self, n_clusters, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = sparse_subspace_clustering_orthogonal_matching_pursuit(X, self.n_nonzero, self.thr)


def least_squares_subspace_clustering(X, gamma=10.0, exclude_self=False):
    """Least squares subspace clustering. 
    Compute self-representation matrix C by solving the following optimization problem
        min_{c_j} ||c_j||_2^2 + gamma ||x_j - X c_j||_2^2 s.t. c_jj = 0  (*)

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data to be clustered
    gamma : float
        Parameter on noise regularization term
    exclude_self : boolean, default False
        When False, solves (*) without the constraint c_jj = 0
		
    Returns
    -------
    representation_matrix_ : shape n_samples by n_samples
        The self-representation matrix.
		
    References
    -----------	
    C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    n_samples, n_features = X.shape
  
    if exclude_self == False:
        if n_samples < n_features:
            gram = np.matmul(X, X.T)
            return np.linalg.solve(gram + np.eye(n_sample) / gamma, gram).T
        else:
            tmp = np.linalg.solve(np.matmul(X.T, X) + np.eye(n_features) / gamma, X.T)
            return np.matmul(X, tmp).T
    else:
        if n_samples < n_features:
            D = np.linalg.solve(np.matmul(X, X.T) + np.eye(n_sample) / gamma, np.eye(n_sample))  # see Theorem 6 in https://arxiv.org/pdf/1404.6736.pdf
        else:
            tmp = np.linalg.solve(np.matmul(X.T, X) + np.eye(n_features) / gamma, X.T)
            D = eye(n_samples) - np.matmul(X, tmp)
        D = D / D.diagonal()[None,:]
        np.fill_diagonal(D, 0.0)
        return -1.0 * D.T


class LeastSquaresSubspaceClustering(SelfRepresentation):
    """Least squares subspace clustering.
	
	Parameters
    -----------
    gamma : float
        Parameter on noise regularization term
    exclude_self : boolean, default False
        When False, solves (*) without the constraint c_jj = 0
	
    Attributes
    ----------
    representation_matrix_ : array-like, shape (n_samples, n_samples)
        Self-representation matrix. Available only if after calling
        ``fit`` or ``fit_self_representation``.
    labels_ :
        Labels of each point. Available only if after calling ``fit``
		
    References
    -----------	
    C. Lu, et al. Robust and efficient subspace segmentation via least squares regression, ECCV 2012
    """
    def __init__(self, n_clusters=8, random_state=None, n_init=None, n_jobs=1, gamma=10.0, exclude_self=False):
        self.gamma = gamma
        self.exclude_self = exclude_self
        SelfRepresentation.__init__(self, n_clusters, random_state, n_init, n_jobs)
    
    def _self_representation(self, X):
        self.representation_matrix_ = least_squares_subspace_clustering(X, self.gamma, self.exclude_self)
