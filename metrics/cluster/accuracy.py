from scipy import sparse
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import _supervised
from sklearn.preprocessing import normalize

import numpy as np
import warnings

def clustering_accuracy(labels_true, labels_pred):
    """Clustering Accuracy between two clusterings.
    Clustering Accuracy is a measure of the similarity between two labels of
    the same data. Assume that both labels_true and labels_pred contain n 
    distinct labels. Clustering Accuracy is the maximum accuracy over all
    possible permutations of the labels, i.e.
    \max_{\sigma} \sum_i labels_true[i] == \sigma(labels_pred[i])
    where \sigma is a mapping from the set of unique labels of labels_pred to
    the set of unique labels of labels_true. Clustering accuracy is one if 
    and only if there is a permutation of the labels such that there is an
    exact match
    This metric is independent of the absolute values of the labels:
    a permutation of the class or cluster label values won't change the
    score value in any way.
    This metric is furthermore symmetric: switching ``label_true`` with
    ``label_pred`` will return the same score value. This can be useful to
    measure the agreement of two independent label assignments strategies
    on the same dataset when the real ground truth is not known.
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    labels_pred : array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    
    Returns
    -------
    accuracy : float
       return clustering accuracy in the range of [0, 1]
    """
    labels_true, labels_pred = _supervised.check_clusterings(labels_true, labels_pred)
    # value = _supervised.contingency_matrix(labels_true, labels_pred, sparse=False)
    value = _supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)
  
 
def self_representation_loss(labels_true, representation_matrix):
    """Self-representation error for self-expressive subspace clustering.
    Self-representation error is a measure of whether the representation 
    matrix has the property that the i,j-th entry is nonzero only if the
    i-th and j-th data are from the same class according to labels_true.
    Denote vec_i the i-th representation vector (i.e., the i-th row of the 
    representation_matrix). Self representation loss is computed as the 
    fraction of the L1 norm of vec_i that comes from wrong classes, averaged
    over all i. That is,
    \sum_i ( 1 - \sum_j(w_i[j] * |vec_i[j]|) / \sum_j(|vec_i[j]|) ) / n_samples
    where w_i[j] is the true affinity, i.e., w_i[j] = 1 if labels_true[i] ==
    labels_true[j]. 
    This metric takes value zero iff there is no false connections in
    representation matrix, and one iff here is no correct connections.
    For more details, see [1].
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    representation_matrix : array, shape = [n_samples, n_samples]
    	Each row is a representation vector.
    
    Returns
    -------
    loss : float
       return self_representation_loss in the range of [0, 1].
       
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    n_samples = labels_true.shape[0]
    loss = 0.0
    for i in range(n_samples):
        representation_vec = np.abs(representation_matrix[i, :])
        mask = (labels_true != labels_true[i]).reshape(1, -1)
        loss += np.sum(representation_vec[mask]) / (np.sum(representation_vec) +1e-10)
    
    return loss / n_samples
    

def self_representation_sparsity(representation_matrix):
    """Evaluation of sparsity for self-expressive subspace clustering methods.
    Parameters
    ----------
    representation_matrix : array, shape = [n_samples, n_samples]
    	Each row is a representation vector
    
    Returns
    -------
    sparsity : float
       return averaged sparsity in the range of [0, n_samples]
    """
    return representation_matrix.count_nonzero() / representation_matrix.shape[0]
    
    
def self_representation_connectivity(labels_true, representation_matrix):
    """Connectivity for self-expressive subspace clustering.
    Connectivity is a measure of how well points within each class is connected
    to each other according to the representation_matrix.
    Let mat_i be the submatrix of the representation_matrix coorespondig to points
    in the i-th class, i.e., 
    mat_i = representation_matrix[labels_true==i, labels_true==i].
    Connectivity is computed as the algebraic connectivity of class i, defined as
    the second smallest eigenvalue lambda_i(2) of the graph Laplacian associated
    with the weight matrix |mat_i| + |mat_i|^T, minimized over all i, i.e.,
    connectivity = min_i lambda_i(2).
    Connectivity is zero iff for any of the classes, the representation_matrix 
    associated with that class is not fully connected.
    Connectivity is a complementary measurement to self_representation_loss for
    evaluating the quality of a representation matrix. In principle, a correct 
    clustering can be obtained when self_representation_loss is not large and 
    self_representation_connectivity is not too small.
    For more details, see [1].
    
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
    	A clustering of the data into disjoint subsets.
    representation_matrix : array, shape = [n_samples, n_samples]
    	Each row is a representation vector.
    
    Returns
    -------
    connectivity : float
       return connectivity in the range of [0, 1].
       
    References
    -----------			
    C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
    """
    connectivity = 1.
    normalized_representation_matrix_ = normalize(representation_matrix, 'l2')
    for i in np.unique(labels_true):
        mask = (labels_true == i)
        class_representation_matrix = normalized_representation_matrix_[mask, :]
        class_representation_matrix = class_representation_matrix[:, mask]
        try:
            class_affinity_matrix_ = 0.5 * (np.absolute(class_representation_matrix) + np.absolute(class_representation_matrix.T))
            laplacian = sparse.csgraph.laplacian(class_affinity_matrix_, normed=True)
        

            val = sparse.linalg.eigsh(sparse.identity(laplacian.shape[0]) - laplacian, 
                                      k=2, sigma=None, which='LA', return_eigenvectors=False)  
        except Exception as e:
            print(e)
            val = [1.0]       
                   
        connectivity = min(connectivity, 1.0 - val[0])
    return connectivity