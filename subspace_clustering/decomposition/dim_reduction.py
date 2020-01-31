import numpy as np


def dim_reduction(X, dim):
    """Dimension reduction by principal component analysis
		Let X^T = U S V^T be the SVD of X^T in which the singular values are
	in ascending order. The output Xp^T is the last `dim` rows of S * V^T.
  
    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    dim: int
        Target dimension. 
		
    Returns
    -------
    Xp : shape (n_samples, dim)
        Dimension reduced data
	"""
    if dim == 0:
        return X

    w, v = np.linalg.eigh(X.T @ X)
  
    return X @ v[:, -dim:]
  
  
