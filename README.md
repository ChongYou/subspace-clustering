# Subspace-Clustering
Toolbox for large scale sparse subspace clustering

This project provides python implementation of large scale subspace clustering algorithms described in the following papers:

- C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
- C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016

The clustering algorithms are implemented as a class called SparseSubspaceClustering that has a fit function to learn the clusters, in a way that is compatible with the sklearn.cluster module in scikit-learn. As such, SparseSubspaceClustering may be used in the same way as the K-Means, SpectralClustering and others that are in sklearn.cluster.

# Example
```
import numpy as np
from cluster.selfrepresentation import SparseSubspaceClustering

# generate 7 data points from 3 independent subspaces as columns of data matrix X
X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
              [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

model = SparseSubspaceClustering(n_clusters=3,algorithm='lasso_cd',active_support=False, gamma=50).fit(X.T)
print(model.labels_)
# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labels
```

See also run_synthetic.py for extensive experiments on synthetically generated data

# Dependencies
numpy, scipy, scikit-learn

The SPAMS package (http://spams-devel.gforge.inria.fr/downloads.html) is optional for faster computation. On Ubuntu 16.04, SPAMS may be installed by the following commands:
```
sudo apt install liblapack-dev libopenblas-dev
pip install --index-url https://test.pypi.org/simple/ spams
```
