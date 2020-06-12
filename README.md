# Subspace-Clustering
Toolbox for large scale subspace clustering

This project provides python implementation of the **elastic net subspace clustering (EnSC)** and the **sparse subspace clustering by orthogonal matching pursuit (SSC-OMP)** algorithms described in the following two papers:

- C. You, C.-G. Li, D. Robinson, R. Vidal, Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
- C. You, D. Robinson, R. Vidal, Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016

The clustering algorithms are implemented as two classes _ElasticNetSubspaceClustering_ and _SparseSubspaceClusteringOMP_ that have a fit function to learn the clusters. They may be used in the same way as the _KMeans_, _SpectralClustering_ and others that are in _sklearn.cluster_.

# Example
```
import numpy as np
from cluster.selfrepresentation import ElasticNetSubspaceClustering

# generate 7 data points from 3 independent subspaces as columns of data matrix X
X = np.array([[1.0, -1.0, 0.0, 0.0, 0.0,  0.0, 0.0],
              [1.0,  0.5, 0.0, 0.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 1.0, 0.2, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.2, 1.0, 0.0,  0.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
              [0.0,  0.0, 0.0, 0.0, 1.0,  1.0, -1.0]])

model = ElasticNetSubspaceClustering(n_clusters=3,algorithm='lasso_lars',gamma=50).fit(X.T)
print(model.labels_)
# this should give you array([1, 1, 0, 0, 2, 2, 2]) or a permutation of these labels
```

# Results on synthetic data
We compare EnSC and SSC-OMP with the k-means and spectral clustering algorithms on synthetically generated dataset where data is sampled from a union of subspaces.  

The following two figures report the clustering accuracy and running time as the scale of the dataset increases from 500 to 0.5 million.

![image](https://github.com/ChongYou/subspace-clustering/blob/master/figs/synthetic_acc.png)    ![image](https://github.com/ChongYou/subspace-clustering/blob/master/figs/synthetic_time.png)

EnSC and SSC-OMP not only achieves significantly higher clustering accuracy than K-means and spectral clustering but are also very efficient. 

To reproduce the results, run run_synthetic.py.

# Results on the MNIST digit dataset
We evaluate the performance of different methods for the task of clustering 70,000 (i.e., 60,000 training plus 10,000 testing) images from the MNIST dataset. The following table reports the clustering accuracy and running time.

|                         | KMeans | SpectralClustering | EnSC  | SSC-OMP |
|   --------              | ------ | ------------------ | ----- | ------- |
| Accuracy                | 53.52  | 73.38              | 97.62 | 92.79   |
| NMI                     | 49.74  | 86.74              | 93.69 | 84.25   |
| ARI                     | 36.30  | 74.53              | 94.84 | 84.91   |
| Running time (sec.)     | 50     | 1515               | 3620  | 1676    |

EnSC and SSC-OMP are able to achieve very high clustering accuracy within about an hour.

To reproduce the results, run run_mnist.py.

# Dependencies
numpy, scipy, scikit-learn

The SPAMS package (http://spams-devel.gforge.inria.fr/downloads.html) is recommended for faster computation. It may be used by setting algorithm='spams' in ElasticNetSubspaceClustering. On Ubuntu 16.04, SPAMS may be installed by the following commands:
```
sudo apt install liblapack-dev libopenblas-dev
pip install --index-url https://test.pypi.org/simple/ spams
```

The Kymatio package (https://www.kymat.io/) is required for running experiments on MNIST. It may be installed by
```
pip install kymatio
```
or by following the instructions on their webpage.

# Citing this work

```
@inproceedings{you2016scalable,
  title={Scalable sparse subspace clustering by orthogonal matching pursuit},
  author={You, Chong and Robinson, Daniel and Vidal, Ren{\'e}},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3918--3927},
  year={2016}
}
```
```
@inproceedings{you2016oracle,
  title={Oracle based active set algorithm for scalable elastic net subspace clustering},
  author={You, Chong and Li, Chun-Guang and Robinson, Daniel P and Vidal, Ren{\'e}},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3928--3937},
  year={2016}
}
```

