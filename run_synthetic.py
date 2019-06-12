import numpy as np
import sys
import time

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from gen_union_of_subspaces import gen_union_of_subspaces
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster

# =================================================
# Generate dataset where data is drawn from a union of subspaces
# =================================================
ambient_dim = 9
subspace_dim = 6
num_subspaces = 5
num_points_per_subspace = 50

data, label = gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, 0.00)

# =================================================
# Create cluster objects
# =================================================

# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=num_subspaces)  # k-means as baseline
model_spectral = cluster.SpectralClustering(n_clusters=num_subspaces,affinity='nearest_neighbors',n_neighbors=6)  # spectral clustering as baseline

# Elastic net subspace clustering with a scalable active support elastic net solver
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=num_subspaces,algorithm='spams',gamma=500)

# Sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=num_subspaces,n_nonzero=subspace_dim,thr=1e-5)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC', model_ensc),
    ('SSC-OMP', model_ssc_omp)
)

for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(data)
    t_end = time.time()
    accuracy = clustering_accuracy(label, algorithm.labels_)

    print('Algorithm: {}. Clustering accuracy: {}. Running time: {}'.format(name, accuracy, t_end - t_begin))




