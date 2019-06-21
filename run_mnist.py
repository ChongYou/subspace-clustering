import numpy as np
import time

import torch

from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from decomposition.dim_reduction import dim_reduction
from kymatio import Scattering2D
from metrics.cluster.accuracy import clustering_accuracy
from sklearn import cluster
from sklearn.preprocessing import normalize
from torchvision import datasets, transforms

# =================================================
# Prepare MNIST dataset
# =================================================

print('Prepare scattering...')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

scattering = Scattering2D(J=3, shape=(32, 32))
if use_cuda:
    scattering = scattering.cuda()

print('Prepare MNIST...')
transforms_MNIST = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
MNIST_train = datasets.MNIST('./data', train=True, download=True, transform=transforms_MNIST)
MNIST_test = datasets.MNIST('./data', train=False, download=True, transform=transforms_MNIST)
MNIST_train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=60000, shuffle=False)
MNIST_test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=10000, shuffle=False)

raw_train_data, label_train = next(iter(MNIST_train_loader))  # data shape: torch.Size([60000, 1, 28, 28])
raw_test_data, label_test = next(iter(MNIST_test_loader))  # data shape: torch.Size([10000, 1, 28, 28])
label = torch.cat((label_train, label_test), 0)

print('Computing scattering on MNIST...')
if use_cuda:
    raw_train_data = raw_train_data.cuda()
    raw_test_data = raw_test_data.cuda()

train_data = scattering(raw_train_data) 
test_data = scattering(raw_test_data)
data = torch.cat((train_data, test_data), 0)

print('Data preprocessing....')
n_sample = data.shape[0]

# scattering transform normalization
data = data.cpu().numpy().reshape(n_sample, data.shape[2], -1)
image_norm = np.linalg.norm(data, ord=np.inf, axis=2, keepdims=True)  # infinity norm of each transform
data = data / image_norm  # normalize each scattering transform to the range [-1, 1]
data = data.reshape(n_sample, -1)  # fatten and concatenate all transforms

# dimension reduction
data = dim_reduction(data, 500)  # dimension reduction by PCA

label = label.numpy()          

# =================================================
# Create cluster objects
# =================================================
print('Begin clustering...')

# Baseline: non-subspace clustering methods
model_kmeans = cluster.KMeans(n_clusters=10)  # k-means as baseline

model_spectral = cluster.SpectralClustering(n_clusters=10,affinity='nearest_neighbors',n_neighbors=5)  # spectral clustering as baseline

# Our work: elastic net subspace clustering (EnSC)
# You et al., Oracle Based Active Set Algorithm for Scalable Elastic Net Subspace Clustering, CVPR 2016
model_ensc = ElasticNetSubspaceClustering(n_clusters=10,affinity='nearest_neighbors',algorithm='spams',active_support=True,gamma=200,tau=0.9)

# Our work: sparse subspace clusterign by orthogonal matching pursuit (SSC-OMP)
# You et al., Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit, CVPR 2016
model_ssc_omp = SparseSubspaceClusteringOMP(n_clusters=10,affinity='symmetrize',n_nonzero=5,thr=1.0e-5)

clustering_algorithms = (
    ('KMeans', model_kmeans),
    ('Spectral Clustering', model_spectral),
    ('EnSC via active support solver', model_ensc),
    ('SSC-OMP', model_ssc_omp),
)

for name, algorithm in clustering_algorithms:
    t_begin = time.time()
    algorithm.fit(data)
    t_end = time.time()
    accuracy = clustering_accuracy(label, algorithm.labels_)

    print('Algorithm: {}. Clustering accuracy: {}. Running time: {}'.format(name, accuracy, t_end - t_begin))
