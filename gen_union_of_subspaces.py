import numpy as np

from scipy.linalg import orth
from sklearn.preprocessing import normalize

def gen_union_of_subspaces(ambient_dim, subspace_dim, num_subspaces, num_points_per_subspace, noise_level=0.0):
    """This funtion generates a union of subspaces under random model, i.e., 
    subspaces are independently and uniformly distributed in the ambient space,
    data points are independently and uniformly distributed on the unit sphere of each subspace

    Parameters
    -----------
    ambient_dim : int
        Dimention of the ambient space
    subspace_dim : int
        Dimension of each subspace (all subspaces have the same dimension)
    num_subspaces : int
        Number of subspaces to be generated
    num_points_per_subspace : int
        Number of data points from each of the subspaces
    noise_level : float
        Amount of Gaussian noise on data
		
    Returns
    -------
    data : shape (num_subspaces * num_points_per_subspace) by ambient_dim
        Data matrix containing points drawn from a union of subspaces as its rows
    label : shape (num_subspaces * num_points_per_subspace)
        Membership of each data point to the subspace it lies in
    """

    data = np.empty((num_points_per_subspace* num_subspaces, ambient_dim))
    label = np.empty(num_points_per_subspace * num_subspaces, dtype=int)
  
    for i in range(num_subspaces):
        basis = np.random.normal(size=(ambient_dim, subspace_dim))
        basis = orth(basis)
        coeff = np.random.normal(size=(subspace_dim, num_points_per_subspace))
        coeff = normalize(coeff, norm='l2', axis=0, copy=False)
        data_per_subspace = np.matmul(basis, coeff).T

        base_index = i*num_points_per_subspace
        data[(0+base_index):(num_points_per_subspace+base_index), :] = data_per_subspace
        label[0+base_index:num_points_per_subspace+base_index,] = i

    data += np.random.normal(size=(num_points_per_subspace * num_subspaces, ambient_dim)) * noise_level
  
    return data, label
  

