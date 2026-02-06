import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn

import operator
from functools import reduce
from functools import partial

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path, mode='r')
            self.old_mat = False
        # self.data = h5py.File(self.file_path, mode='r')
        # self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        print('in1')
        x = self.data[field]
        print('in2')
        if not self.old_mat:
            print('x', x.shape)
            x = np.transpose(x, (3, 1, 2, 0))
            # x = x[()]
            # x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        print('in3')
        if self.to_float:
            x = x.astype(np.float32)
        print('in4')
        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

# MeshDataset
from pathlib import Path
from timeit import default_timer
from typing import List, Union

import numpy as np

# import open3d for io if built. Otherwise,
# the class will build, but no files will be loaded.
try:
    import open3d as o3d

    o3d_warn = False
except ModuleNotFoundError:
    o3d_warn = True

import torch
from torch.utils.data import DataLoader

from typing import List

from torch.utils.data import Dataset


class DictDataset(Dataset):
    """DictDataset is a basic dataset form that stores each batch
    as a dictionary of tensors or other data structures


    """

    def __init__(
        self,
        data_list: List[dict],
        constant: dict = None,
    ):
        """

        Parameters
        ----------
        data_list : List[dict]
            list of individual batch dictionaries
        constant : dict, optional
            if each data batch shares some constant valued key/val pairs,
            they can be stored in constant for simplicity
        """

        self.data_list = data_list
        self.constant = constant

    def __getitem__(self, index):
        return_dict = self.data_list[index]

        if self.constant is not None:
            return_dict.update(self.constant)

        return return_dict

    def __len__(self):
        return len(self.data_list)

from typing import Dict

from math import prod
def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2 * n_params
    return n_params

from abc import abstractmethod, ABCMeta
from typing import List

import torch


class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to
    model inputs or outputs, respectively
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass
    
import torch

class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.

    Parameters
    ----------
    mean : torch.tensor or None
        has to include batch-size as a dim of 1
        e.g. for tensors of shape ``(batch_size, channels, height, width)``,
        the mean over height and width should have shape ``(1, channels, 1, 1)``
    std : torch.tensor or None
    eps : float, default is 0
        for safe division by the std
    dim : int list, default is None
        if not None, dimensions of the data to reduce over to compute the mean and std.

        .. important::

            Has to include the batch-size (typically 0).
            For instance, to normalize data of shape ``(batch_size, channels, height, width)``
            along batch-size, height and width, pass ``dim=[0, 2, 3]``

    mask : torch.Tensor or None, default is None
        If not None, a tensor with the same size as a sample,
        with value 0 where the data should be ignored and 1 everywhere else

    Notes
    -----
    The resulting mean will have the same size as the input MINUS the specified dims.
    If you do not specify any dims, the mean and std will both be scalars.
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        """Initialize the UnitGaussianNormalizer.

        See class docstring for detailed parameter descriptions.
        """
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            # print(samples.shape)
            # if batch_size == 1:
            #     samples = samples.unsqueeze(0)
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        if self.mask is None:
            self.n_elements = count_tensor_params(data_batch, self.dim)
            self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True)
            self.squared_mean = torch.mean(data_batch**2, dim=self.dim, keepdim=True)
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)
        else:
            batch_size = data_batch.shape[0]
            dim = [i - 1 for i in self.dim if i]
            shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
            self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size
            self.mean = torch.zeros(shape)
            self.std = torch.zeros(shape)
            self.squared_mean = torch.zeros(shape)
            data_batch[:, self.mask == 1] = 0
            self.mean[self.mask == 1] = (
                torch.sum(data_batch, dim=dim, keepdim=True) / self.n_elements
            )
            self.squared_mean = (
                torch.sum(data_batch**2, dim=dim, keepdim=True) / self.n_elements
            )
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            n_elements = count_tensor_params(data_batch, self.dim)
            dim = self.dim
        else:
            dim = [i - 1 for i in self.dim if i]
            n_elements = torch.count_nonzero(self.mask, dim=dim) * data_batch.shape[0]
            data_batch[:, self.mask == 1] = 0

        self.mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.mean + torch.sum(data_batch, dim=dim, keepdim=True)
        )
        self.squared_mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.squared_mean
            + torch.sum(data_batch**2, dim=dim, keepdim=True)
        )
        self.n_elements += n_elements

        # 1/(n_i + n_j) * (n_i * sum(x_i^2)/n_i + sum(x_j^2) - (n_i*sum(x_i)/n_i + sum(x_j))^2)
        # = 1/(n_i + n_j)  * (sum(x_i^2) + sum(x_j^2) - sum(x_i)^2 - 2sum(x_i)sum(x_j) - sum(x_j)^2))
        # multiply by (n_i + n_j) / (n_i + n_j + 1) for unbiased estimator
        self.std = (
            torch.sqrt(self.squared_mean - self.mean**2)
            * self.n_elements
            / (self.n_elements - 1)
        )

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances



class MeshDataModule:
    def __init__(
        self,
        root_dir: Union[str, Path],
        item_dir_name: Union[str, Path],
        n_train: int = None,
        n_test: int = None,
        query_res: List[int] = None,
        attributes: List[str] = None,
    ):
        """MeshDataModule provides a general dataset for irregular coordinate meshes
            for use in a GNO-based architecture

        Parameters
        ----------
        root_dir : Union[str, Path]
            str or Path to root directory of CFD dataset
        item_dir_name : Union[str, Path]
            directory in which individual item subdirs are stored
        n_train : int, optional
            hard limit on number of training examples
            if n_train is greater than the actual number
            of training examples available, nothing is changed
        n_test : int, optional
            hard limit on number of test examples
            if n_test is greater than the actual number
            of testing examples available, nothing is changed
        query_res : List[int], optional
            resolution of latent query points along each dimension
        attributes : List[str], optional
            list of string keys for attributes in the dataset to return
            as keys for each batch dict
        """

        if o3d_warn:
            print("Warning: you are attempting to run MeshDataModule without the required dependency open3d.")
            raise ModuleNotFoundError()
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)

        # Ensure path is valid
        root_dir = root_dir.expanduser()
        assert root_dir.exists(), "Path does not exist"
        assert root_dir.is_dir(), "Path is not a directory"

        # Read train and test indicies
        with open(root_dir / "train.txt") as file:
            train_ind = file.readline().split(",")

        with open(root_dir / "test.txt") as file:
            test_ind = file.readline().split(",")

        if n_train is not None:
            if n_train < len(train_ind):
                train_ind = train_ind[0:n_train]

        if n_test is not None:
            if n_test < len(test_ind):
                test_ind = test_ind[0:n_test]

        # set train and test sizes
        train_ind = train_ind[0:n_train]
        test_ind = test_ind[0:n_test]
        n_train = len(train_ind)
        n_test = len(test_ind)
        print("n_train n_test are", n_train, n_test)

        mesh_ind = train_ind + test_ind
        # remove trailing newlines from train and test indices
        mesh_ind = [x.rstrip() for x in mesh_ind]

        data_dir = root_dir / "data"

        # Load all meshes

        meshes = []
        for ind in mesh_ind:
            mesh = o3d.io.read_triangle_mesh(
                str(data_dir / (item_dir_name + ind + "/tri_mesh.ply"))
            )
            meshes.append(mesh)

        # Dataset wide bounding box
        min_b, max_b = self.get_global_bounding_box(meshes)

        # are_watertight = self.are_watertight(meshes)
        are_watertight = True

        # Uniform query points if not provided
        if isinstance(query_res, list) or isinstance(query_res, tuple):
            tx = np.linspace(min_b[0], max_b[0], query_res[0])
            ty = np.linspace(min_b[1], max_b[1], query_res[1])
            tz = np.linspace(min_b[2], max_b[2], query_res[2])

            query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        else:
            raise TypeError()

        # Compute data from meshes
        data = []
        deleted_meshes = []
        self.time_to_distance = 0.0
        for i, mesh in enumerate(meshes):
            item_dict = {}

            mesh = mesh.compute_triangle_normals()
            mesh = mesh.compute_vertex_normals()

            item_dict["vertices"] = np.asarray(mesh.vertices)
            item_dict["vertex_normals"] = np.asarray(mesh.vertex_normals)
            item_dict["triangle_normals"] = np.asarray(mesh.triangle_normals)

            centroids, area = self.compute_triangle_centroids(
                item_dict["vertices"], np.asarray(mesh.triangles)
            )

            item_dict["centroids"] = centroids
            item_dict["triangle_areas"] = area

            # Normalize vertex data based on global bound
            item_dict["vertices"] = self.range_normalize(
                item_dict["vertices"], min_b, max_b, 0, 1
            )

            item_dict["centroids"] = self.range_normalize(
                item_dict["centroids"], min_b, max_b, 0, 1
            )

            if query_points is not None:
                tt = default_timer()
                try:
                    distance, closest = self.compute_distances(
                        mesh, query_points, are_watertight
                    )
                except:
                    deleted_meshes.append(mesh_ind[i])
                    print(f"{i}-th mesh is empty and will not be added to the dataset")
                    continue
                self.time_to_distance += default_timer() - tt
                item_dict["distance"] = np.expand_dims(distance, -1)
                item_dict["closest_points"] = closest

                # Normalize vertex data based on global bound
                item_dict["closest_points"] = self.range_normalize(
                    item_dict["closest_points"], min_b, max_b, 0, 1
                )
            data.append(item_dict)

        self.time_to_distance /= len(meshes)

        del meshes

        # remove all broken meshes from training set
        n_train -= len(deleted_meshes)
        print(f"{deleted_meshes=}")

        # Bounds based on training data
        min_dist, max_dist = self.get_bounds_from_data(data[0:n_train], "distance")
        min_area, max_area = self.get_bounds_from_data(
            data[0:n_train], "triangle_areas"
        )

        for data_dict in data:
            data_dict["distance"] = self.range_normalize(
                data_dict["distance"], min_dist, max_dist, 1e-6, 1
            )
            data_dict["normalized_triangle_areas"] = self.range_normalize(
                data_dict["triangle_areas"], min_area, max_area, 1e-6, 1
            )

        # Convert to torch
        for data_dict in data:
            for key in data_dict:
                data_dict[key] = torch.from_numpy(data_dict[key]).to(torch.float32)

        # Load non-mesh data
        if attributes is not None:
            for j, ind in enumerate(mesh_ind):
                # skip corrupted meshes we caught while adding to dataset
                if ind in deleted_meshes:
                    print(f"{j}-th pressure field ind {ind} was deleted.")
                    continue
                for attr in attributes:
                    path = str(data_dir / (item_dir_name + ind + "/" + attr + ".npy"))
                    data[j][attr] = torch.from_numpy((np.load(path)))

                    if isinstance(data[j][attr], torch.Tensor):
                        data[j][attr] = data[j][attr].to(torch.float32)

            # Compute Gaussian normalizers based on training data
            normalizer_keys = []
            for attr in attributes:
                if isinstance(data[0][attr], torch.Tensor):
                    normalizer_keys.append(attr)
            # returns keyed dict of UnitGaussianNormalizer instances
            self.normalizers = UnitGaussianNormalizer.from_dataset(
                data, dim=[1], keys=normalizer_keys
            )

            # Encode all data
            for attr in normalizer_keys:
                for j in range(len(data)):
                    data_elem = data[j][attr]
                    if data_elem.shape[0] != 1:
                        data_elem = data_elem.unsqueeze(0)
                    data[j][attr] = self.normalizers[attr].transform(data_elem)

            if not bool(self.normalizers):
                self.normalizers = None
        else:
            self.normalizers = None

        # Set-up constant dict
        query_points = self.range_normalize(query_points, min_b, max_b, 0, 1)

        query_points = torch.from_numpy(query_points).to(torch.float32)
        constant = {"query_points": query_points}

        # Datasets
        self.train_data = DictDataset(data[0:n_train], constant)
        self.test_data = DictDataset(data[n_train:], constant)
        
        for i, data in enumerate(self.train_data.data_list):
            press = data["press"]
            self.train_data.data_list[i]["press"] = torch.cat(
                (press[:, 0:16], press[:, 112:]), axis=1
            )
        for i, data in enumerate(self.test_data.data_list):
            press = data["press"]
            self.test_data.data_list[i]["press"] = torch.cat(
                (press[:, 0:16], press[:, 112:]), axis=1
            )

    def get_global_bounding_box(self, meshes):
        min_b = np.zeros((3, len(meshes)))
        max_b = np.zeros((3, len(meshes)))
        for j, mesh in enumerate(meshes):
            try:
                min_b[:, j] = mesh.get_min_bound()
                max_b[:, j] = mesh.get_max_bound()
            except IndexError:
                print(f"{j}-th mesh could not be bounded. ")
                pass

        min_b = min_b.min(axis=1)
        max_b = max_b.max(axis=1)

        return min_b, max_b

    def are_watertight(self, meshes):
        for mesh in meshes:
            if not mesh.is_watertight():
                return False

        return True

    def compute_triangle_centroids(self, vertices, triangles):
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )

        centroids = (A + B + C) / 3
        areas = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2

        return centroids, areas

    def compute_distances(self, mesh, query_points, signed_distance):
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)

        if signed_distance:
            dist = scene.compute_signed_distance(query_points).numpy()
        else:
            dist = scene.compute_distance(query_points).numpy()

        closest = scene.compute_closest_points(query_points)["points"].numpy()

        return dist, closest

    def range_normalize(self, data, min_b, max_b, new_min, new_max):
        data = (data - min_b) / (max_b - min_b)
        data = (new_max - new_min) * data + new_min

        return data

    def get_bounds_from_data(self, data, key):
        global_min = data[0][key].min()
        global_max = data[0][key].max()

        for j in range(1, len(data)):
            current_min = data[j][key].min()
            current_max = data[j][key].max()

            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max

        return global_min, global_max

    def train_loader(self, **kwargs):
        return DataLoader(self.train_data, **kwargs)

    def test_loader(self, **kwargs):
        return DataLoader(self.test_data, **kwargs)

class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    
    def __init__(self):
        """
        DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    # default wrap method
    def wrap(self, model):
        self.model = model
        return self

    # default train and eval methods
    def train(self, val: bool = True):
        super().train(val)
        if self.model is not None:
            self.model.train()

    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.eval()

    @abstractmethod
    def forward(self, x):
        pass


class GINOCFDDataProcessor(DataProcessor):
    """
    Data processor for GINO training on CFD car-pressure dataset.

    This processor handles the conversion of CFD mesh data into the format
    expected by the GINO model, including graph construction and
    feature extraction from geometric inputs.
    """

    def __init__(self, normalizer, device="cuda"):
        super().__init__()
        self.normalizer = normalizer
        self.device = device
        self.model = None

    def preprocess(self, sample):
        """
        Convert CFD mesh data into GINO input format.

        Transforms the data dictionary from MeshDataModule's DictDataset
        into the form expected by the GINO model.
        """

        # input geometry: just vertices
        in_p = sample["vertices"].squeeze(0).to(self.device)
        latent_queries = sample["query_points"].squeeze(0).to(self.device)
        out_p = sample["vertices"].squeeze(0).to(self.device)
        f = sample["distance"].to(self.device)

        # Output pressure data
        truth = sample["press"].squeeze(0).unsqueeze(-1)

        # Take the first 3586 vertices of the output mesh to correspond to pressure
        # if there are less than 3586 vertices, take the maximum number of truth points
        output_vertices = truth.shape[1]
        if out_p.shape[0] > output_vertices:
            out_p = out_p[:output_vertices, :]

        truth = truth.to(device)

        batch_dict = dict(
            input_geom=in_p,
            latent_queries=latent_queries,
            output_queries=out_p,
            latent_features=f,
            y=truth,
            x=None,
        )

        sample.update(batch_dict)

        return sample

    def postprocess(self, out, sample):
        """
        Postprocess model output and ground truth data.

        Applies inverse normalization to both predictions and ground truth
        when not in training mode.
        """
        if not self.training:
            out = self.normalizer.inverse_transform(out)
            y = self.normalizer.inverse_transform(sample["y"].squeeze(0))
            sample["y"] = y

        return out, sample

    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(device)
        return self

    def wrap(self, model):
        self.model = model

    def forward(self, sample):
        """
        Complete forward pass through the data processor and model.
        """
        sample = self.preprocess(sample)
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample