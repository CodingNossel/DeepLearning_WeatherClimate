import numpy as np
import torch
import zarr
import torch.utils.data

class MarsDataset(torch.utils.data.IterableDataset):
    def __init__(self, path_file, batch_size, level_from_bottom=35):
        super(MarsDataset, self).__init__()
        self.batch_size = batch_size
        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)
        self.rng = np.random.default_rng()
        if self.sources['lev'].shape[0] <= level_from_bottom:
            self.levels = self.sources['lev'].shape[0]
        else:
            self.levels = level_from_bottom
        self.shuffle()

    def shuffle(self):
        len = self.sources['time'].shape[0]
        self.idxs = self.rng.permutation(np.arange(len - 1))
        self.len = self.idxs.shape[0]

    def __iter__(self):
        iter_start, iter_end = self.worker_workset()
        source = None
        target = None
        for bidx in range(iter_start, iter_end, self.batch_size):
            if bidx + self.batch_size >= iter_end:
                bidx = iter_end - self.batch_size
            idx_t = self.idxs[bidx: bidx + self.batch_size]
            source = torch.stack([
                torch.tensor(np.array([normalize_temp(x) for x in self.sources['temp'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['u'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['v'][idx_t]]))
            ], 1)
            source = source.transpose(1, 3).transpose(2, 4)
            source = source[..., :self.levels]
            source = np.reshape(source, (self.batch_size, source.shape[1], source.shape[2], -1))
            source = source.transpose(1, 2).transpose(1, 3)

            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(np.array([normalize_temp(x) for x in self.sources['temp'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['u'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['v'][idx_t]]))
            ], 1)
            target = target.transpose(1, 3).transpose(2, 4)
            target = target[..., :self.levels]
            target = np.reshape(target, (self.batch_size, target.shape[1], target.shape[2], -1))
            target = target.transpose(1, 2).transpose(1, 3)
            
            idx_t -= 1
            ## to transform back np.reshape(source, (8, 36, 72, 3, 70))
            yield source, target

    def __len__(self):
        return self.len

    def worker_workset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self)
        else:
            # split workload
            temp = len(self)
            per_worker = ((int(temp / float(worker_info.num_workers)) // self.batch_size) * self.batch_size) + self.batch_size
            if worker_info.id+1 == worker_info.num_workers:
                # per_worker = int(temp // float(worker_info.num_workers)) - (int(temp // float(worker_info.num_workers)) % self.batch_size)
                iter_end = int(temp)
                iter_start = int(iter_end - per_worker)
            else:
                worker_id = worker_info.id
                iter_start = int(worker_id * per_worker)
                iter_end = int(iter_start + per_worker)

        return iter_start, iter_end
    
    def get_lat(self):
        """
        Returns the latitude array
        """
        return self.sources['lat']
    
    def get_lev(self):
        """
        Returns the level array
        """
        # return self.sources['lev'][:10]
        return self.sources['lev']
    
    def get_lon(self):
        """
        Returns the longitude array
        """
        return self.sources['lon']


class MarsDatasetArray(torch.utils.data.IterableDataset):
    def __init__(self, path_file, batch_size, level_from_bottom=35):
        super(MarsDatasetArray, self).__init__()
        self.batch_size = batch_size

        self.sources = []
        for file in path_file:
            store = zarr.DirectoryStore(file)
            self.sources.append(zarr.group(store=store))

        self.models = len(self.sources)

        # use the length of time for the model length for every model
        self.model_len = self.sources[0]['time'].shape[0]

        if self.sources[0]['lev'].shape[0] <= level_from_bottom:
            self.levels = self.sources[0]['lev'].shape[0]
        else:
            self.levels = level_from_bottom

        self.rng = np.random.default_rng()
        self.shuffle()

    def shuffle(self):
        # calculate the length of the whole dataset
        len = self.models * self.model_len
        self.idxs = self.rng.permutation(np.arange(len - 1))
        self.len = self.idxs.shape[0]

    def __iter__(self):
        iter_start, iter_end = self.worker_workset()
        source = []
        target = []
        for bidx in range(iter_start, iter_end, self.batch_size):
            if bidx + self.batch_size >= iter_end:
                bidx = iter_end - self.batch_size
            idx_t = self.idxs[bidx: bidx + self.batch_size]
            mod_len = self.model_len
            for mid, model in enumerate(self.sources):
                temp_idx_t = [val for val in idx_t if mid * mod_len <= val < (mid + 1) * mod_len]
                new_idx_t = np.array([x - mid * mod_len for x in temp_idx_t])
                stack = torch.stack([
                    torch.tensor(np.array([normalize_temp(x) for x in model['temp'][new_idx_t]])),
                    torch.tensor(np.array([normalize_wind(x) for x in model['u'][new_idx_t]])),
                    torch.tensor(np.array([normalize_wind(x) for x in model['v'][new_idx_t]]))
                ], 1)

                if mid == 0:
                    source = stack
                else:
                    source = torch.cat((source, stack), 0)
            
            source = source.transpose(1, 3).transpose(2, 4)
            source = source[..., :self.levels]
            source = np.reshape(source, (self.batch_size, source.shape[1], source.shape[2], -1))
            source = source.transpose(1, 2).transpose(1, 3)

            # target is subsequent time step
            idx_t += 1

            for mid, model in enumerate(self.sources):
                temp_idx_t = [val for val in idx_t if mid * mod_len <= val < (mid + 1) * mod_len]
                new_idx_t = np.array([x - mid * mod_len for x in temp_idx_t])
                stack = torch.stack([
                    torch.tensor(np.array([normalize_temp(x) for x in model['temp'][new_idx_t]])),
                    torch.tensor(np.array([normalize_wind(x) for x in model['u'][new_idx_t]])),
                    torch.tensor(np.array([normalize_wind(x) for x in model['v'][new_idx_t]]))
                ], 1)

                if mid == 0:
                    target = stack
                else:
                    target = torch.cat((target, stack), 0)

            target = target.transpose(1, 3).transpose(2, 4)
            target = target[..., :self.levels]
            target = np.reshape(target, (self.batch_size, target.shape[1], target.shape[2], -1))
            target = target.transpose(1, 2).transpose(1, 3)
            
            idx_t -= 1
            ## to transform back np.reshape(source, (8, 36, 72, 3, 70))
            yield source, target

    def __len__(self):
        return self.len

    def worker_workset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self)
        else:
            # split workload
            temp = len(self)
            per_worker = ((int(temp / float(worker_info.num_workers)) // self.batch_size) * self.batch_size) + self.batch_size
            if worker_info.id+1 == worker_info.num_workers:
                # per_worker = int(temp // float(worker_info.num_workers)) - (int(temp // float(worker_info.num_workers)) % self.batch_size)
                iter_end = int(temp)
                iter_start = int(iter_end - per_worker)
            else:
                worker_id = worker_info.id
                iter_start = int(worker_id * per_worker)
                iter_end = int(iter_start + per_worker)

        return iter_start, iter_end
    
    def get_lat(self):
        """
        Returns the latitude array
        """
        return self.sources['lat']
    
    def get_lev(self):
        """
        Returns the level array
        """
        # return self.sources['lev'][:10]
        return self.sources['lev']
    
    def get_lon(self):
        """
        Returns the longitude array
        """
        return self.sources['lon']

class MarsDatasetLevel(torch.utils.data.IterableDataset):
    def __init__(self, path_file, batch_size, levels):
        super(MarsDatasetLevel, self).__init__()
        self.batch_size = batch_size
        self.levels = levels
        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)
        self.rng = np.random.default_rng()
        self.shuffle()

    def shuffle(self):
        len = self.sources['time'].shape[0]
        self.idxs = self.rng.permutation(np.arange(len - 1))
        self.len = self.idxs.shape[0]

    def __iter__(self):
        iter_start, iter_end = self.worker_workset()
        source = None
        target = None
        for bidx in range(iter_start, iter_end, self.batch_size):
            idx_t = self.idxs[bidx: bidx + self.batch_size]
            source = torch.stack([
                torch.tensor(np.array([normalize_temp(x) for x in self.sources['temp'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['u'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['v'][idx_t]]))
            ], 1)
            source = source.transpose(1, 3).transpose(2, 4)
            source = source[..., :self.levels]
            source = source.transpose(3, 4)

            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(np.array([normalize_temp(x) for x in self.sources['temp'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['u'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['v'][idx_t]]))
            ], 1)
            target = target.transpose(1, 3).transpose(2, 4)
            target = target[..., :self.levels]
            target = target.transpose(3, 4)

            yield source, target

    def __len__(self):
        return self.len

    def worker_workset(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self)
        else:
            # split workload
            temp = len(self)
            per_worker = ((int(temp / float(worker_info.num_workers)) // self.batch_size) * self.batch_size) + self.batch_size
            if worker_info.id+1 == worker_info.num_workers:
                # per_worker = int(temp // float(worker_info.num_workers)) - (int(temp // float(worker_info.num_workers)) % self.batch_size)
                iter_end = int(temp)
                iter_start = int(iter_end - per_worker)
            else:
                worker_id = worker_info.id
                iter_start = int(worker_id * per_worker)
                iter_end = int(iter_start + per_worker)

        return iter_start, iter_end
    
    def get_lat(self):
        """
        Returns the latitude array
        """
        return self.sources['lat']
    
    def get_lev(self):
        """
        Returns the level array
        """
        return self.sources['lev'][:self.levels]
    
    def get_lon(self):
        """
        Returns the longitude array
        """
        return self.sources['lon']


def normalize_temp(temp):
    """
    Normalizes temperature to 0-1 range
    """
    return (temp - 80) / 280

def normalize_wind(wind):
    """
    Normalizes wind to 0-1 range
    """
    return (wind + 200) / 450

def denormalize_temp(temp):
    """
    Denormalizes temperature to 80-280 range
    """
    return (temp * 280) + 80


def denormalize_wind(wind):
    """
    Denormalizes wind to (-200)-250 range
    """
    return (wind * 450) - 200

def create_one_demension_normalized_tensor(matrix):
    """
    Creates a one-dimensional normalized tensor for a given 4D matrix. 
    The matrix shape should has to be [3, 70, 36, 72]
    retrun a one-dimensional normalized tensor
    """
    normal_flat = torch.tensor([], dtype=torch.float32)
    for i in range(matrix.shape[1] - 1):
        for j in range(matrix.shape[2] - 1): 
            for k in range(matrix.shape[3] - 1):
                temp = matrix[0, i, j, k].item()
                u = matrix[1, i, j, k].item()
                v = matrix[2, i, j, k].item()
                norm_temp = normalize_temp(temp)
                norm_u = normalize_wind(u)
                norm_v = normalize_wind(v)
                normal_flat = torch.cat((normal_flat, torch.tensor([norm_temp, norm_u, norm_v])), dim=0)
    return normal_flat

def create_denormalized_matrix_from_tensor(vector):
    """
    Creates a 4D matrix from a one-dimensional normalized tensor. 
    retrun a 4D matrix with the shape [3, 70, 36, 72]
    """
    denormalized_matrix = torch.zeros(3, 70, 36, 72)
    for i in range(denormalized_matrix.shape[1] - 1):
        for j in range(denormalized_matrix.shape[2] - 1): 
            for k in range(denormalized_matrix.shape[3] - 1):
                idx = i * denormalized_matrix.shape[2] + denormalized_matrix.shape[3] * j + denormalized_matrix.shape[3] * k
                norm_temp = vector[idx * 3]
                norm_u = vector[idx * 3 + 1]
                norm_v = vector[idx * 3 + 2]
                temp = denormalize_temp(norm_temp)
                u = denormalize_wind(norm_u)
                v = denormalize_wind(norm_v)
                denormalized_matrix[0, i, j, k] = temp
                denormalized_matrix[1, i, j, k] = u
                denormalized_matrix[2, i, j, k] = v
    return denormalized_matrix