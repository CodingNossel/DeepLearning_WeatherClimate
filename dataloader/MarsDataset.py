import numpy as np
import torch
import zarr
import torch.utils.data


class MarsDataset(torch.utils.data.IterableDataset):
    """
    Iterable Dataset for the OpenMARS weather Dataset.  

    Args:
        path_file (str): Path to the zarr.
        batch_size (int): Batch size for each iteration.
        level_from_bottom (int, optional): Number of levels from the bottom. Defaults to 35.
    """

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

            source = source[:, :, :self.levels, :, :]
            source = np.reshape(source, (self.batch_size, -1, source.shape[3], source.shape[4]))

            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(np.array([normalize_temp(x) for x in self.sources['temp'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['u'][idx_t]])),
                torch.tensor(np.array([normalize_wind(x) for x in self.sources['v'][idx_t]]))
            ], 1)

            target = target[:, :, :self.levels, :, :]
            target = np.reshape(target, (self.batch_size, -1, target.shape[3], target.shape[4]))

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
            per_worker = ((int(temp / float(
                worker_info.num_workers)) // self.batch_size) * self.batch_size) + self.batch_size
            if worker_info.id + 1 == worker_info.num_workers:
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


def normalize_temp(temp):
    """
    Normalizes temperature to 0-1 range
    """
    return (temp - 80) / 280

def denormalize_temp(temp):
    """
    Denormalizes temperature to 80-280 range
    """
    return (temp * 280) + 80

def normalize_wind(wind):
    """
    Normalizes wind to 0-1 range
    """
    return (wind + 200) / 450

def denormalize_wind(wind):
    """
    Denormalizes wind to (-200)-250 range
    """
    return (wind * 450) - 200


def create_denormalized_matrix_from_tensor(vector, level):
    """
    Creates a 4D Matrix from a 3D normalized matrix with the shape [3*level, 36, 72] 
    return a 4D matrix with the shape [36, 72, 3, level]
    """
    vector = vector.transpose(0, 1).transpose(1, 2)
    vector = np.reshape(vector, (36, 72, 3, level))
    vector = torch.stack([
        torch.tensor(np.array([denormalize_temp(x) for x in vector[:,:,0]])),
        torch.tensor(np.array([denormalize_wind(x) for x in vector[:,:,1]])),
        torch.tensor(np.array([denormalize_wind(x) for x in vector[:,:,2]]))
    ], 1)
    vector = vector.transpose(1, 2)
    return vector

