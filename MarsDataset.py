import numpy as np
import torch
import zarr
import pandas as pd
import code


###################################################
class MarsDataset(torch.utils.data.IterableDataset):

    #####################
    def __init__(self, path_file, batch_size):

        super(MarsDataset, self).__init__()
    #####################
    def __init__(self, path_file, batch_size):
    
        super(MarsDataset, self).__init__()

        self.batch_size = batch_size
        self.batch_size = batch_size

        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)
        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)

        self.rng = np.random.default_rng()
        self.shuffle()

    #####################
    def shuffle(self):
        self.rng = np.random.default_rng()

        # self.num_batches = len(self.sources['time']) - 1 

        self.shuffle()
  
    #####################
    def shuffle(self) :

        len = self.sources['time'].shape[0]
        self.idxs = self.rng.permutation(np.arange(len))

        self.len = self.idxs.shape[0]

    #####################
    def __len__(self):
        return self.len
        len = self.sources['time'].shape[0]
        self.idxs = self.rng.permutation( np.arange(len))

        self.len = self.idxs.shape[0]
        # self.idxs = self.rng.permutation(len(self.sources['time']) - 1)

    #####################
    # def __len__(self):
    #     return self.len
        # return self.num_batches

    #####################
    def __iter__(self):
    #####################
    def __iter__(self):

        self.shuffle()
        iter_start, iter_end = self.worker_workset()
        self.shuffle()
        iter_start, iter_end = self.worker_workset()

        for bidx in range(iter_start, iter_end, self.batch_size):
            idx_t = self.idxs[bidx: bidx + self.batch_size]
        for bidx in range( iter_start, iter_end, iter_end - iter_start) :

            idx_t = self.idxs[bidx : bidx+ iter_end - iter_start]
            print(idx_t)

            source = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)
            source = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)

            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)

            # TODO: data normalization

            yield (source, target)
            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)

        #     # TODO: data normalization
        # print(source.shape)
        # print(target.shape)
        yield (source, target)

    #####################
    def __len__(self):
        return self.len // self.batch_size
        # self.shuffle()

        # # print(4)

        # # Get worker information
        # worker_info = torch.utils.data.get_worker_info()
        # # print(5)

        # # Calculate workset for the current worker
        # if worker_info is None: 
        #     print(6)
        #     iter_start = 0
        #     iter_end = self.num_batches
        # else:
        #     print(7)
        #     # split workload
        #     num_workers = worker_info.num_workers
        #     per_worker = self.num_batches // num_workers
        #     worker_id = worker_info.id
        #     iter_start = per_worker * worker_id
        #     iter_end = iter_start + per_worker
        #     if worker_id == num_workers - 1:  # Last worker
        #         iter_end = self.num_batches

        # # Iterate over batches assigned to this worker
        # for batch_idx in range(iter_start, iter_end):
        #     # print(8)
        #     start_idx = batch_idx * self.batch_size
        #     end_idx = (batch_idx + 1) * self.batch_size

        #     idx_t = self.idxs[start_idx:end_idx]

        #     source = torch.stack([
        #         torch.tensor(self.sources['temp'][idx_t]),
        #         torch.tensor(self.sources['u'][idx_t]),
        #         torch.tensor(self.sources['v'][idx_t])
        #     ], 1)

        #     # target is subsequent time step
        #     idx_t += 1
        #     target = torch.stack([
        #         torch.tensor(self.sources['temp'][idx_t]),
        #         torch.tensor(self.sources['u'][idx_t]),
        #         torch.tensor(self.sources['v'][idx_t])
        #     ], 1)
        #     print(source.shape)
        #     print(target.shape)
        #     yield source, target

    def __len__(self):
        # print(self.len)
        # print(self.batch_size)
        # print(self.len // self.batch_size)
        # print(len(self.sources['time']) - 1)
        return self.len

    #####################
    def worker_workset(self):
    #####################
    def worker_workset( self) :

        worker_info = torch.utils.data.get_worker_info()
        worker_info = torch.utils.data.get_worker_info()

        print(worker_info)

        if worker_info is None:
            iter_start = 0
            iter_end = len(self)

        else:
            # split workload
            temp = len(self)
            per_worker = int(np.floor(temp / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = int(worker_id * per_worker)
            iter_end = int(iter_start + per_worker)
            if worker_info.id + 1 == worker_info.num_workers:
                iter_end = int(temp)
        if worker_info is None: 
            iter_start = 0
            iter_end = len(self)
        else:  
            # split workload
            temp = len(self)
            per_worker = int( np.floor( temp / float(worker_info.num_workers) ) )
            worker_id = worker_info.id
            iter_start = int(worker_id * per_worker)
            iter_end = int(iter_start + per_worker)
            if worker_info.id+1 == worker_info.num_workers :
                iter_end = int(temp)

        return iter_start, iter_end

        print(iter_start)
        print(iter_end)
        return iter_start, iter_end

    #####################
#     def __len__(self):
#         return self.len // self.batch_size

#   #####################
#     def worker_workset( self) :

#         worker_info = torch.utils.data.get_worker_info()

#         if worker_info is None: 
#             iter_start = 0
#             iter_end = len(self)
    
#         else:  
#             # split workload
#             temp = len(self)
#             per_worker = int( np.floor( temp / float(worker_info.num_workers) ) )
#             worker_id = worker_info.id
#             iter_start = int(worker_id * per_worker)
#             iter_end = int(iter_start + per_worker)
#             if worker_info.id+1 == worker_info.num_workers :
#                 iter_end = int(temp)
    
#         return iter_start, iter_end
            
"""
class MarsDataset(torch.utils.data.IterableDataset):

    def __init__(self, path_file, batch_size):
        super(MarsDataset, self).__init__()

        self.batch_size = batch_size

        store = zarr.DirectoryStore(path_file)
        self.sources = zarr.group(store=store)

        self.rng = np.random.default_rng()

        # Calculate the number of batches
        self.num_samples = len(self.sources['time'])
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def shuffle(self):
        # Shuffle indices for each epoch
        self.idxs = self.rng.permutation(self.num_samples)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Shuffle indices at the start of each iteration
        self.shuffle()

        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.num_samples)

            idx_t = self.idxs[start_idx:end_idx]

            source = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)

            # target is subsequent time step
            idx_t += 1
            target = torch.stack([
                torch.tensor(self.sources['temp'][idx_t]),
                torch.tensor(self.sources['u'][idx_t]),
                torch.tensor(self.sources['v'][idx_t])
            ], 1)

            yield source, target
"""