# from MarsDataset import MarsDataset
# import torch
# from torch.utils.data import DataLoader
# import pickle

# marsdata = MarsDataset('data/test.zarr', 32)

# dataset_iter = iter(marsdata)

# a = next(dataset_iter)

# # print(a[0].shape)

# torch.Size([32, 3, 70, 36, 72])


# loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False,  'num_workers': 1, 'pin_memory': True }

# data_loader = DataLoader(marsdata, **loader_params, sampler = None) 

# data_loader_iter = iter(data_loader)

# print(data_loader_iter)

# for bidx, data in enumerate(dataset_iter):
#     print(bidx)
#     print(data)
#     # feed data to network

# from MarsDataset import MarsDataset
# import torch

# marsdata = MarsDataset('data/test.zarr', 32)

# dataset_iter = iter(marsdata)

# a = next(dataset_iter)

# # print(a[0].shape)

# torch.Size([32, 3, 70, 36, 72])


# loader_params = { 'batch_size': 32, 'batch_sampler': None, 'shuffle': False,  'num_workers': 1, 'pin_memory': True }

# data_loader = torch.utils.data.DataLoader(marsdata, **loader_params, sampler = None) 

# data_loader_iter = iter(data_loader)

# for batch_idx, (source, target) in enumerate(data_loader_iter):
#     # Here, source and target are tensors containing your batch of data
#     # Do whatever processing you need to do with this batch
    
#     # Example processing:
#     print("Batch Index:", batch_idx)
#     print("Source Shape:", source.shape)
#     print("Target Shape:", target.shape)

"""
if __name__ == '__main__':
    from MarsDataset_old import MarsDataset
    import torch

    num_workers = 4
    marsdata = MarsDataset('data/test.zarr', num_workers)

    loader_params = {
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': True
    }

    data_loader = torch.utils.data.DataLoader(marsdata, batch_size=None, **loader_params, sampler=None)

    data_loader_iter = iter(data_loader)

    # print(2)
    for batch_idx, (source, target) in enumerate(data_loader_iter):
        # print(3)
        # Here, source and target are tensors containing your batch of data
        # Do whatever processing you need to do with this batch
        
        # Example processing:
        print("Batch Index:", batch_idx)
        print("Source Shape:", source.shape)
        print("Target Shape:", target.shape)
"""

if __name__ == '__main__':
    import torch
    from era5_dataset import ERA5Dataset
    from MarsDataset import MarsDataset

    # ds = ERA5Dataset( 'data/era5_6hourly.zarr', 32)
    ds = MarsDataset( 'data/test.zarr', 8)
    dataset_iter = iter(ds)
    a = next( dataset_iter)
    a[0].shape

    torch.Size([8, 5, 5, 121, 240])

    # parallel loaders

    loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False,  'num_workers': 4, 'pin_memory': True }

    data_loader = torch.utils.data.DataLoader(ds, **loader_params, sampler = None) 

    data_loader_iter = iter( data_loader)

    # iterate over all batches

    for bidx, (source, target) in enumerate(dataset_iter) :
        print("Batch Index:", bidx)
        print(source.shape)
