from MarsDataset import MarsDataset
import torch

marsdata = MarsDataset('data/test.zarr', 32)

dataset_iter = iter(marsdata)

a = next(dataset_iter)

# print(a[0].shape)

torch.Size([32, 3, 70, 36, 72])


loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False,  'num_workers': 1, 'pin_memory': True }

data_loader = torch.utils.data.DataLoader(marsdata, **loader_params, sampler = None) 

data_loader_iter = iter(data_loader)


for bidx, data in enumerate(dataset_iter) :
    print(bidx)
    print(data)
  # feed data to network