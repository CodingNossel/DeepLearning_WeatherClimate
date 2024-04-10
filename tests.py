from MarsDataset import MarsDataset
import torch.utils.data

if __name__ == "__main__":
    ds = MarsDataset('data/test.zarr', 8)
    dataset_iter = iter(ds)
    a = next(dataset_iter)

    loader_params = {'batch_size': None, 'batch_sampler': None, 'shuffle': False, 'num_workers': 1, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(ds, **loader_params, sampler=None)
    data_loader_iter = iter(data_loader)

    for bidx, (source, target) in enumerate(dataset_iter):
        print("Batch Index:", bidx)
        print(source.shape)
