
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

    self.batch_size = batch_size

    store = zarr.DirectoryStore(path_file)
    self.sources = zarr.group(store=store)

    self.rng = np.random.default_rng()
    self.shuffle()
  
  #####################
  def shuffle(self) :

    len = self.sources['time'].shape[0]
    self.idxs = self.rng.permutation( np.arange(len))
    
    self.len = self.idxs.shape[0]
    
  #####################
  def __len__(self):
    return self.len

  #####################
  def __iter__(self):

    self.shuffle()
    iter_start, iter_end = self.worker_workset()

    for bidx in range( iter_start, iter_end, self.batch_size) :

      idx_t = self.idxs[bidx : bidx+self.batch_size]

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

  #####################
  def __len__(self):
      return self.len // self.batch_size

  #####################
  def worker_workset( self) :

    worker_info = torch.utils.data.get_worker_info()

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
  