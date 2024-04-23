import numpy as np
import torch.utils.data
from unet.MarsDataset import MarsDataset, create_denormalized_matrix_from_tensor, denormalize_temp, denormalize_wind
from visualisation import heat_plotting

if __name__ == "__main__":
    # testdata = ['data/beta.zarr']
    # ds = MarsDatasetArray(testdata, 20)
    testdata = 'data/my27.zarr'
    ds = MarsDataset(testdata, 20, 10)

    loader_params = {'batch_size': None, 'batch_sampler': None, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(ds, **loader_params, sampler=None)
    data_loader_iter = iter(data_loader)

    for bidx, (source, target) in enumerate(data_loader_iter):
        print("Batch Index:", bidx)
        print(source.shape)
        print(target.shape)
        if bidx == 2:
            for id, batch in enumerate(source):
                # matrix = create_denormalized_matrix_from_tensor(batch, 5)
                # print(f"Temp: {denormalize_temp(batch[0][0][0])} and {matrix[0][0][0][0]}")
                # print(f"Wind: {denormalize_wind(batch[5][0][0])} and {matrix[0][0][1][0]}")
                # print(f"Wind: {denormalize_wind(batch[10][0][0])} and {matrix[0][0][2][0]}")
                # print(batch[:,0,0])
                # print(matrix[0,0])
                if id == 2:
                    calc = batch
                    batch = batch.transpose(0, 1).transpose(1, 2)
                    batch = np.reshape(batch, (36, 72, 3, 10))
                    heat_plotting(batch, "Temp_source", 0, 0)


                    matrix = create_denormalized_matrix_from_tensor(calc, 10)
                    heat_plotting(matrix, "Temp", 0, 0)

        ## Iteriert über jeden Zeitschritt. Hier können source[i] und target[i] aufgerufen werden.
        """ for i in range(source.shape[0]):
            print(source[i].shape)
            print(target[i].shape)
            one_s = create_one_demension_normalized_tensor(source[i])
            one_t = create_one_demension_normalized_tensor(target[i])
            print(one_s)
            print(one_t)
            mat_s = create_denormalized_matrix_from_tensor(one_s)
            mat_t = create_denormalized_matrix_from_tensor(one_t)
            print(mat_s.shape)
            print(mat_t.shape) """
        # print(source[0][0][0][0][1])
        # print(source[0][0][0][1][0])
        # print(source[0][0][0][0])
        # print(source[0][0][0][1])
        # print(source[0][0][0][2])
        # print(source[0][0][0][160])
        # print(target[0][0][0][0])
        # print(target[0][0][0][70])
        # print(target[0][0][0][90])
        # print(target[0][0][0][140])
        # print(target[0][0][0][0][1])
        # print(target[0][0][0][1][0])
    
## values for normalisation
# max_temp = tensor(261.5564)  # 262.3379211425781     # 280
# min_temp = tensor(109.9754)  # 105.26789093017578    # 80
# max_u = tensor(233.2244)     # 228.05648803710938    # 250
# min_u = tensor(-145.5250)    # -174.0041046142578    # -200
# max_v = tensor(167.2612)     # 180.96763610839844    # 250
# min_v = tensor(-140.6787)    # -155.6337432861328    # -200
