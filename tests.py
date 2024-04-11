from MarsDataset import MarsDataset, create_one_demension_normalized_tensor, create_denormalized_matrix_from_tensor
import torch.utils.data

if __name__ == "__main__":
    ds = MarsDataset('data/test.zarr')

    loader_params = {'batch_size': None, 'batch_sampler': None, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(ds, **loader_params, sampler=None)
    data_loader_iter = iter(data_loader)

    # max_temp = 1
    # min_temp = 10000
    # max_u = 0
    # min_u = 10000
    # max_v = 0
    # min_v = 10000

    for bidx, (source, target) in enumerate(data_loader_iter):
        print("Batch Index:", bidx)
        # print(source.shape)
        # print(target.shape)
        # norm = create_one_demension_normalized_tensor(source[0])
        # print(norm.shape)
        # print(norm)
        # matri = create_denormalized_matrix_from_tensor(norm)
        # print(matri.shape)
        # print(source[0][0][0][0][0].item())
        # print(source[0][1][0][0][0].item())
        # print(source[0][2][0][0][0].item())
        # print(matri[0][0][0][0].item())
        # print(matri[1][0][0][0].item())
        # print(matri[2][0][0][0].item())

        ## Iteriert über jeden Zeitschritt. Hier können source[i] und target[i] aufgerufen werden.
        for i in range(source.shape[0]):
            print(source[i].shape)
            print(target[i].shape)
            # norm = create_one_demension_normalized_tensor(source[i])
            # norm_t = create_one_demension_normalized_tensor(target[i])
            # print(norm.shape)
            # print(norm)
            # print(norm_t.shape)
            # print(norm_t)
            # matri = create_denormalized_matrix_from_tensor(norm)
            # print(matri.shape)
        """ for i in range(source.shape[2] - 1):
            for j in range(source.shape[3] - 1): 
                for k in range(source.shape[4] - 1):
                    # print(source[0, :, i, j, k][:3])
                    # print(source[0, 0, i, j, k])
                    # print(source[0, 1, i, j, k])
                    # print(source[0, 2, i, j, k])
                    if source[0, 0, i, j, k] > max_temp:
                        max_temp = source[0, 0, i, j, k].item()
                    elif source[0, 0, i, j, k] < min_temp:
                        min_temp = source[0, 0, i, j, k].item()
                    if source[0, 1, i, j, k] > max_u:
                        max_u = source[0, 1, i, j, k].item()
                    elif source[0, 1, i, j, k] < min_u:
                        min_u = source[0, 1, i, j, k].item()
                    if source[0, 2, i, j, k] > max_v:
                        max_v = source[0, 2, i, j, k].item()
                    elif source[0, 2, i, j, k] < min_v:
                        min_v = source[0, 2, i, j, k].item() """
        """ if bidx == 7:
            # print(torch.flatten(source[1]))
            for i in range(source.shape[2] - 1):
                for j in range(source.shape[3] - 1): 
                    for k in range(source.shape[4] - 1):
                        # print(source[0, :, i, j, k][:3])
                        # print(source[0, 0, i, j, k])
                        # print(source[0, 1, i, j, k])
                        # print(source[0, 2, i, j, k])
                        if source[0, 0, i, j, k] > max_temp:
                            max_temp = source[0, 0, i, j, k].item()
                        elif source[0, 0, i, j, k] < min_temp:
                            min_temp = source[0, 0, i, j, k].item()
                        if source[0, 1, i, j, k] > max_u:
                            max_u = source[0, 1, i, j, k].item()
                        elif source[0, 1, i, j, k] < min_u:
                            min_u = source[0, 1, i, j, k].item()
                        if source[0, 2, i, j, k] > max_v:
                            max_v = source[0, 2, i, j, k].item()
                        elif source[0, 2, i, j, k] < min_v:
                            min_v = source[0, 2, i, j, k].item() """
    # print(max_temp) # tensor(261.5564)  # 262.3379211425781     # 280
    # print(min_temp) # tensor(109.9754)  # 105.26789093017578    # 80
    # print(max_u) # tensor(233.2244)     # 228.05648803710938    # 250
    # print(min_u) # tensor(-145.5250)    # -174.0041046142578    # -200
    # print(max_v) # tensor(167.2612)     # 180.96763610839844    # 250
    # print(min_v) # tensor(-140.6787)    # -155.6337432861328    # -200
