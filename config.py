import torch

class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.name = 'Northeast'
configs.scale = 2

configs.device = torch.device('cuda:0')

configs.batch_size = 32
configs.batch_size_test = 1

configs.epochs = 100
configs.lr = 0.001
configs.opt_size = 128

configs.dims = ['u100', 'v100']
configs.downsample = 'mean'
configs.train_shuffle = False

configs.num_layers = 6
configs.num_heads = 6
configs.display_interval = 30 
configs.warm_up = 1000
configs.patience = 100


configs.train_path = r"Northeast/High_Res_train.npy"
configs.val_path = r"Northeast/High_Res_val.npy"
configs.test_path = r"Northeast/High_Res_test.npy"
configs.geo_path = r"Northeast/DEM_0.25deg_NortheastChina_norm.npy"
