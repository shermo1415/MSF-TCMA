import torch

class Configs:
    def __init__(self):
        pass


configs = Configs()

configs.device = torch.device('cuda:0')

configs.batch_size = 8
configs.batch_size_test = 2

configs.num_epochs = 200
configs.opt = 0.0001 #原 0.001； swin 0.0002

configs.train_path = '../D2_24composite_onezero_train.npy'
configs.val_path = '../D2_24composite_onezero_val.npy'
configs.test_path = '../D2_24composite_onezero_test.npy'
configs.dims = ['t2m', 'sp', 'u10', 'v10']
configs.interinterval = 10
configs.trainframes = (configs.interinterval-1)//2

# configs.image_size = (64, 64)
configs.train_shuffle = False

configs.Cin = 4
configs.emb_dim = 64
configs.attn_layer = 4
configs.blocks = 2

configs.display_interval = 8
configs.eval_interval = 5
