import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from model.model import network
from makedataset import TimeDownScalingDataset
import pickle
import math
from torch.cuda.amp import autocast, GradScaler
from config import configs
from log import printwrite
file = "log/log.txt"

class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.Cin = configs.Cin
        self.network =network(configs.Cin, configs.emb_dim, configs.blocks, configs.attn_layer).to(self.device)
        self.interinterval = configs.interinterval
                    
    def test_loss(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)
                
    def test(self, testset, dataloader_eval):
        self.network.eval()
        predlist = []
        truelist = []
        i1i2list = []
        with torch.no_grad():
            for img, it, time, fy, fygt in dataloader_eval:
                i1 = img[:,:self.Cin]
                i2 = img[:,self.Cin:self.Cin*2]
                
                pred = self.network(i1.float().to(self.device), i2.float().to(self.device), time, fy.float().to(self.device), self.interinterval - 1)
                predlist.append(pred)
                truelist.append(it.float().to(self.device))
                i1i2list.append(torch.cat([i1, i2], dim = 1))
                
            predlist =torch.cat(predlist, dim=0)
            truelist =torch.cat(truelist, dim=0)
            i1i2list =torch.cat(i1i2list, dim = 0)
            testloss = self.test_loss(predlist, truelist)
            print(testloss)
            print(predlist.shape, truelist.shape)
            np.save("result/pred.npy", predlist.cpu().detach().numpy())
            np.save("result/true.npy", truelist.cpu().detach().numpy())
            np.save("result/i1i2.npy", i1i2list.cpu().detach().numpy())
        return
    
    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)
    def save_model(self, path):
        torch.save({'net': self.network.state_dict()}, path)

if __name__ == '__main__':
    
    dataset_eval = TimeDownScalingDataset(configs.test_path, configs.dims, configs.interinterval, train= False)
    # dataset_eval.indexs = dataset_eval.indexs[::25]
    print(dataset_eval.GetDataShape())
    dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False)
    
    trainer = Trainer(configs)    
    model = torch.load('exp/SRNet.chk')
    trainer.network.load_state_dict(model['net'])
    trainer.network.eval()
    trainer.test(dataset_eval, dataloader_eval)
