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

def SingleFrameDeltaEnLoss(y_pred, y_true, i0, i1): #B,W,H
    TrueDeltaEn_0t = (i0 - y_true) ** 2
    PredDeltaEn_0t = (i0 - y_pred) ** 2
    TrueDeltaEn_t1 = (i1 - y_true) ** 2
    PredDeltaEn_t1 = (i1 - y_pred) ** 2
    return F.l1_loss(TrueDeltaEn_0t, PredDeltaEn_0t) + F.l1_loss(TrueDeltaEn_t1, PredDeltaEn_t1)

def SingleFrameMAELoss(y_pred, y_true):
    return F.l1_loss(y_pred, y_true)

def FrameWeightedLoss(losses):
    loss = torch.tensor(0).float().cuda()
    loss.requires_grad=True
    frames = len(losses)
    ts = np.linspace(0,1,frames+2)[1:-1]
    ws = np.exp(np.sin(np.pi*ts))/np.exp(np.sin(np.pi*ts)).sum()*frames
    for f in range(frames):
        loss = loss + ws[f]*losses[f]
    return loss
class Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.Cin = configs.Cin
        self.network =network(configs.Cin, configs.emb_dim, configs.blocks, configs.attn_layer).to(self.device)
        # self.awl = AutomaticWeightedLoss(configs.Cin)
        
        self.opt = torch.optim.Adam(self.network.parameters(), lr = configs.opt)#, weight_decay=configs.weight_decay)
        self.scaler = GradScaler()
        self.trainframes = configs.trainframes
        self.interinterval = configs.interinterval
        
    def train_loss(self, y_preds, y_trues, i0, i1):
        losses = []
        # frameMAEloss = []
        # framedEnloss = []
        for f in range(self.trainframes):
            y_pred_frame = y_preds[:, f]
            y_true_frame = y_trues[:, f]
            # frameMAEloss.append(SingleFrameMAELoss(y_pred_frame, y_true_frame))
            # framedEnloss.append(SingleFrameDeltaEnLoss(y_pred_frame, y_true_frame, i0, i1))
            losses.append(SingleFrameMAELoss(y_pred_frame, y_true_frame)+SingleFrameDeltaEnLoss(y_pred_frame, y_true_frame, i0, i1))
        return FrameWeightedLoss(losses)

    def test_loss(self, y_pred, y_true):
        return F.l1_loss(y_pred, y_true)
    
    def train_once(self, i1, i2, time, it, fy):
        i1 = i1.float().to(self.device)
        i2 = i2.float().to(self.device)
        it = it.float().to(self.device)
        fy = fy.float().to(self.device)
        
        pred = self.network(i1, i2, time, fy, configs.trainframes)
        loss = self.train_loss(pred, it, i1, i2)
        return loss
    
    def test(self, testset, dataloader_eval):
        self.network.eval()
        predlist = []
        truelist = []
        with torch.no_grad():
            for img, imggt, time, fy, fygt in dataloader_eval:
                i1 = img[:,:self.Cin]
                i2 = img[:,self.Cin:self.Cin*2]
                it = imggt
                
                pred = self.network(i1.float().to(self.device), i2.float().to(self.device), time, fy.float().to(self.device), self.interinterval - 1)
                predlist.append(pred)
                truelist.append(it.float().to(self.device))
                
            predlist =torch.cat(predlist, dim=0)
            truelist =torch.cat(truelist, dim=0)
            testloss = self.test_loss(predlist, truelist)
        return testloss
    
    def train(self, dataset_train, dataset_eval, chk_path):
        # torch.manual_seed(0)
        printwrite(file, 'loading train dataloader')
        dataloader_train = DataLoader(dataset_train, batch_size=self.configs.batch_size, shuffle=True)
        printwrite(file, 'loading eval dataloader')
        dataloader_eval = DataLoader(dataset_eval, batch_size=self.configs.batch_size_test, shuffle=False)
        count = 0
        best = math.inf
        self.network.train()
        for i in range(self.configs.num_epochs):
            printwrite(file, '\nepoch: {0}'.format(i + 1))
            j = 0         
            data_iter = iter(dataloader_train)
            self.opt.zero_grad()
            while j < len(dataloader_train):
                j += 1
                self.network.train()
                img, imggt, time, fy, fygt = next(data_iter)            

                i1 = img[:,:self.Cin]
                i2 = img[:,self.Cin:self.Cin*2]
                it = imggt
                
                loss = self.train_once(i1 ,i2, time, it, fy)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                if (j + 1) % self.configs.display_interval == 0:
                    printwrite(file, 'batch training loss: {:.4f}'.format(loss))
                if (j + 1) % (self.configs.display_interval * configs.eval_interval) == 0:
                    loss = self.test(dataset_eval, dataloader_eval)
                    printwrite(file, 'batch eval loss: {:.4f}'.format(loss))
                    if loss < best:
                        count = 0
                        printwrite(file, 'eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss))           
                        self.save_model(chk_path)
                        best = loss
                        
            printwrite(file, 'epoch eval loss: {:.4f}'.format(loss))
            if loss >= best:
                count += 1
                printwrite(file, 'eval loss is not reduced for {} epoch'.format(count))
                printwrite(file, 'best is {} until now'.format(best))
            else:
                count = 0
                printwrite(file, 'eval loss is reduced from {:.5f} to {:.5f}, saving model'.format(best, loss))
                self.save_model(chk_path)
                best = loss
            self.save_model('exp/last.chk')
    def save_configs(self, config_path):
        with open(config_path, 'wb') as path:
            pickle.dump(self.configs, path)
    def save_model(self, path):
        torch.save({'net': self.network.state_dict()}, path)
########################################################################################################################
if __name__ == '__main__':
    
    dataset_train = TimeDownScalingDataset(configs.train_path, configs.dims, configs.interinterval, train= True)
    # dataset_train.indexs = dataset_train.indexs[::25]
    print(dataset_train.GetDataShape())

    dataset_eval = TimeDownScalingDataset(configs.val_path, configs.dims, configs.interinterval, train= False)
    # dataset_eval.indexs = dataset_eval.indexs[::25]
    print(dataset_eval.GetDataShape())
    trainer = Trainer(configs)
    trainer.save_configs('exp/config_train.pkl')

    # model = torch.load('exp/SRNet_200e.chk')
    # trainer.network.load_state_dict(model['net'])
    
    trainer.train(dataset_train, dataset_eval, 'exp/SRNet.chk')