import numpy as np
from datetime import datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def makesampleindex(df, interval=6, train=True):
    xindlist = []
    yindlist = []
    for i in range(0, len(df['day'])-interval-1):
        d0 = datetime.strptime(
            str(df['year'][i]) + '-' + str(df['month'][i]) + '-' + str(df['day'][i]) + '-' + str(df['hour'][i]) + '-' + str(
                df['minute'][i]), '%Y-%m-%d-%H-%M')
        d1 = datetime.strptime(
            str(df['year'][i+interval+1]) + '-' + str(df['month'][i+interval+1]) + '-' + str(df['day'][i+interval+1]) + '-' + str(df['hour'][i+interval+1]) + '-' + str(
                df['minute'][i+interval+1]), '%Y-%m-%d-%H-%M')
        if (d1-d0).seconds==(interval+1)*3600:
            xindlist.append([i, i+interval])
            l = []
            for t in range(1,interval):
                l.append(i+t)
            if train:
                yindlist.append(l[1::2]) #
            else:
                yindlist.append(l)
    return xindlist, yindlist

class TimeDownScalingDataset(Dataset):
    def __init__(self, dataset_path = '24composite.npy', dims =['t2m', 'sp', 'u10'], interval = 6, train=True):
        raw_data = np.load(dataset_path, allow_pickle=True).item()
        self.xindl, self.yindl = makesampleindex(raw_data, interval, train)
        self.meta_data = []
        for dim in dims:
            self.meta_data.append(raw_data[dim])
        self.meta_data = np.stack(self.meta_data, 1)[:,:,:64,:64]
        self.meta_fy4b = raw_data['fy'][:,:,:64,:64]
        self.meta_time = np.stack([raw_data['year'], raw_data['month'], raw_data['day'], raw_data['hour'], raw_data['minute']],1)
        self.interval = interval

    def __len__(self):
        return len(self.yindl)
    def GetDataShape(self):
        return len(self.yindl)
    def __getitem__(self, index):
        xind = self.xindl[index]
        yind = self.yindl[index]

        img01 = self.meta_data[xind]
        img0 = torch.from_numpy(img01[0])
        img1 = torch.from_numpy(img01[1])
        imggt = torch.from_numpy(self.meta_data[yind])
        fy01 = torch.from_numpy(self.meta_fy4b[xind])
        fygt = torch.from_numpy(self.meta_fy4b[yind])

        time = torch.from_numpy(self.meta_time[xind])

        return torch.cat((img0, img1),0), imggt, time, fy01, fygt

if __name__ == "__main__":

    dataset_path = '../Dataset/24composite_onezero.npy'
    dims = ['t2m', 'sp', 'u10']
    inter_frames = 4

    d = TimeDownScalingDataset(dataset_path, dims, inter_frames)
    train_data = DataLoader(d, batch_size=4, pin_memory=True, drop_last=True)

    for i, k in enumerate(train_data):
        print(i)

    np.save('im01.npy', k[0][0])
    np.save('imgt.npy', k[1][0])
    np.save('fy01.npy', k[2][0])
    np.save('fygt.npy', k[3][0])