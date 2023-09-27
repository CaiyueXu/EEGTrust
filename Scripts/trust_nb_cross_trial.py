import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.linalg import svd
from tensorboardX import SummaryWriter
import logging
import os
import random
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets.constants.emotion_recognition.deta import \
    DETA_CHANNEL_LOCATION_DICT
from torcheeg.datasets import NumpyDataset
from torcheeg.model_selection import KFold, train_test_split
from torcheeg.models import CCNN
import csv
from torcheeg.datasets.constants.emotion_recognition.deta import DETA_CHANNEL_LIST
from torcheeg.utils import plot_raw_topomap
from torcheeg.utils import plot_signal
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from torchmetrics.classification import BinaryF1Score

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#数据主要包括脑电信号序列，结果
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_idx, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y = batch[1].to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(X)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return loss

def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")

    return correct, loss

X=np.array([])

# eeglab_raw_file = f'../remote-home/221810845/subs/sub01/sub01_epoch_eye.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]540
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub02/sub02_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,120,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

eeglab_raw_file = f'../remote-home/221810845/subs/sub03/sub03_epochs_eye.set'
epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
for epoch in epochs:
    data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
    data = data.reshape((64,120,250))
    data = data.transpose(1,0,2)
    X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub04/sub04_epochs1.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub05/sub05_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub06/sub06_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub07/sub07_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub08/sub08_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub09/sub09.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub10/sub10.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub11/sub11.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub12/sub12.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub13/sub13.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub14/sub14.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub15/sub15.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/221810845/subs/sub16/sub16.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# # Writer.add_figure('original',eeglab_raw.plot(),0)

X = X.reshape((1800, 64, 250))

#sub01的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y1 = np.concatenate((label1,label0,label0,label1,label0,label1,label0,label1,label1),axis = 0)


#sub02的epoch label
# label1 = np.ones(120, dtype = int)
# label0 = np.zeros(120, dtype = int)
# y2 = np.concatenate((label1,label0,label1,label0,label1,
#                      label1,label1,label1,label0,label0,
#                      label1,label0,label0,label0,label0),axis = 0)

#sub03的epoch label
label1 = np.ones(120, dtype = int)
label0 = np.zeros(120, dtype = int)
y3 = np.concatenate((label1,label1,label0,label1,label0,
                     label0,label0,label1,label1,label0,
                     label0,label0,label1,label0,label1),axis = 0)

# # sub04的label 1 改
# 数据集太不均衡了，所以F1和acc差很多
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y4 = np.concatenate((label1, label1,label1,label0,label1,
#                         label1,label1,label1,label0,label1,
#                         label1,label0,label0,label1,label0,
#                         label1,label0,label1,label0,label0,
#                         label1,label0,label0,label0,label0,
#                         label0,label1,label1,label0,label0,),axis = 0)

# sub05的label 改
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y5 = np.concatenate((label1, label1,label1,label1,label1,
#                         label1,label1,label1,label0,label0,
#                         label0,label0,label1,label0,label1,
#                         label0,label0,label1,label0,label1,
#                         label1,label0,label0,label1,label1,
#                         label0,label0,label0,label1,label0,),axis = 0)

# sub06的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y6 = np.concatenate((label1, label1,label1,label1,label1,
#                         label1,label0,label1,label0,label1,
#                         label0,label0,label0,label1,label0,
#                         label1,label1,label0,label0,label0,
#                         label1,label1,label0,label0,label0,
#                         label0,label0,label1,label0,label1,),axis = 0)

# sub07的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y7 = np.concatenate((label1, label1,label1,label0,label1,
#                         label1,label1,label1,label1,label1,
#                         label1,label0,label0,label0,label0,
#                         label1,label0,label1,label0,label0,
#                         label0,label1,label1,label0,label0,
#                         label0,label0,label0,label0,label0,),axis = 0)

# sub08的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y8 = np.concatenate((label0, label0,label1,label1,label1,
#                         label1,label0,label1,label1,label1,
#                         label1,label1,label0,label1,label1,
#                         label1,label1,label0,label0,label0,
#                         label0,label0,label1,label1,label0,
#                         label0,label0,label0,label0,label0,),axis = 0)

# sub09的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y9 = np.concatenate((label0, label0,label1,label1,label0,
#                         label1,label1,label1,label1,label1,
#                         label1,label1,label0,label0,label0,
#                         label0,label0,label1,label0,label1,
#                         label0,label0,label0,label1,label0,
#                         label1,label1,label0,label1,label0,),axis = 0)


# sub10的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y10 = np.concatenate((label0, label1,label0,label1,label1,
#                         label0,label0,label0,label1,label1,
#                         label1,label1,label0,label0,label1,
#                         label1,label0,label1,label0,label0,
#                         label0,label0,label0,label1,label1,
#                         label0,label1,label1,label1,label0,),axis = 0)


# sub11的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y11 = np.concatenate((label0, label1,label1,label1,label1,
#                         label0,label0,label1,label1,label0,
#                         label1,label1,label0,label0,label0,
#                         label1,label1,label0,label1,label0,
#                         label1,label1,label0,label0,label0,
#                         label0,label0,label0,label1,label1,),axis = 0)


# sub12的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y12 = np.concatenate((label1, label1,label1,label0,label1,
#                         label1,label1,label1,label0,label1,
#                         label0,label1,label1,label0,label0,
#                         label0,label0,label1,label0,label0,
#                         label0,label1,label1,label0,label0,
#                         label0,label0,label0,label1,label0,),axis = 0)


# sub13的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y13 = np.concatenate((label1, label1,label1,label1,label1,
#                         label0,label1,label0,label0,label1,
#                         label0,label1,label1,label1,label0,
#                         label0,label1,label0,label0,label0,
#                         label0,label1,label1,label0,label0,
#                         label1,label0,label1,label0,label0,),axis = 0)

# sub14的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y14 = np.concatenate((label1, label1,label0,label1,label1,
#                         label1,label0,label1,label1,label1,
#                         label1,label0,label1,label1,label0,
#                         label1,label1,label1,label0,label0,
#                         label0,label0,label0,label0,label0,
#                         label0,label0,label0,label0,label1,),axis = 0)


# sub15的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y15 = np.concatenate((label1, label1,label1,label1,label1,
#                         label0,label0,label1,label0,label0,
#                         label0,label1,label0,label1,label1,
#                         label1,label0,label1,label1,label0,
#                         label0,label0,label0,label0,label1,
#                         label0,label1,label1,label0,label0,),axis = 0)

# sub016的label
# label1 = np.ones(60, dtype = int)
# label0 = np.zeros(60, dtype = int)
# y16 = np.concatenate((label1, label1,label1,label1,label0,
#                         label1,label0,label1,label1,label0,
#                         label1,label1,label1,label1,label0,
#                         label0,label1,label1,label1,label0,
#                         label0,label1,label0,label0,label0,
#                         label0,label0,label0,label0,label0,),axis = 0)

# y = np.concatenate((y1,y3,y4),axis=0)

if __name__ == '__main__':

    seed_everything(42)

    os.makedirs("./tmp_out/trust_nb/sub03_session", exist_ok=True)

    logger = logging.getLogger('nb_sub03')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./tmp_out/trust_nb/sub03_session/nb_sub03.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # sub5
    # X_test = np.concatenate((X[60*1:60*3-1,:,:], X[60*11:60*12-1,:,:], X[60*15:60*16-1,:,:], X[60*20:60*21-1,:,:], X[60*22:60*23-1,:,:]),axis=0)
    # y_test = {'trust': np.concatenate((y5[60*1:60*3-1], y5[60*11:60*12-1], y5[60*15:60*16-1], y5[60*20:60*21-1], y5[60*22:60*23-1]),axis=0)}
    # X_train = np.concatenate((X[60*0:60*1-1,:,:], X[60*3:60*11-1,:,:], X[60*12:60*15-1,:,:], X[60*16:60*20-1,:,:], X[60*21:60*22-1,:,:], X[60*23:,:,:]),axis=0)
    # y_train = {'trust': np.concatenate((y5[60*0:60*1-1], y5[60*3:60*11-1], y5[60*12:60*15-1], y5[60*16:60*20-1], y5[60*21:60*22-1], y5[60*23:]),axis=0)}
    # sub03
    X_test = np.concatenate((X[120*4:120*5-1,:,:], X[120*8:120*9-1,:,:], X[120*10:120*11-1,:,:]),axis=0)
    y_test = {'trust': np.concatenate((y3[120*4:120*5-1], y3[120*8:120*9-1], y3[120*10:120*11-1]),axis=0)}
    X_train = np.concatenate((X[120*0:120*4-1,:,:], X[120*5:120*8-1,:,:], X[120*9:120*10-1,:,:], X[120*11:,:,:]),axis=0)
    y_train = {'trust': np.concatenate((y3[120*0:120*4-1], y3[120*5:120*8-1], y3[120*9:120*10-1], y3[120*11:]),axis=0)}
    # sub09
    # X_test = np.concatenate((X[60*0:60*1-1,:,:], X[60*2:60*3-1,:,:], X[60*8:60*9-1,:,:], X[60*11:60*12-1,:,:], X[60*18:60*19-1,:,:], X[60*22:60*23-1,:,:]),axis=0)
    # y_test = {'trust': np.concatenate((y9[60*0:60*1-1], y9[60*2:60*3-1], y9[60*8:60*9-1], y9[60*11:60*12-1], y9[60*18:60*19-1], y9[60*22:60*23-1]),axis=0)}
    # X_train = np.concatenate((X[60*1:60*2-1,:,:], X[60*3:60*8-1,:,:], X[60*9:60*11-1,:,:], X[60*12:60*18-1,:,:], X[60*19:60*22-1,:,:], X[60*23:,:,:]),axis=0)
    # y_train = {'trust': np.concatenate((y9[60*1:60*2-1], y9[60*3:60*8-1], y9[60*9:60*11-1], y9[60*12:60*18-1], y9[60*19:60*22-1], y9[60*23:]),axis=0)}
    # sub14
    # X_test = np.concatenate((X[60*5:60*6-1,:,:], X[60*9:60*11-1,:,:], X[60*18:60*19-1,:,:], X[60*20:60*22-1,:,:]),axis=0)
    # y_test = {'trust': np.concatenate((y14[60*5:60*6-1], y14[60*9:60*11-1], y14[60*18:60*19-1], y14[60*20:60*22-1]),axis=0)}
    # X_train = np.concatenate((X[60*0:60*5-1,:,:], X[60*6:60*9-1,:,:], X[60*11:60*18-1,:,:], X[60*19:60*20-1,:,:], X[60*22:,:,:]),axis=0)
    # y_train = {'trust': np.concatenate((y14[60*0:60*5-1], y14[60*6:60*9-1], y14[60*11:60*18-1], y14[60*19:60*20-1], y14[60*22:]),axis=0)}
    # sub15
    # X_test = np.concatenate((X[60*0:60*1-1,:,:], X[60*5:60*6-1,:,:], X[60*10:60*11-1,:,:],X[60*18:60*19-1,:,:], X[60*20:60*21-1,:,:], X[60*26:60*27-1,:,:]),axis=0)
    # y_test = {'trust': np.concatenate((y15[60*0:60*1-1], y15[60*5:60*6-1], y15[60*10:60*11-1], y15[60*18:60*19-1], y15[60*20:60*21-1], y15[60*26:60*27-1]),axis=0)}
    # X_train = np.concatenate((X[60*1:60*5-1,:,:], X[60*6:60*10-1,:,:], X[60*11:60*18-1,:,:], X[60*19:60*20-1,:,:],X[60*21:60*26-1,:,:], X[60*27:,:,:]),axis=0)
    # y_train = {'trust': np.concatenate((y15[60*1:60*5-1], y15[60*6:60*10-1], y15[60*11:60*18-1], y15[60*19:60*20-1], y15[60*21:60*26-1], y15[60*27:]),axis=0)}
    # sub16
    # X_test = np.concatenate((X[60*5:60*7-1,:,:], X[60*16:60*17-1,:,:], X[60*23:60*24-1,:,:]),axis=0)
    # y_test = {'trust': np.concatenate((y16[60*5:60*7-1], y16[60*16:60*17-1],y16[60*23:60*24-1]),axis=0)}
    # X_train = np.concatenate((X[60*0:60*5-1,:,:], X[60*7:60*16-1,:,:], X[60*17:60*23-1,:,:],  X[60*24:,:,:]),axis=0)
    # y_train = {'trust': np.concatenate((y16[60*0:60*5-1], y16[60*7:60*16-1], y16[60*17:60*23-1], y16[60*24:]),axis=0)}
    #读数据到内存
    train_dataset = NumpyDataset(X=X_train,
                        y=y_train,
                        io_path=f'./tmp_out/trust_nb/sub03_session/train',
                        # offline_transform=transforms.Compose(
                        #     [transforms.Concatenate([
                        #         transforms.BandDifferentialEntropy(),
                        #     transforms.BandPowerSpectralDensity()
                        #     ]),
                        #     transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                        # ]),
                        offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(sampling_rate=250), 
                              #transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                          ]),
                        online_transform=transforms.ToTensor(),
                        # online_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
                        label_transform=transforms.Compose([
                            transforms.Select('trust'),
                            #transforms.Binary(4.0),
                        ]),
                        num_worker=2,
                        num_samples_per_worker=50)
    test_dataset = NumpyDataset(X=X_train,
                        y=y_train,
                        io_path=f'./tmp_out/trust_nb/sub03_session/test',
                        # offline_transform=transforms.Compose(
                        #     [transforms.Concatenate([
                        #         transforms.BandDifferentialEntropy(),
                        #     transforms.BandPowerSpectralDensity()
                        #     ]),
                        #     transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                        # ]),
                        offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(sampling_rate=250), 
                              #transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                          ]),
                        online_transform=transforms.ToTensor(),
                        # online_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
                        label_transform=transforms.Compose([
                            transforms.Select('trust'),
                            #transforms.Binary(4.0),
                        ]),
                        num_worker=2,
                        num_samples_per_worker=50)
    # np.savetxt("save/mydataset.csv", dataset[0], delimiter=",")
    # np.savetxt("save/mylabels.csv", dataset[1], delimiter=",")
    # img = plot_raw_topomap(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=128)
    # img2 = plot_signal(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=250)
    # # Writer.add_figure('topomap',img,0)
    # Writer.add_figure('topomap',img,0)
    # Writer.close()
    # EEG signal (torch.Tensor[32, 4]),
    # coresponding baseline signal (torch.Tensor[32, 4]),
    # label (int)
    # k_fold = KFold(n_splits=10, split_path=f'./tmp_out/trust_nb/sub16_session/split', shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.CrossEntropyLoss()#交叉熵损失，适用于分类器训练
    batch_size = 5120
    #每次给model的样本个数，每一个batch之后更新参数；如果太小，可能导致梯度下降不明显，收敛速度慢，如果太大，容易陷入局部最优

    test_accs = []
    test_losses = []

    #分别遍历训练数据集和测试数据集进行训练和测试
    #这里的i应该是32x10=320个一共
    test_accs = []
    test_f1_scores = []

    # for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):

        # model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
        #可是这个模型也是确定的呀，
        #9x9与32通道是对应的，对于64通道是否可以直接修改9，9x9倒是也可以满足64通道，就是位置对应怎么实现呢？
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#数据加载器，训练集加载
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)#验证集加载
    #model = svm.SVC(gamma='auto',kernel='poly',C=100000,max_iter=10000)
    model=GaussianNB(var_smoothing=1e-2)
    for batch_idx, batch in enumerate(train_loader):
        train_X = batch[0].to(device).cpu().reshape(-1,256)
        train_y = batch[1].to(device).cpu()

    # print(train_X.shape)
    # print(train_y.shape)
    
    scaler = StandardScaler()
    train_X  = scaler.fit_transform(train_X)
    model.fit(train_X,train_y)

    for batch_idx, batch in enumerate(test_loader):
        test_X = batch[0].to(device).cpu().reshape(-1,256)
        test_y = batch[1].to(device).cpu()

    # print(test_X.shape)
    # print(test_y.shape)
    test_X = scaler.fit_transform(test_X)
    pre_rst=model.predict(test_X)

    val_acc=np.sum((pre_rst == test_y.numpy()))/pre_rst.shape
    f1_score=BinaryF1Score()
    test_f1_score = f1_score(torch.tensor(pre_rst),torch.tensor(test_y.numpy()))
    # test_f1_score = f1_score(pre_rst,test_y.numpy())
    logger.info(f"Test Error {0}: \n Accuracy: {(100*val_acc[0]):>0.1f}%, f1_score: {test_f1_score:>8f}")
    # #当前分割训练出的准确率和损失
    test_accs.append(val_acc)
    test_f1_scores.append(test_f1_score)
    logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%,  f1_score:{np.mean(test_f1_scores):>8f}")