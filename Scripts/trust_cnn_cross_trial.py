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
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassConfusionMatrix

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
    loss, correct, = 0, 0
    preds=[]
    ys=[]
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            preds.extend(pred.argmax(1).cpu().numpy())
            ys.extend(y.cpu().numpy())

    loss /= num_batches
    correct /= size
    f1_score=MulticlassF1Score(num_classes = 2)
    acc = MulticlassAccuracy(num_classes = 2)
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f}, f1_score:{f1_score(torch.tensor(preds),torch.tensor(ys)):>8f}, acc:{acc(torch.tensor(preds),torch.tensor(ys)):>8f} \n")

    return correct, loss, f1_score(torch.tensor(preds),torch.tensor(ys))

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

# eeglab_raw_file = f'../remote-home/1710824/subs/sub03/sub03_epochs_eye.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,120,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'../remote-home/1710824/subs/sub04/sub04_epochs1.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

eeglab_raw_file = f'../remote-home/221810845/subs/sub05/sub05_epochs.set'
epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
for epoch in epochs:
    data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
    data = data.reshape((64,60,250))
    data = data.transpose(1,0,2)
    X = np.append(X, data)

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

X = X.reshape((1800, 64, 250))

label2 = 2 * np.ones(60, dtype = int)
label1 = np.ones(60, dtype = int)
label0 = np.zeros(60, dtype = int)

# label2 = 2 * np.ones(120, dtype = int)
# label1 = np.ones(120, dtype = int)
# label0 = np.zeros(120, dtype = int)

#sub01的label
# y1 = np.concatenate((label2,label1,label0,label1,label0,label2,label0,label1,label2),axis = 0)

# #sub02的label
# y2 = np.concatenate((label2,label1,label2,label1,label2,
#                      label1,label2,label1,label0,label0,
#                      label2,label0,label0,label1,label0),axis = 0)


#sub03的epoch label
# y3 = np.concatenate((label2,label1,label0,label2,label0,
#                      label1,label0,label2,label2,label0,
#                      label1,label0,label2,label1,label1),axis = 0)


# sub04的label
# y4 = np.concatenate((label2, label1,label2,label0,label2,
#                         label2,label2,label1,label1,label1,
#                         label2,label0,label1,label2,label0,
#                         label2,label0,label1,label1,label0,
#                         label2,label0,label0,label1,label0,
#                         label1,label2,label1,label0,label0,),axis = 0)


# sub05的label
label1 = np.ones(60, dtype = int)
label0 = np.zeros(60, dtype = int)
y5 = np.concatenate((label1, label1,label1,label1,label1,
                        label1,label1,label1,label0,label0,
                        label0,label0,label1,label0,label1,
                        label0,label0,label1,label0,label1,
                        label1,label0,label0,label1,label1,
                        label0,label0,label0,label1,label0,),axis = 0)


# sub06的label
# y6 = np.concatenate((label2, label1,label2,label2,label2,
#                         label1,label0,label2,label0,label1,
#                         label1,label1,label1,label2,label0,
#                         label1,label2,label0,label0,label0,
#                         label2,label2,label0,label0,label1,
#                         label0,label1,label1,label0,label2,),axis = 0)


# sub07的label
# y7 = np.concatenate((label1, label2,label2,label1,label2,
#                         label2,label2,label2,label1,label2,
#                         label1,label1,label1,label0,label1,
#                         label2,label1,label1,label0,label0,
#                         label1,label2,label2,label0,label0,
#                         label0,label0,label0,label0,label0,),axis = 0)


# sub08的label
# y8 = np.concatenate((label1, label1,label2,label2,label1,
#                         label2,label0,label1,label1,label2,
#                         label2,label2,label0,label1,label2,
#                         label2,label1,label1,label0,label0,
#                         label1,label0,label2,label2,label0,
#                         label0,label0,label0,label0,label1,),axis = 0)


# sub09的label
# y9 = np.concatenate((label1, label0,label2,label2,label0,
#                         label2,label2,label2,label2,label2,
#                         label1,label1,label1,label0,label0,
#                         label1,label1,label2,label0,label1,
#                         label0,label0,label0,label2,label1,
#                         label2,label1,label0,label1,label0,),axis = 0)



# sub10的label
# y10 = np.concatenate((label1, label1,label0,label1,label2,
#                         label1,label0,label1,label2,label2,
#                         label2,label2,label0,label0,label2,
#                         label2,label0,label1,label1,label0,
#                         label0,label0,label0,label1,label1,
#                         label0,label2,label2,label2,label1,),axis = 0)



# sub11的label
# y11 = np.concatenate((label0, label2,label1,label2,label2,
#                         label1,label0,label2,label1,label1,
#                         label1,label2,label0,label0,label0,
#                         label2,label2,label0,label2,label0,
#                         label2,label2,label1,label1,label0,
#                         label1,label0,label0,label1,label1,),axis = 0)


# sub12的label
# y12 = np.concatenate((label2, label2,label1,label0,label2,
#                         label2,label2,label2,label1,label1,
#                         label1,label1,label2,label1,label0,
#                         label0,label0,label1,label0,label0,
#                         label0,label2,label2,label1,label1,
#                         label0,label1,label0,label2,label0,),axis = 0)


# sub13的label
# y13 = np.concatenate((label2, label2,label2,label1,label2,
#                         label1,label2,label1,label0,label1,
#                         label0,label2,label1,label2,label1,
#                         label0,label2,label0,label0,label0,
#                         label1,label1,label1,label0,label0,
#                         label2,label1,label2,label0,label0,),axis = 0)


# sub14的label
# y14 = np.concatenate((label2, label2,label0,label2,label1,
#                         label2,label1,label2,label2,label1,
#                         label2,label0,label2,label2,label1,
#                         label1,label2,label1,label0,label0,
#                         label0,label0,label0,label1,label0,
#                         label1,label0,label0,label1,label1,),axis = 0)


# sub15的label
# y15 = np.concatenate((label2, label2,label2,label1,label1,
#                         label0,label1,label1,label0,label0,
#                         label0,label2,label0,label2,label1,
#                         label2,label0,label2,label2,label0,
#                         label0,label1,label1,label0,label1,
#                         label0,label2,label2,label1,label1,),axis = 0)


# sub016的label
# y16 = np.concatenate((label2, label2,label2,label2,label0,
#                         label2,label0,label1,label1,label0,
#                         label1,label1,label2,label2,label0,
#                         label0,label2,label1,label2,label0,
#                         label1,label2,label1,label0,label0,
#                         label0,label0,label1,label1,label1,),axis = 0)

# y = np.concatenate((y1,y3,y4,y5),axis=0)
y = y5
y = {
    'trust': y,
}

if __name__ == '__main__':

    seed_everything(42)

    os.makedirs("./tmp_out/trust_ccnn/sub05-session", exist_ok=True)

    logger = logging.getLogger('ccnn_sub05')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./tmp_out/trust_ccnn/sub05-session/ccnn_sub05.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    Writer = SummaryWriter('./tmp_out/trust_ccnn/sub05-session')
    # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a frequency of 128 sampled by 32 electrodes.
    # X = np.random.randn(100, 32, 128)#我的64x30000，可以分成120x64x250，这120个sample对应同一个label，一共120x15=1800个sample
    
    # Mock 100 labels, denoting valence and arousal of subjects during EEG recording.

    X_test = np.concatenate((X[60*1:60*3-1,:,:], X[60*11:60*12-1,:,:], X[60*15:60*16-1,:,:], X[60*20:60*21-1,:,:], X[60*22:60*23-1,:,:]),axis=0)
    y_test = {'trust': np.concatenate((y5[60*1:60*3-1], y5[60*11:60*12-1], y5[60*15:60*16-1], y5[60*20:60*21-1], y5[60*22:60*23-1]),axis=0)}
    print(y_test)

    X_train = np.concatenate((X[60*0:60*1-1,:,:], X[60*3:60*11-1,:,:], X[60*12:60*15-1,:,:], X[60*16:60*20-1,:,:], X[60*21:60*22-1,:,:], X[60*23:,:,:]),axis=0)
    y_train = {'trust': np.concatenate((y5[60*0:60*1-1], y5[60*3:60*11-1], y5[60*12:60*15-1], y5[60*16:60*20-1], y5[60*21:60*22-1], y5[60*23:]),axis=0)}
    # X_train = X[60*3:60*24-1,:,:]
    # y_train = {'trust': y16[60*3:60*24-1]}

    #读数据到内存
    train_dataset = NumpyDataset(X=X_train,
                        y=y_train,
                        io_path=f'./tmp_out/trust_ccnn/sub05-session/train',
                        # offline_transform=transforms.Compose(
                        #     [transforms.Concatenate([
                        #         transforms.BandDifferentialEntropy(),
                        #     transforms.BandPowerSpectralDensity()
                        #     ]),
                        #     transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                        # ]),
                        offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(sampling_rate=250), 
                              transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                          ]),
                        online_transform=transforms.ToTensor(),
                        # online_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
                        label_transform=transforms.Compose([
                            transforms.Select('trust'),
                            #transforms.Binary(4.0),
                        ]),
                        num_worker=2,
                        num_samples_per_worker=50)

    test_dataset = NumpyDataset(X=X_test,
                            y=y_test,
                            io_path=f'./tmp_out/trust_ccnn/sub05-session/test',

                            offline_transform=transforms.Compose([
                                transforms.MeanStdNormalize(),
                                transforms.BandDifferentialEntropy(sampling_rate=250),
                                # transforms.BaselineRemoval(),
                                transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                            ]),
                            online_transform=transforms.ToTensor(),
                            # online_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
                            label_transform=transforms.Compose([
                                transforms.Select('trust'),
                                #transforms.Binary(4.0),
                            ]),
                            num_worker=2)

    # img = plot_raw_topomap(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=128)
    # img2 = plot_signal(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=250)
    # # Writer.add_figure('topomap',img,0)
    # Writer.add_figure('topomap',img,0)


    # device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(0)
    device = "cuda"
    loss_fn = nn.CrossEntropyLoss()#交叉熵损失，适用于分类器训练
    batch_size = 256
    #每次给model的样本个数，每一个batch之后更新参数；如果太小，可能导致梯度下降不明显，收敛速度慢，如果太大，容易陷入局部最优

    test_accs = []
    test_losses = []
    test_f1_scores = []
    test_metrics = []

    #分别遍历训练数据集和测试数据集进行训练和测试
    #这里的i应该是32x10=320个一共
    # for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):

    model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
    #可是这个模型也是确定的呀，
    #9x9与32通道是对应的，对于64通道是否可以直接修改9，9x9倒是也可以满足64通道，就是位置对应怎么实现呢？

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-3)  # official: weight_decay=5e-1

    #前面是训练集和测试集的分割，这里是从训练集中再分割出训练集和验证集
    #验证集和交叉验证用于模型选择
    #交叉验证：从训练集中选出一部分样本，每次训练对应的都是同一个train_set，通过不同的分割方法分割为训练集和验证集
    #训练集和验证集的划分
    #把前面分的被试和折的训练集再分为训练集和验证集，对应的train和test比例为8:2，还是索引到起始点和终止点
    #每个起始点和终止点对应一个样本，每次dataloader加载batch_size个样本
    # train_dataset, val_dataset = train_test_split(train_dataset,
    #                                                         test_size=0.1,#测试集比例
    #                                                         split_path=f'./tmp_out/trust_ccnn/sub16-session/split{0}',#第i次分割，会训练出第i个模型
    #                                                         shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#数据加载器，训练集加载
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#验证集加载

    epochs = 500
    #训练集中的样本全部训练的次数
    #假如有100个样本，
    # batch_size=20 一个iteration迭代训练的样本为20个，计算损失函数并更新参数
    # iteration=5 一个epoch的迭代次数。实际大小为（样本数=batch_size x iteration)根据batch_size，把所有样本遍历一次的迭代次数
    # epoch=N 1个epoch为所有样本训练一次，实际中需要多训练和更新几次
    best_val_acc = 0.0
    for t in range(epochs):#每个epoch也就是遍历了所有样本，对比并更新一下最优的模型，每个分割中通过50个epoch的训练保存一个最佳模型
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_loader, model, loss_fn, optimizer)
        Writer.add_scalar('train_loss',train_loss,t)
        # val_acc, val_loss, val_f1_score, metric = valid(val_loader, model, loss_fn)
        # Writer.add_scalar('val_acc',val_acc,t)
        # Writer.add_scalar('val_loss',val_loss,t)
        # Writer.add_scalar('f1_score',val_f1_score,t)
        #通过验证集做模型选择，验证集用于模型选择，从50个epoch中选出最佳模型
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        torch.save(model.state_dict(), f'./tmp_out/trust_ccnn/sub05-session/model{0}.pt')
    #加载测试集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #加载训练好的模型，也就是截止目前，这个数据分割情况下最优的模型
    model.load_state_dict(torch.load(f'./tmp_out/trust_ccnn/sub05-session/model{0}.pt'))
    #测试结果
    test_acc, test_loss, test_f1_score = valid(test_loader, model, loss_fn)
    logger.info(f"Test Error {0}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}, f1_score: {test_f1_score:>8f}")
    #当前分割训练出的准确率和损失
    test_accs.append(test_acc)
    test_losses.append(test_loss)
    test_f1_scores.append(test_f1_score)
    Writer.close()
    #前面保存的是每次分割的准确率和损失，也就是每个subject分割10次，共获得320个分割，分别训练得到准确率和损失，最后一步求这些获得的模型的平均准确率和损失
    logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}, f1_score:{np.mean(test_f1_scores):>8f}")