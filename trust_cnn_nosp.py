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
from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryROC, BinaryAUROC

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

def valid(dataloader, model, loss_fn, iftest):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct, = 0, 0
    preds=[]
    predss=[]
    ys=[]
    fpr=[]
    tpr=[]
    auc=[]
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y = batch[1].to(device)

            pred = model(X)
            loss += loss_fn(pred, y).item()
            preds.extend(pred.argmax(1).cpu().numpy())
            predss.extend(pred.cpu().numpy())
            ys.extend(y.cpu().numpy())

    loss /= num_batches
    f1_score=BinaryF1Score()
    acc = BinaryAccuracy()
    print(f"Avg loss: {loss:>8f}, f1_score:{f1_score(torch.tensor(np.array(preds)),torch.tensor(np.array(ys))):>8f}, acc:{acc(torch.tensor(np.array(preds)),torch.tensor(np.array(ys))):>8f} \n")
    if (iftest):
        metric_roc = BinaryROC(thresholds=None)
        metric_auc = BinaryAUROC(threshholds=None)
        # print(predss)
        fpr,tpr, threshholds = metric_roc(torch.tensor(np.array(predss))[:,1],torch.tensor(np.array(ys)))
        auc = metric_auc(torch.tensor(np.array(predss))[:,1],torch.tensor(np.array(ys)))
        # fprs.extend(fpr)
        # tprs.extend(tpr)
        # aucs.extend(auc)
        # fig_, ax_ = plt.subplots()
        # ax_.plot(fpr,tpr)
        # Writer.add_figure('ROC',fig_,0)
    return fpr, tpr, auc, acc(torch.tensor(np.array(preds)),torch.tensor(np.array(ys))), loss, f1_score(torch.tensor(np.array(preds)),torch.tensor(np.array(ys)))


X=np.array([])

# eeglab_raw_file = f'./subs/sub01/sub01_epoch_eye.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]540
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub02/sub02_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,120,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub03/sub03_epochs_eye.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,120,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub04/sub04_epochs1.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)


eeglab_raw_file = f'./subs/sub05/sub05_epochs.set'
epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
for epoch in epochs:
    data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
    data = data.reshape((64,60,250))
    data = data.transpose(1,0,2)
    X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub06/sub06_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub07/sub07_epochs.set'
# epochs = mne.io.read_epochs_eeglab(eeglab_raw_file)
# for epoch in epochs:
#     data=epoch[:,:] #去基线的话要把前3s去掉 data=epoch[:,750:]1800
#     data = data.reshape((64,60,250))
#     data = data.transpose(1,0,2)
#     X = np.append(X, data)

# eeglab_raw_file = f'./subs/sub08/sub08_epochs.set'
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
# label1 = np.ones(120, dtype = int)
# label0 = np.zeros(120, dtype = int)
# y3 = np.concatenate((label1,label1,label0,label1,label0,
#                      label0,label0,label1,label1,label0,
#                      label0,label0,label1,label0,label1),axis = 0)

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
label1 = np.ones(60, dtype = int)
label0 = np.zeros(60, dtype = int)
y5 = np.concatenate((label1, label1,label1,label1,label1,
                        label1,label1,label1,label0,label0,
                        label0,label0,label1,label0,label1,
                        label0,label0,label1,label0,label1,
                        label1,label0,label0,label1,label1,
                        label0,label0,label0,label1,label0,),axis = 0)

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
#                         label1,label1,label0,label0,label1,
#                         label1,label1,label0,label0,label0,
#                         label0,label0,label1,label1,label0,
#                         label0,label0,label0,label0,label0,),axis = 0)

# y = np.concatenate((y1,y3,y4,y5),axis=0)
y = y5
y = {
    'trust': y,
}

if __name__ == '__main__':

    seed_everything(42)

    os.makedirs("./tmp_out/trust_ccnn_nosp/sub05", exist_ok=True)

    logger = logging.getLogger('ccnn_sub05')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('./tmp_out/trust_ccnn_nosp/sub05/ccnn_sub05.log')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    Writer = SummaryWriter('./tmp_out/trust_ccnn_nosp/sub05')
    # Mock 100 EEG samples. Each EEG signal contains a signal of length 1 s at a frequency of 128 sampled by 32 electrodes.
    # X = np.random.randn(100, 32, 128)#我的64x30000，可以分成120x64x250，这120个sample对应同一个label，一共120x15=1800个sample
    
    # Mock 100 labels, denoting valence and arousal of subjects during EEG recording.


    #读数据到内存
    dataset = NumpyDataset(X=X,
                        y=y,
                        io_path=f'./tmp_out/trust_ccnn_nosp/sub05/deta',
                        # offline_transform=transforms.Compose(
                        #     [transforms.Concatenate([
                        #         transforms.BandDifferentialEntropy(),
                        #     transforms.BandPowerSpectralDensity()
                        #     ]),
                        #     transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                        # ]),
                        offline_transform=transforms.Compose([
                              transforms.BandDifferentialEntropy(sampling_rate=250),
                              transforms.RandomChannelShuffle(),
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

    # img = plot_raw_topomap(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=128)
    # img2 = plot_signal(dataset[0][0].squeeze(), channel_list=DETA_CHANNEL_LIST, sampling_rate=250)
    # # Writer.add_figure('topomap',img,0)
    # Writer.add_figure('topomap',img,0)

    k_fold = KFold(n_splits=10, split_path=f'./tmp_out/trust_ccnn_nosp/sub05/split', shuffle=True)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
 
    device = "cuda"
    # loss_fn = nn.CrossEntropyLoss(weight = torch.tensor([1.0,2.0]))#交叉熵损失，适用于分类器训练
    # loss_fn.cuda()
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 256
    #每次给model的样本个数，每一个batch之后更新参数；如果太小，可能导致梯度下降不明显，收敛速度慢，如果太大，容易陷入局部最优

    test_accs = []
    test_losses = []
    test_f1_scores = []
    # fprs=[]
    tprs=[]
    aucs=[]
    mean_fpr = np.linspace(0,1,100)
    #分别遍历训练数据集和测试数据集进行训练和测试
    #这里的i应该是32x10=320个一共
    for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):

        model = CCNN(num_classes=2, in_channels=4, grid_size=(9, 9)).to(device)
        #可是这个模型也是确定的呀，
        #9x9与32通道是对应的，对于64通道是否可以直接修改9，9x9倒是也可以满足64通道，就是位置对应怎么实现呢？
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-3)  # official: weight_decay=5e-1

        #前面是训练集和测试集的分割，这里是从训练集中再分割出训练集和验证集
        #验证集和交叉验证用于模型选择
        #交叉验证：从训练集中选出一部分样本，每次训练对应的都是同一个train_set，通过不同的分割方法分割为训练集和验证集
        #训练集和验证集的划分
        #把前面分的被试和折的训练集再分为训练集和验证集，对应的train和test比例为8:2，还是索引到起始点和终止点
        #每个起始点和终止点对应一个样本，每次dataloader加载batch_size个样本
        train_dataset, val_dataset = train_test_split(train_dataset,
                                                              test_size=0.1,#测试集比例
                                                              split_path=f'./tmp_out/trust_ccnn_nosp/sub05/split{i}',#第i次分割，会训练出第i个模型
                                                              shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#数据加载器，训练集加载
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#验证集加载

        epochs = 5000
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
            fpr,tpr,auc,val_acc, val_loss, val_f1_score = valid(val_loader, model, loss_fn, iftest=False)
            Writer.add_scalar('val_acc',val_acc,t)
            Writer.add_scalar('val_loss',val_loss,t)
            # Writer.add_scalar('f1_score',val_f1_score,t)
            #通过验证集做模型选择，验证集用于模型选择，从50个epoch中选出最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'./tmp_out/trust_ccnn_nosp/sub05/model{i}.pt')
        #加载测试集
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        #加载训练好的模型，也就是截止目前，这个数据分割情况下最优的模型
        model.load_state_dict(torch.load(f'./tmp_out/trust_ccnn_nosp/sub05/model{i}.pt'))
        #测试结果
        fpr,tpr,auc,test_acc, test_loss, test_f1_score = valid(test_loader, model, loss_fn, iftest=True)
        logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}, f1_score: {test_f1_score:>8f}")
        #当前分割训练出的准确率和损失
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        test_f1_scores.append(test_f1_score)
        interp_tpr = np.interp(mean_fpr,fpr,tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc)
    tprs = np.array(tprs)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs,axis=0)
    np.savetxt("./tmp_out/trust_ccnn_nosp/sub05/fprs.csv", mean_fpr, delimiter=',')
    np.savetxt("./tmp_out/trust_ccnn_nosp/sub05/tprs_std.csv", std_tpr, delimiter=',')
    np.savetxt("./tmp_out/trust_ccnn_nosp/sub05/tprs.csv", mean_tpr, delimiter=',')
    np.savetxt("./tmp_out/trust_ccnn_nosp/sub05/aucs.csv", aucs, delimiter=',')
    Writer.close()
    #前面保存的是每次分割的准确率和损失，也就是每个subject分割10次，共获得320个分割，分别训练得到准确率和损失，最后一步求这些获得的模型的平均准确率和损失
    logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}, f1_score:{np.mean(test_f1_scores):>8f}")