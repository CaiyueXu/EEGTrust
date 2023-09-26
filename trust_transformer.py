"""Arjun et al\'s ViT with the AMIGOS Dataset
======================================
In this case, we introduce how to use TorchEEG to train a vision transformer proposed by Arjun et al. on the AMIGOS dataset for emotion classification.
"""

import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torcheeg import transforms
from torcheeg.datasets import AMIGOSDataset
from torcheeg.datasets.constants.emotion_recognition.amigos import \
    AMIGOS_CHANNEL_LOCATION_DICT
from torcheeg.model_selection import KFoldGroupbyTrial
from torcheeg.models import ArjunViT
from torcheeg.models import SimpleViT
from torcheeg.models import ViT
from torcheeg.trainers import ClassificationTrainer
import pandas as pd
import mne
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.linalg import svd
from tensorboardX import SummaryWriter
import logging
import os
import random
import torch.nn as nn
from torcheeg.datasets import NumpyDataset
from torcheeg.model_selection import KFold, train_test_split
import csv
from torcheeg.datasets.constants.emotion_recognition.deta import DETA_CHANNEL_LIST
from torcheeg.utils import plot_raw_topomap
from torcheeg.utils import plot_signal
from torcheeg.datasets.constants.emotion_recognition.deta import \
    DETA_CHANNEL_LOCATION_DICT
from torchmetrics.classification import BinaryF1Score,BinaryAccuracy, BinaryROC, BinaryAUROC
###############################################################################
# Pre-experiment Preparation to Ensure Reproducibility
# -----------------------------------------
# Use the logging module to store output in a log file for easy reference while printing it to the screen.
# 频域特征的transformer

os.makedirs('./tmp_out/trust_transformer/sub05', exist_ok=True)
logger = logging.getLogger('transformer_sub05')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
timeticks = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
file_handler = logging.FileHandler('./tmp_out/trust_transformer/sub05/transformer_sub05.log')
logger.addHandler(console_handler)
logger.addHandler(file_handler)
Writer = SummaryWriter('./tmp_out/trust_transformer/sub05')
###############################################################################
# Set the random number seed in all modules to guarantee the same result when running again.


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

###############################################################################
# Customize Trainer
# -----------------------------------------
# TorchEEG provides a large number of trainers to help complete the training of classification models, generative models and cross-domain methods. Here we choose the simplest classification trainer, inherit the trainer and overload the log function to save the log using our own defined method; other hook functions can also be overloaded to meet special needs.
#

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

        if batch_idx % 20 == 0:
            loss, current = loss.item(), batch_idx * batch_size
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss


# validation process
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
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            preds.extend(pred.argmax(1).cpu().numpy())
            predss.extend(pred.cpu().numpy())
            ys.extend(y.cpu().numpy())

    loss /= num_batches
    correct /= size
    f1_score=BinaryF1Score()
    acc = BinaryAccuracy()
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f}, f1_score:{f1_score(torch.tensor(preds),torch.tensor(ys)):>8f}, acc:{acc(torch.tensor(preds),torch.tensor(ys)):>8f} \n")
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

class MyClassificationTrainer(ClassificationTrainer):
    def log(self, *args, **kwargs):
        if self.is_main:
            logger.info(*args, **kwargs)


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

# y = np.concatenate((y1,y2),axis=0)
y = y5
y = {
    'trust': y,
}

######################################################################
# Building Deep Learning Pipelines Using TorchEEG
# -----------------------------------------
# Step 1: Initialize the Dataset
#
# We use the AMIGOS dataset supported by TorchEEG. Here, we set an EEG sample to 1 second long and include 128 data points. The baseline signal is 5 seconds long, cut into five, and averaged as the baseline signal for the trial. In offline preprocessing, all EEG signals are debaselined and normalized, and the preprocessed EEG signals are stored in the local IO. In online processing, all EEG signals are converted into Tensors for input into neural networks.
#

dataset = NumpyDataset(X=X,
                    y=y,
                    io_path=f'./tmp_out/trust_transformer/sub05/deta',
                    # offline_transform=transforms.Compose(
                    #     [transforms.Concatenate([
                    #         transforms.BandDifferentialEntropy(),
                    #     transforms.BandPowerSpectralDensity()
                    #     ]),
                    #     transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                    # ]),
                    offline_transform=transforms.Compose([
                            # transforms.MeanStdNormalize(),
                            # transforms.CWTSpectrum(), 
                            # transforms.Downsample(128),
                            transforms.BandDifferentialEntropy(sampling_rate=250),
                            transforms.ToGrid(DETA_CHANNEL_LOCATION_DICT)
                        ]),
                    online_transform=transforms.ToTensor(),
                    # online_transform=transforms.Compose([transforms.To2d(), transforms.ToTensor()]),
                    label_transform=transforms.Compose([
                        transforms.Select('trust'),
                        #transforms.Binary(4.0),
                    ]),
                    num_worker=2
                    # num_samples_per_worker=50
                    )


######################################################################
# .. warning::
#    If you use TorchEEG under the `Windows` system and want to use multiple processes (such as in dataset or dataloader), you should check whether :obj:`__name__` is :obj:`__main__` to avoid errors caused by multiple :obj:`import`.
#
# That is, under the :obj:`Windows` system, you need to:
#  .. code-block::
#
#    if __name__ == "__main__":
#        dataset = AMIGOSDataset(io_path='./tmp_out/examples_amigos_arjunvit/amigos',
#                         root_path='./tmp_in/data_preprocessed',
#                         offline_transform=transforms.Compose([
#                             transforms.BaselineRemoval(),
#                             transforms.MeanStdNormalize()
#                         ]),
#                         online_transform=transforms.ToTensor(),
#                         label_transform=transforms.Compose([
#                             transforms.Select('valence'),
#                             transforms.Binary(5.0)
#                         ]),
#                         io_mode='pickle',
#                         chunk_size=128,
#                         baseline_chunk_size=128,
#                         num_baseline=5,
#                         num_worker=4)
#        # the following codes
#
# .. note::
#    LMDB may not be optimized for parts of Windows systems or storage devices. If you find that the data preprocessing speed is slow, you can consider setting :obj:`io_mode` to :obj:`pickle`, which is an alternative implemented by TorchEEG based on pickle.

######################################################################
# Step 2: Divide the Training and Test samples in the Dataset
#
# Here, the dataset is divided using 5-fold cross-validation. In the process of division, we group according to the trial index, and every trial takes 4 folds as training samples and 1 fold as test samples. Samples across trials are aggregated to obtain training set and test set.
device = "cuda"
loss_fn = nn.CrossEntropyLoss()
batch_size = 256
k_fold = KFold(n_splits=10, split_path=f'./tmp_out/trust_transformer/sub05/split', shuffle=True)

######################################################################
# Step 3: Define the Model and Start Training
#
# We first use a loop to get the dataset in each cross-validation.
# In each cross-validation, we initialize the ArjunViT model, and define the hyperparameters. For example, each EEG sample contains 128 time points, we divide it into 4 patches, each patch contains 32 time points, etc.
#
# We then initialize the trainer and set the hyperparameters in the trained model, such as the learning rate, the equipment used, etc. The :obj:`fit` method receives the training dataset and starts training the model. The :obj:`test` method receives a test dataset and reports the test results. The :obj:`save_state_dict` method can save the trained model.
test_accs = []
test_losses = []
test_f1_scores = []
tprs=[]
aucs=[]
mean_fpr = np.linspace(0,1,100)

for i, (train_dataset, test_dataset) in enumerate(k_fold.split(dataset)):

    # 模型调复杂后效果变好
    model = ViT(chunk_size=4,#分几个频段
                  grid_size=(9, 9),
                  t_patch_size=1,
                  num_classes=2,
                  depth = 6,
                  heads=9,
                  hid_channels=128,
                  dropout=0.2).to(device)

    # Initialize the trainer and use the 0-th GPU for training, or set device_ids=[] to use CPU
    # trainer = MyClassificationTrainer(model=model,
    #                                   lr=1e-3,
    #                                   weight_decay=1e-4,#1e-4
    #                                   device_ids=[0])

    # Initialize several batches of training samples and test samples
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 1e-5)
    train_dataset, val_dataset = train_test_split(train_dataset,
                                                              test_size=0.1,#测试集比例
                                                              split_path=f'./tmp_out/trust_transformer/sub05/split{i}',#第i次分割，会训练出第i个模型
                                                              shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)

    # # Do 50 rounds of training
    # trainer.fit(train_loader, val_loader, num_epochs=700)
    # trainer.test(val_loader)
    # trainer.save_state_dict(
    #     f'./tmp_out/trust_transformer/sub05/model{i}.pt')
    epochs = 3000
    best_val_acc=0.0
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train(train_loader, model, loss_fn, optimizer)
        Writer.add_scalar('train_loss',loss,t)
        fpr,tpr, auc,val_acc, val_loss, f1_score = valid(val_loader, model, loss_fn, iftest=False)
        Writer.add_scalar('val_acc',val_acc,t)
        Writer.add_scalar('val_loss',val_loss,t)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'./tmp_out/trust_transformer/sub05/model{i}.pt')
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model.load_state_dict(torch.load(f'./tmp_out/trust_transformer/sub05/model{i}.pt'))
    fpr,tpr,auc,test_acc, test_loss, test_f1_score = valid(test_loader, model, loss_fn, iftest=True)
    logger.info(f"Test Error {i}: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}, f1_score: {test_f1_score:>8f}")
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
np.savetxt("./tmp_out/trust_ccnn/sub05/fprs.csv", mean_fpr, delimiter=',')
np.savetxt("./tmp_out/trust_ccnn/sub05/tprs_std.csv", std_tpr, delimiter=',')
np.savetxt("./tmp_out/trust_ccnn/sub05/tprs.csv", mean_tpr, delimiter=',')
np.savetxt("./tmp_out/trust_ccnn/sub05/aucs.csv", aucs, delimiter=',')
Writer.close()
logger.info(f"Test Error: \n Accuracy: {100*np.mean(test_accs):>0.1f}%, Avg loss: {np.mean(test_losses):>8f}, f1_score:{np.mean(test_f1_scores):>8f}")