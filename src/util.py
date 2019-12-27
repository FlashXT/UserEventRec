# @Author:FlashXT;
# @Date:2019/12/25 11:10;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
#辅助方法类
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import *
def train(model,trainloader,epoch_id,epoch):
    # user trainning
    learning_rates = learn_rate
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= epoch/2:
        lr = learning_rates[1]

    if epoch_id  > epoch*3/4:
        lr =learning_rates[2]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum = momentum)
    # optimizer = optim.Adam(model.parameters(), lr = lr,betas=(0.9,0.999),eps=1e-08,weight_decay=0.01)

    model.train()

    training_loss = 0.0
    for i, data in enumerate(trainloader):

        optimizer.zero_grad()

        data,labels = data
        crition = nn.BCELoss()  # 输入前需要sigmoid()
        outputs = model(data.long())
        loss = crition(outputs,labels.float())
        loss.backward()
        training_loss += loss.item()
        optimizer.step()
        if i%100 == 99:
            print('Epoch %d,Batch %d, training loss: %.3f...' % (epoch_id + 1, i + 1, training_loss/100))
            training_loss = 0
        # 每10个batch画个点用于loss曲线,ACC曲线
        if i % 100 == 0:
            niter = epoch_id * len(trainloader) + i
            # if type =='Group':
            #     writer.add_scalar('GroupTrain/Loss', loss.item(), niter)
            # else:
            writer.add_scalar('UserTrain/Loss', loss.item(), niter)

    # print("Finish Training.")
def test():
    raise  NotImplementedError

def evaluate():
    raise NotImplementedError

def HitRatio():
    raise NotImplementedError

def NDCG():
    raise NotImplementedError
