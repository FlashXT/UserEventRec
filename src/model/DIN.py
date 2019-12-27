# @Author:FlashXT;
# @Date:2019/12/25 11:10;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
import torch
import numpy as np
import torch.nn as nn
from src.model.layers import *
from src.config import *
class DeepInterestNetWork(nn.Module):
    def __init__(self,usernum,groupnum,eventnum,userAdj,userCate,categorynum=40,blocknum=100):
        super(DeepInterestNetWork,self).__init__()
        #public Features
        self.category = EmbeddingBagLayer(categorynum,embedding_dim)
        self.block = EmbeddingLayer(blocknum,embedding_dim)
        #event Features
        self.events = EmbeddingLayer(eventnum, embedding_dim)
        self.month = EmbeddingLayer(13,embedding_dim)
        self.day = EmbeddingLayer(32,embedding_dim)
        self.weekday = EmbeddingLayer(7,embedding_dim)
        self.hour = EmbeddingLayer(25, embedding_dim)
        self.timeslot = EmbeddingLayer(9,embedding_dim)

        #user Features
        self.users = EmbeddingLayer(usernum, embedding_dim)
        # self.userBlock = userBlock
        self.userCate = userCate
        self.userAdj = userAdj

        #group Features
        self.groups = EmbeddingLayer(groupnum,embedding_dim)
        # self.grpBlock = groupBlock
        # self.grpCate = groupCate

        self.userbha = None
        #NN layer
        self.att = AttentionLayer(embedding_dim)

    def userhistory(self,uh):
        self.userbehavior=uh
    def forward(self,x):
        # user = self.get_users(x[:,0:2])
        # grp = self.get_groups(x[:,2:6])
        # eve = self.get_events(x[:,6:])
        ub = self.userBehavier(x[:,0])




    def get_users(self,user):

        usercate = torch.Tensor()

        for i in range(user.shape[0]):
            cateindex = torch.tensor(list(self.userCate[user[i,0].item()]),dtype=torch.long)

            usercate = torch.cat([usercate,self.category(cateindex)])

        userprofile = torch.cat([self.users(user[:,0]),self.block(user[:,1]),usercate],dim=1)
        return userprofile

    def get_groups(self,grp):
        # group
        grpcate = torch.Tensor()
        gc = grp[:, 3]
        for i in range(grp.shape[0]):
            grpcate = torch.cat([grpcate, self.category([gc[i].item()])])

        grpprofile = torch.cat([self.groups(grp[:, 0]).float(),self.block(grp[:, 1]).float(),
                                grpcate, grp[:, 2].float().reshape(-1, 1)], dim=1)
        return grpprofile

    def get_events(self,eve):

        #event
        a = self.events(eve[:,0])
        b = self.month(eve[:,1])
        c = self.day(eve[:,2])
        d = self.hour(eve[:,3])
        e = self.timeslot(eve[:,4])
        f = self.weekday(eve[:,5])
        event = torch.cat([a,b,c,d,e,f,eve[:,6:].float().reshape(-1, 5)],dim=1)
        return eve

    def userBehavier(self,user):
        res = torch.Tensor()
        for u in range(user.shape[0]):
            his = self.userbha[self.userbha['memberid']== user[u].item()]

            data = torch.tensor(np.array(his.values.tolist()), dtype=torch.long)[:,2:17]
            print(data)