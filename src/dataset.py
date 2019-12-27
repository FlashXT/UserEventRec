# @Author:FlashXT;
# @Date:2019/12/25 11:10;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
import torch
import random
from src.config import *
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset,DataLoader
from imblearn.over_sampling import SMOTE
#禁止自动换行(设置为Flase不自动换行，True反之)
pd.set_option('expand_frame_repr', False)
#禁用科学计数法
np.set_printoptions(suppress=True)
random.seed(1)
#数据集获取类
class DataSet():
    def __init__(self):
        print("Reading Data and Generating DataSet...")
        # users
        self.uid2index = self.get_users(datasetpath+"usersWithBlock2.csv")
        # self.userblock = self.get_userFeature(datasetpath + "usersWithBlock2.csv",'block')
        self.userCate = self.get_userFeature(datasetpath + "usersCategory.csv",'categoryid')
        # self.userTopic = self.get_userFeature(datasetpath + "usersCategory.csv",'topicid')
        self.userOnAdj = self.get_userAdjList(datasetpath + "usersOnAdjList.txt")
        self.userOffAdj = self.get_userAdjList(datasetpath + "usersOffAdjList.txt")

        # self.topic = self.get_topics(datasetpath + "usersCategory.csv")
        self.category = self.get_category(datasetpath + "usersCategory.csv")
        self.block = self.get_block(datasetpath + "usersWithBlock2.csv")

        #events
        self.eve2index = self.get_events(datasetpath+"eventsWithBlock2.csv")
        # self.eventsBlock = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'block')
        # self.eventsMonth = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'month')
        # self.eventsDay = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'day')
        # self.eventTS = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'timeslot')
        # self.eventWeekDay = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'weekday')
        # self.eventWeekEnd = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'weekend')
        # self.eventDuration = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'duration')
        # self.eventAttendance = self.get_eventFeature(datasetpath+"eventsWithBlock2.csv",'attendance')
        #
        #groups
        self.grp2index = self.get_groups(datasetpath+"groupsWithBlock2.csv")
        # self.groupBlock = self.get_groupFeature(datasetpath+"groupsWithBlock2.csv",'block')
        # self.groupCate = self.get_groupFeature(datasetpath + "groupsWithBlock2.csv", 'categoryid')
        # self.groupSize = self.get_groupFeature(datasetpath + "groupsWithBlock2.csv", 'members')
        # self.groupMembers = self.get_groupUsers(datasetpath + "groupsMember.txt")

        self.trainset,self.testset = self.getTrainTestSet(datasetpath + "trainset.csv",datasetpath + "testset.csv",
                                         datasetpath+"usersWithBlock2.csv",datasetpath+"eventsWithBlock2.csv",
                                         datasetpath+"groupsWithBlock2.csv")

        # self.trainloader = self.trainDataLoader(id)


    def get_users(self,path):
        user2index={}
        userdf = pd.read_csv(path,encoding='utf-8')[['userid']]
        # print(userdf.info())
        for uid in userdf['userid']:
            if uid not in user2index.keys():
                user2index[uid]=len(user2index)
        return user2index
    def get_userFeature(self,path,fea):
        uid2index = pd.DataFrame.from_dict(self.uid2index, orient='index')
        uid2index = uid2index.reset_index().rename(columns={'index': 'userid',0:'index'})

        userdf = pd.read_csv(path,encoding='utf-8')[['userid',fea]]
        res = uid2index.merge(userdf,how='left',on='userid')
        # print(res.head())
        temp={}
        for item in res.values.tolist():
            if item[1] not in temp.keys():
                temp[item[1]]=set()
            temp[item[1]].add(item[2])

        return temp
    def get_userAdjList(self,path):
        userAdjlist = {}
        with open(path,'r') as file:
            for item in file:
                item = item.strip("\n").split(" ")
                u = self.uid2index[int(item[0])]
                userAdjlist[u]=[]
                for neig in item[1].split(","):
                    userAdjlist[u].append(self.uid2index[int(neig)])
        file.close()
        return userAdjlist
    def get_topics(self,path):
        ut = {}
        userT = pd.read_csv(path,encoding='utf-8')['topicid'].drop_duplicates()
        for item in userT:
            ut[item]=len(ut)
        return ut
    def get_category(self, path):
        uc = set()
        userCate = pd.read_csv(path, encoding='utf-8')['categoryid'].drop_duplicates()
        for item in userCate:
            uc.add(item)
        return list(uc)
    def get_block(self, path):
        uc = set()
        userCate = pd.read_csv(path, encoding='utf-8')['block'].drop_duplicates()
        for item in userCate:
            uc.add(item)
        return list(uc)

    def get_events(self, path):

        event2index = {}
        evedf = pd.read_csv(path, encoding='utf-8')[['eventid']]
        for eid in evedf['eventid']:
            if eid not in event2index.keys():
                event2index[eid] = len(event2index)
        return event2index
    def get_eventFeature(self,path,fea):

        eve2index = pd.DataFrame.from_dict(self.eve2index, orient='index')
        eve2index =eve2index.reset_index().rename(columns={'index': 'eventid', 0: 'index'})

        evedf = pd.read_csv(path, encoding='utf-8')[['eventid', fea]]
        res = eve2index.merge(evedf, how='left', on='eventid')
        # print(res.head())
        temp = {}
        for item in res.values.tolist():
            temp[item[1]] = item[2]

        return temp

    def get_groups(self,path):
        grp2index = {}
        grpdf = pd.read_csv(path, encoding='utf-8')[['groupid']]
        # print(userdf.info())
        for gid in grpdf['groupid']:
            if gid not in grp2index.keys():
                grp2index[gid] = len(grp2index)
        return grp2index
    def get_groupFeature(self,path,fea):

        grp2index = pd.DataFrame.from_dict(self.grp2index, orient='index')
        grp2index =grp2index.reset_index().rename(columns={'index': 'groupid', 0: 'index'})

        grpdf = pd.read_csv(path, encoding='utf-8')[['groupid', fea]]
        res = grp2index.merge(grpdf, how='left', on='groupid')
        # print(res.head())
        temp = {}
        for item in res.values.tolist():
            temp[item[1]] = item[2]
        return temp
    def get_groupUsers(self, path):
        with open(path, 'r') as file:
            gusers = {}
            for item in file:
                gu = item.strip("\n").split(" ")

                us = gu[1].split(",")
                gid = self.grp2index[int(gu[0])]
                gusers[gid] = []
                for u in us:
                    gusers[gid].append(self.uid2index[int(u)])
        file.close()

        return gusers

    def getTrainTestSet(self, path1,path2,path3,path4,path5):
        train = pd.read_csv(path1,encoding='utf-8')[['memberid','groupid','eventid','year_month','eventHost','guests','rsvp']]
        test = pd.read_csv(path2,encoding='utf-8')[['memberid','groupid','eventid','year_month','eventHost','guests','rsvp']]
        train = pd.concat([train,test],axis=0)
        users = pd.read_csv(path3,encoding='utf-8')[['userid','block']]
        users.rename(columns={'userid':'memberid','block':'memblock'},inplace=True)
        train = train.merge(users, how='left', on='memberid')
        # print(train.head())
        # userscate = pd.read_csv(path3, encoding='utf-8')[['userid', 'categoryid']]
        # userscate.rename(columns={'userid': 'memberid', 'categoryid': 'membcateid'}, inplace=True)
        # train = train.merge(userscate, how='left', on='memberid')
        # print(train.head())
        events = pd.read_csv(path4,encoding='utf-8')[['eventid','month','day','weekday','weekend','hour','timeslot','duration','attendance']]
        train = train.merge(events,how='left',on='eventid')
        # print(train.head())
        groups = pd.read_csv(path5, encoding='utf-8')[['groupid', 'block', 'members','categoryid']]
        groups.rename(columns={'members':'grpsize','categoryid': 'grpcate', 'block': 'grpblock'}, inplace=True)
        train = train.merge(groups, how='left', on='groupid')

        train = train[['memberid','memblock', 'groupid', 'grpblock', 'grpsize', 'grpcate','eventid','year_month',
                       'month', 'day','hour','timeslot', 'weekday', 'weekend', 'duration', 'attendance',
                       'eventHost', 'guests','rsvp']]
        # print(train.head())
        # print(train.columns)
        uid2index = pd.DataFrame.from_dict(self.uid2index, orient='index')
        uid2index = uid2index.reset_index().rename(columns={'index': 'memberid', 0: 'userindex'})
        temp1 = train.merge(uid2index, how='left', on='memberid')

        eve2index = pd.DataFrame.from_dict(self.eve2index, orient='index')
        eve2index = eve2index.reset_index().rename(columns={'index': 'eventid', 0: 'eveindex'})
        temp2 = temp1.merge(eve2index, how='left', on='eventid')

        grp2index = pd.DataFrame.from_dict(self.grp2index, orient='index')
        grp2index = grp2index.reset_index().rename(columns={'index': 'groupid', 0: 'grpindex'})
        res = temp2.merge(grp2index, how='left', on='groupid')
        # print(res.head())
        # print(res.columns)
        train = res[['userindex', 'memblock', 'grpindex', 'grpblock', 'grpsize', 'grpcate',
                    'eveindex', 'year_month', 'month', 'day', 'hour', 'timeslot', 'weekday',
                    'weekend', 'duration', 'attendance', 'eventHost', 'guests', 'rsvp',
                     ]]
        train.rename(columns={'userindex':'memberid','eveindex':'eventid','grpindex':'groupid'},inplace=True)
        # print(train.head())

        train['memberid'] = train['memberid'].astype('int')
        train['groupid'] = train['groupid'].astype('int')
        train['eventid'] = train['eventid'].astype('int')
        # print(train.head())
        # print(train.info())
        def eventHost(x):
            if x is False:
                return 0
            else:
                return 1
        train['eventHost'] = train['eventHost'].apply(eventHost)
        # print(train.head())
        trainset = {}

        history0 = train[train['year_month'].isin(['2018-01','2018-02','2018-03','2018-04'])]
        temp = train[(train['year_month'].isin(['2018-05','2018-06']))]
        train0 = temp[temp['memberid'].isin(set(history0['memberid']))]

        history1 = train[(train['year_month'].isin(['2018-03', '2018-04', '2018-05','2018-06']))]
        temp1 = train[(train['year_month'].isin(['2018-07','2018-08']))]
        train1 = temp1[temp1['memberid'].isin(set(history1['memberid']))]

        history2 = train[(train['year_month'].isin(['2018-05', '2018-06', '2018-07','2018-08']))]
        temp2 = train[(train['year_month'].isin(['2018-09','2018-10']))]
        train2 = temp2[temp2['memberid'].isin(set(history2['memberid']))]

        history3 = train[(train['year_month'].isin(['2018-07', '2018-08', '2018-09', '2018-10']))]
        temp3 = train[(train['year_month'].isin(['2018-11', '2018-12']))]
        test = temp3[temp3['memberid'].isin(set(history3['memberid']))]


        trainset["set0"] = {"history":history0,'train':train0}
        trainset["set1"] = {"history":history1,'train':train1}
        trainset["set2"] = {"history":history2,'train':train2}
        testset = {"history": history3, 'test': test}
        return trainset,testset



    # def userHistory(self,history):
    #     print(history.head())
    #     data = history[['memberid','groupid','eventid','rsvp']]
    #     print(data.head())
    #     userhis={}
    #     for item in data.values.tolist():
    #         uid = item[0]
    #         gid = item[1]
    #         eid = item[2]
    #         label =item[3]
    #         if uid not in userhis.keys():
    #             userhis[uid] = {'pos':[],'neg':[]}
    #         if label == 1:
    #             userhis[uid]['pos'].append([gid,eid])
    #         else:
    #             userhis[uid]['neg'].append([gid, eid])
    #     return userhis
    def behavior(self,id):
        behavior = self.trainset['set' + str(id)]['history'][['memberid', 'memblock', 'groupid', 'grpblock',
                                                             'grpsize', 'grpcate', 'eventid', 'month', 'day', 'hour',
                                                             'timeslot', 'weekday','weekend', 'duration', 'attendance',
                                                             'eventHost', 'guests','rsvp']].reset_index(drop=True)
        # print(history.info())
        return behavior
    def trainDataLoader(self,id):
        history = self.behavior(id)
        train = self.trainset['set' + str(id)]['train'][['memberid', 'memblock', 'groupid', 'grpblock',
                                                         'grpsize', 'grpcate', 'eventid', 'month', 'day', 'hour',
                                                         'timeslot', 'weekday','weekend', 'duration', 'attendance',
                                                         'eventHost', 'guests','rsvp']].reset_index(drop=True)

        # print(train.head())
        data = torch.tensor(np.array(train.values.tolist()),dtype=torch.int)
        # print(data.shape)
        train_data = TensorDataset(data[:,0:17].reshape(-1,17),
                                   data[:,17].reshape(-1,1))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return train_loader,history
    def testDataLoader(self):
        behavior = self.testset['history'][['memberid', 'memblock', 'groupid', 'grpblock',
                                            'grpsize', 'grpcate', 'eventid', 'month', 'day', 'hour',
                                            'timeslot', 'weekday', 'weekend', 'duration','attendance',
                                            'eventHost', 'guests', 'rsvp']].reset_index(drop=True)

        test = self.testset['test'][['memberid', 'memblock', 'groupid', 'grpblock',
                                    'grpsize', 'grpcate', 'eventid', 'month', 'day', 'hour',
                                    'timeslot', 'weekday','weekend', 'duration', 'attendance',
                                    'eventHost', 'guests','rsvp']].reset_index(drop=True)

        data = torch.tensor(np.array(test.values.tolist()),dtype=torch.int)
        testdata = TensorDataset(data[:,0:17].reshape(-1,17),
                                   data[:,17].reshape(-1,1))
        # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return testdata,behavior

if __name__=="__main__":
    data = DataSet()
    print('AAAAA')