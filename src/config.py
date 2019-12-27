# @Author:FlashXT;
# @Date:2019/12/25 11:10;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.

#参数配置类
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('../log')
#dataset path
datasetpath = "../UserEventDataSet/"

# training parameters
batch_size =4
embedding_dim = 8
drop_ratio = 0.5
epoch = 500
topK=5
learn_rate = [0.005, 0.0001, 0.00005]
momentum=0.9
history = 5

#model save path
modelpath = "./model.mdl"


