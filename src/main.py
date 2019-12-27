# @Author:FlashXT;
# @Date:2019/12/26 17:37;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
from src.util import *
from src.dataset import *
from src.model.DIN import DeepInterestNetWork
from sklearn.metrics import f1_score,roc_auc_score,precision_score,recall_score
def main():
    data = DataSet()

    # model = DeepInterestNetWork(len(data.uid2index),len(data.grp2index),len(data.eve2index),
    #                             data.userOffAdj,data.userCate)
    # for i in range(0,4):
    #     trainloader,userbehavior = data.trainDataLoader(i)
    #     model.userbha = userbehavior
    #     for epoch_id in range(epoch):
    #         # model.train()
    #         train(model, trainloader, epoch_id, epoch)

            # model.eval()
            # testset = data.testset
            # examples = torch.LongTensor(testset[:, 0:17]).reshape(-1, 17)
            # labels = torch.LongTensor(testset[:, 17]).reshape(-1, 1)
            # outputs = model(examples).detach().numpy()
            # pred = np.rint(outputs)
            # f1 = f1_score(labels, pred)
            # # # print(pred)
            # # # print(labels)
            # pre = precision_score(labels, pred)
            # rec = recall_score(labels, pred)
            # auc = roc_auc_score(labels, outputs)  # 验证集上的auc值
            # print("Epoch:%d,Precision is :%.4f,Recall is :%.4f, F1_Score is : %.4f,AUC is : %.4f..." % (
            # epoch_id + 1, pre, rec, f1, auc))
            # writer.add_scalar('Test/F1', f1, epoch_id + 1)
            # writer.add_scalar('Test/AUC', auc, epoch_id + 1)
            # writer.add_scalar('Test/Pre', pre, epoch_id + 1)
            # writer.add_scalar('Test/Rec', rec, epoch_id + 1)
            # hr,ndcg = evaluate(model,dataset.groupTestset,dataset.groupUsers,topK)
            # writer.add_scalar('Test/HR@5', hr,epoch_id)
            # writer.add_scalar('Test/NDCG@5', ndcg, epoch_id)
            #
            # print(str('Epoch %d: HR@'+str(topK)+': %.3f,NDCG@'+str(topK)+':%.3f...') % (epoch_id + 1, hr,ndcg))


if __name__=="__main__":
    main()