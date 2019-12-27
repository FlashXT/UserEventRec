# @Author:FlashXT;
# @Date:2019/12/27 22:07;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.

# his1 = data.trainset['set0']['history']
# his2 = data.trainset['set1']['history']
# his3 = data.trainset['set2']['history']
# u1 = his1.groupby('memberid',as_index=False).size().reset_index()
# u1.rename(columns={0:'num'},inplace=True)
# print("mer history length；",int(np.sum(u1['num'])/u1.shape[0]))
# u2 = his2.groupby('memberid', as_index=False).size().reset_index()
# u2.rename(columns={0: 'num'}, inplace=True)
# print("mer history length；",int(np.sum(u2['num']) / u2.shape[0]))
# u3 = his3.groupby('memberid', as_index=False).size().reset_index()
# u3.rename(columns={0: 'num'}, inplace=True)
# print("mer history length；", int(np.sum(u3['num']) / u3.shape[0]))