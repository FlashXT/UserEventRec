# @Author:FlashXT;
# @Date:2019/12/26 19:41;
# @Version 1.0
# CopyRight © 2018-2020,FlashXT & turboMan . All Right Reserved.
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    def __init__(self,entry_num,emdedding_dim):
        super(EmbeddingLayer,self).__init__()
        self.embedd = nn.Embedding(entry_num,emdedding_dim)

    def forward(self,x):
        return self.embedd(torch.LongTensor(x))


class EmbeddingBagLayer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(EmbeddingBagLayer,self).__init__()

        self.embedd = nn.EmbeddingBag(in_dim,out_dim,mode='mean')

    def forward(self, x):
        # print(x)
        out = self.embedd(torch.LongTensor(x),torch.LongTensor([0]))
        return out


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionLayer, self).__init__()

        self.local_att = LocalActivationUnit(hidden_size=[64, 16], bias=True, embedding_dim=embedding_dim,
                                             batch_norm=False)

    def forward(self, query_ad, user_behavior, user_behavior_length):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # user behavior length: size -> batch_size * 1
        # output              : size -> batch_size * 1 * embedding_size

        attention_score = self.local_att(query_ad, user_behavior)
        attention_score = torch.transpose(attention_score, 1, 2)  # B * 1 * T
        # print(attention_score.size())

        # define mask by length
        user_behavior_length = user_behavior_length.type(torch.LongTensor)
        mask = torch.arange(user_behavior.size(1))[None, :] < user_behavior_length[:, None]

        # mask
        output = torch.mul(attention_score, mask.type(torch.cuda.FloatTensor))  # batch_size *

        # multiply weight
        output = torch.matmul(output, user_behavior)

        return output


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_size=[80, 40], bias=True, embedding_dim=4, batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='relu')

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=bias,
                                       batch_norm=batch_norm,
                                       activation='relu')
        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)


    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([queries, user_behavior, queries - user_behavior, queries * user_behavior], dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output


class FullyConnectedLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias, batch_norm=True, dropout_rate=0.5, activation='relu',
                 sigmoid=False):
        super(FullyConnectedLayer, self).__init__()

        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias))

        for i, h in enumerate(hidden_size[:-1]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))

            else:
                layers.append(nn.PReLU())

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)
