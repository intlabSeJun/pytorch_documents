import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class): #vocab, 100, 10, [3, 4, 5], 2
        super(TextCNN, self).__init__()

        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors) # vacab에 있는 w2v weight load.

        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(len(kernel_wins) * dim_channel, num_class)

    def forward(self, x):  # (N,Vocab_num) Vocab_num: 각 입력(문장)의 단어(형태소+명사)의 번호가 들어있음.
                                                      #해당 번호에 따라 nn.Embedding()에서 해당 단어의 weight를 가져옴.
        emb_x = self.embed(x)  # Embedding이 weight를 가져오는 형식?? 각 단어의 embedding vector를 가져오는구만? (N,Vocab_num,100)@
        emb_x = emb_x.unsqueeze(1)  # (N,1,Vocab_num,100)

        con_x = [self.relu(conv(emb_x)) for conv in self.convs]  #(3,4,5)줄씩 conv 수행. [(N,10,dim,1)]x3

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]  #(N,10,1)

        fc_x = torch.cat(pool_x, dim=1)  #(N,30,1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)

        logit = self.fc(fc_x)

        return logit