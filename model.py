import torch
import torch.nn as nn
from data_load import hp, device
from pytorch_pretrained_bert import BertModel

class Net(nn.Module):
    def __init__(self, training=False):
        super().__init__()
        self.training = training
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.bert.to(device)
        self.bert.eval()
        self.bert = nn.DataParallel(self.bert)
        self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
        self.fc = nn.Linear(768, len(hp.VOCAB))
        self.fc = nn.DataParallel(self.fc)

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64

        Returns
        enc: (N, T, VOCAB)
        '''
        x = x.to(device)
        y = y.to(device)

        if self.train:
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        enc, _ = self.rnn(enc)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

