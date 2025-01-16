class EmbedEncode(nn.Module):
    def __init__(self, input_size, hidden_size, encoding_size, g_norm):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoding = nn.Linear(encoding_size, hidden_size)
        self.g_norm = g_norm

    def forward(self, x):
        if self.g_norm == True:
          x = self.embedding(x.data_norm) + self.encoding(x.pe)
        else:
          x = self.embedding(x.x) + self.encoding(x.pe)

        return x


class LoadBalancingLoss(nn.Module):
    def __init__(self, criterion, w_load):
        super().__init__()
        self.crit = criterion
        self.w_load = w_load

    def forward(self, output, labels):
        predictions, loads, discard, discard_ = output
        C = self.crit(predictions, labels)
        LBL = self.w_load * loads

        return C + LBL
