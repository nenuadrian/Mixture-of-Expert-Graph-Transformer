class MultiHeadGraphAttention(nn.Module):
    """Multi-Head Graph Attention Module"""

    def __init__(self, hidden_size=12, num_heads=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scaling = self.head_size ** -0.5

    def forward(self, A, h):
        N = h.size(0)  # Number of nodes

        # Compute query keys and value as projection of the input
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)
        k = k.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)
        v = v.view(N, self.num_heads, self.head_size).transpose(0, 1)  # (num_heads, N, head_size)

        scores = torch.matmul(q, k.transpose(1, 2))*self.scaling #(num_heads, N, N) attention score between a pair of nodes for each attention head
        scores.masked_fill_(A == 0, -1e9)
        #scores = entmax15(scores,dim=2)
        scores = F.softmax(scores, dim=2)

        out = torch.matmul(scores, v) #(num_heads, N, head_size)
        out = self.out_proj(out.transpose(0, 1).contiguous().view(N, self.hidden_size))  # (N, hidden_size)

        return out, scores


class GTLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(self, hidden_size=12, num_heads=3, dropout=0.3):
        super().__init__()
        self.MHGAtt = MultiHeadGraphAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.layernorm1 = nn.LayerNorm(hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.FFN1 = nn.Linear(hidden_size, hidden_size * 2)
        self.FFN2 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = dropout

    def forward(self, A, h):
        h1 = h
        h, _ = self.MHGAtt(A, h)  # Compute multi-head graph attention
        h = self.layernorm1(h + h1)  # Add node feature and compute layer norm
        h = F.dropout(h, self.dropout,training=self.training)

        # Compute feed forward
        h2 = h
        h = self.FFN1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout,training=self.training)
        h = self.FFN2(h)
        h = h2 + h  # Residual connection

        return self.layernorm2(h)  # Layer norm



class GraphTransformerModel(nn.Module):
    #pos_enc_size = 0 referes to no pos encoding
    def __init__(self, out_size,input_size = 12, hidden_size=12 ,pos_enc_size=2,num_layers=4, num_heads=3,dropout=0.3, normalization=True):
        super(GraphTransformerModel, self).__init__()
        self.normalization = normalization
        self.pos_enc_size = pos_enc_size
        self.pos_linear = nn.Linear(pos_enc_size, hidden_size)
        self.embedding = nn.Linear(input_size,hidden_size)
        self.layers = nn.ModuleList([GTLayer(hidden_size, num_heads,dropout) for _ in range(num_layers)])

        # Linear layers for prediction
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.aggregate = global_mean_pool
        self.linear3 = nn.Linear(hidden_size // 2, out_size)

    def forward(self, data, batch=None):
        A = to_dense_adj(data.edge_index)[0]  # Convert edge index to adjacency matrix
        if self.pos_enc_size !=0 and self.normalization == True:
           h = self.embedding(data.data_norm) + self.pos_linear(data.pe)                  # Data features Shape: (Batch Size, Num Nodes, Feature Dimension)
        elif self.pos_enc_size !=0 and self.normalization == False:
           h = self.embedding(data.x) + self.pos_linear(data.pe)
        elif self.pos_enc_size ==0 and self.normalization == True:
           h = self.embedding(data.data_norm)
        else:
           h = self.embedding(data.x)

        #Compute the GT layer
        for layer in self.layers:
            h = layer(A, h)

        # Linear layers for prediction
        h = self.linear1(h)
        h = self.relu1(h)
        h = self.linear2(h)
        h = self.relu2(h)
        # Control where you have to aggregate
        h = self.aggregate(h, data.batch)
        h = self.linear3(h)

        return h 
