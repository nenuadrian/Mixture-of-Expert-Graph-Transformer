class MHGAttend(nn.Module):
    def __init__(self, hidden_size, heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size//heads
        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, Adj):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = q.view(q.shape[0], self.heads, self.head_size).transpose(0,1)  # \
        k = k.view(k.shape[0], self.heads, self.head_size).transpose(0,1)  #  |=>  (heads, nodebatch, head_size)
        v = v.view(v.shape[0], self.heads, self.head_size).transpose(0,1)  # /
        attention_scores = (q @ v.transpose(-2,-1)) / np.sqrt(self.head_size)  #  (heads,nodebatch,head_size)*(heads,head_size,nodebatch)
        attention_scores.masked_fill_(Adj == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        o = attention_scores @ v  #  (heads,nodebatch,nodebatch)*(heads,nodebatch,head_size)
        o = o.transpose(0,1).contiguous().view(o.shape[1], self.hidden_size)  #  (nodebatch,hidden_size)

        return self.wo(o), attention_scores





class MoeveForward(nn.Module):
    def __init__(self, num_xprtz, input_size, xprt_size, k, dropout):
        super().__init__()
        self.expert = nn.Sequential(nn.Linear(input_size, xprt_size),nn.LeakyReLU(),nn.Linear(xprt_size, input_size),nn.Dropout(dropout))
        self.experts = nn.ModuleList([self.expert for _ in range(num_xprtz)])
        self.w_noise = nn.Linear(input_size, num_xprtz, bias=False)
        self.router = nn.Linear(input_size, num_xprtz, bias=False)
        self.softplus = nn.Softplus()
        self.num_xprtz = num_xprtz
        self.k = k
        torch.nn.init.zeros_(self.w_noise.weight)

    def forward(self, input):
        router_choice = self.router(input)     ######################## \
        router_noise = self.softplus(self.w_noise(input))     #########  |=> router net with random noise as in https://arxiv.org/pdf/1701.06538
        H = router_choice + torch.randn(self.num_xprtz).to(device)*router_noise  # /
        weights, experts = torch.topk(H, self.k)  # keep top k choices
        weights = nn.functional.softmax(weights, dim=1, dtype=torch.float)
        out = torch.zeros_like(input).to(device)
        Loads = torch.zeros(self.num_xprtz).to(device)
        counter = torch.zeros(self.num_xprtz).to(device)
        what = torch.zeros(self.num_xprtz,input.shape[0]).to(device)
        for i, xprt in enumerate(self.experts):  # loop on the experts to compute the slices of the batch and load for each one
          # slice
            which_nodes, which_xprt = torch.where(experts == i)
            out[which_nodes] += weights[which_nodes, which_xprt, None] * xprt(input[which_nodes])
            what[i][which_nodes] += 1
          # load
            kthXi = torch.topk(torch.cat([H.transpose(0,1)[:i], H.transpose(0,1)[i+1:]]).transpose(0,1), self.k)[0].transpose(0,1)[self.k - 1]
            router_choice_i = router_choice.transpose(0,1)[i]
            normal = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device))
            P_i = normal.cdf((router_choice_i - kthXi) / router_noise.transpose(0,1)[i])
            load_i = torch.sum(P_i)
            Loads[i] += load_i
            counter[i] += len(which_nodes)

        return out, Loads, counter, what





class Encoder(nn.Module):
    def __init__(self, hidden_size, heads, num_xprtz, xprt_size, k, dropout):
        super().__init__()
        self.attend = MHGAttend(hidden_size, heads)
        self.moveforward = MoeveForward(num_xprtz, hidden_size, xprt_size, k, dropout)
        self.normalize1 = nn.LayerNorm(hidden_size)
        self.normalize2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, Adj):
        input = x
        x, _ = self.attend(x,Adj)
        x = self.normalize1(input + x)
        x = self.drop(x)
        y, load, counter, what = self.moveforward(x)
        y = self.normalize2(x + y)
        load = (torch.std(load) / torch.mean(load))**2
    
        return y, load, counter, what





class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, encoding_size, g_norm, heads, num_xprtz, xprt_size, k, dropout_encoder, layers, output_size):
        super().__init__()
        self.embed = EmbedEncode(input_size, hidden_size, encoding_size, g_norm)
        self.encoders = nn.ModuleList([Encoder(hidden_size, heads, num_xprtz, xprt_size, k, dropout_encoder) for _ in range(layers)])
        self.cut = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.pool = global_mean_pool
        self.predict = nn.Linear(hidden_size // 2, output_size)
        self.num_xprtz = num_xprtz
        self.layers = layers

    def forward(self, x):
        batcher = x.batch
        Adj = to_dense_adj(x.edge_index)
        x = self.embed(x)
        Loads = torch.zeros(self.layers).to(device)
        counters = torch.zeros(self.layers).to(device)
        whats = torch.empty(0,x.shape[0]).to(device)
        for i, encoder in enumerate(self.encoders):
            x, load, counter, what = encoder(x, Adj)
            Loads[i] += load
            counters[i] += torch.std(counter)
            whats = torch.cat([whats,what], dim = 0)
        counters = torch.mean(counters, dim=0)
        Loads = torch.mean(Loads, dim = 0)
        x = self.relu(self.cut(x))
        x = self.pool(x, batcher)
        x = self.predict(x)
    
        return x, Loads, counters, whats
