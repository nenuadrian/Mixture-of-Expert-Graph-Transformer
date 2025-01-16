#This class define the attention mechanism for graphs 

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
        k = k.view(k.shape[0], self.heads, self.head_size).transpose(0,1)  #  |=>  (heads, nodebatch, head_size) compute the linear projection
        v = v.view(v.shape[0], self.heads, self.head_size).transpose(0,1)  # /
        attention_scores = (q @ v.transpose(-2,-1)) / np.sqrt(self.head_size)  #  (heads,nodebatch,head_size)*(heads,head_size,nodebatch) 
        attention_scores.masked_fill_(Adj == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        o = attention_scores @ v  #  (heads,nodebatch,nodebatch)*(heads,nodebatch,head_size)
        o = o.transpose(0,1).contiguous().view(o.shape[1], self.hidden_size)  #  (nodebatch,hidden_size)

        return self.wo(o), attention_scores




#this class define the forward pass with MoE 

class MoeveForward(nn.Module):
    def __init__(self, num_xprtz, input_size, xprt_size, k, dropout):
        super().__init__()
        
        # Define a single expert: A simple feedforward neural network (FFN)
        self.expert = nn.Sequential(
            nn.Linear(input_size, xprt_size),  # Linear layer: input -> hidden
            nn.LeakyReLU(),                   # Activation function
            nn.Linear(xprt_size, input_size), # Linear layer: hidden -> output
            nn.Dropout(dropout)               # Dropout for regularization
        )
        
        # Combine multiple experts using ModuleList
        self.experts = nn.ModuleList([self.expert for _ in range(num_xprtz)])

        # Define a noise network to introduce randomness in the routing process
        self.w_noise = nn.Linear(input_size, num_xprtz, bias=False)

        # Define the router network that decides which experts to activate
        self.router = nn.Linear(input_size, num_xprtz, bias=False)

        # Activation function to ensure noise values are non-negative
        self.softplus = nn.Softplus()

        # Save the number of experts and the top-k selection parameter
        self.num_xprtz = num_xprtz  # Number of experts
        self.k = k  # Number of top-k experts to select

        # Initialize the weights of the noise network to zero
        torch.nn.init.zeros_(self.w_noise.weight)

    def forward(self, input):
        # Compute the routing logits from the router network
        router_choice = self.router(input)

        # Compute the noise values using the noise network and Softplus activation
        router_noise = self.softplus(self.w_noise(input))

        # Add random noise to the router's logits to introduce stochasticity
        H = router_choice + torch.randn(self.num_xprtz).to(device) * router_noise

        # Select the top-k experts based on the noisy routing logits
        weights, experts = torch.topk(H, self.k)  # `weights`: top-k values, `experts`: top-k indices

        # Apply a softmax to the selected weights to normalize them
        weights = nn.functional.softmax(weights, dim=1, dtype=torch.float)

        # Initialize outputs and tracking tensors
        out = torch.zeros_like(input).to(device)  # Output tensor (same shape as input)
        Loads = torch.zeros(self.num_xprtz).to(device)  # Tracks load for each expert
        counter = torch.zeros(self.num_xprtz).to(device)  # Tracks number of samples processed by each expert
        what = torch.zeros(self.num_xprtz, input.shape[0]).to(device)  # Tracks which samples go to which expert

        # Loop through each expert to compute outputs and loads
        for i, xprt in enumerate(self.experts):
            # Identify samples assigned to the current expert
            which_nodes, which_xprt = torch.where(experts == i)

            # Compute the output for the assigned samples
            out[which_nodes] += weights[which_nodes, which_xprt, None] * xprt(input[which_nodes])

            # Track which samples were assigned to this expert
            what[i][which_nodes] += 1

            # Calculate the load for the current expert
            # Find the k-th largest routing logit excluding the current expert
            kthXi = torch.topk(torch.cat([
                H.transpose(0, 1)[:i],         # Logits before the current expert
                H.transpose(0, 1)[i + 1:]      # Logits after the current expert
            ]).transpose(0, 1), self.k)[0].transpose(0, 1)[self.k - 1]  # Select k-th largest logit

            # Retrieve the routing logit for the current expert
            router_choice_i = router_choice.transpose(0, 1)[i]

            # Define a normal distribution (mean=0, std=1) for computing probabilities
            normal = torch.distributions.normal.Normal(
                torch.tensor([0.0]).to(device), torch.tensor([1.0]).to(device)
            )

            # Compute the probability of the current expert being selected
            P_i = normal.cdf((router_choice_i - kthXi) / router_noise.transpose(0, 1)[i])

            # Compute the load for the current expert
            load_i = torch.sum(P_i)
            Loads[i] += load_i

            # Track the number of samples processed by this expert
            counter[i] += len(which_nodes)

        # Return the output, loads, sample counts, and assignment tracking
        return out, Loads, counter, what



#this class define the encoder network 

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




#This is the full architecture of Transformer 
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
