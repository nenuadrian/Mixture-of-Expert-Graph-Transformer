from torch_geometric.nn import GCNConv, global_mean_pool
class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channel_1,hidden_channel_2, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(MANUAL_SEED)
        self.conv1 = GCNConv(input_channels, hidden_channel_1)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(hidden_channel_1, hidden_channel_2)
        self.aggregate = global_mean_pool
        self.head = torch.nn.Linear(hidden_channel_2, num_classes)


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.aggregate(x, batch)
        x = self.head(x)
        return x
