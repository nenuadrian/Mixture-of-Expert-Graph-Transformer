#Define the MLP architecture
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, dropout=0.2):
        super(MLPClassifier, self).__init__()

        # Input layer
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.dropout = nn.Dropout(dropout)  # Dropout to avoid overfitting

        # Hidden layer 1
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)

        # Output layer
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)

        # Activation function
        self.relu = nn.ReLU()

        # Softmax for the final output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)
