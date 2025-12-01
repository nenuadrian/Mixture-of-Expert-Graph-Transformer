from sparticles import EventsDataset
from sparticles.transforms import MakeHomogeneous
from sparticles import plot_event_2d
import torch 
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import numpy as np

class CustomMakeHomogeneous(MakeHomogeneous):
    def __init(self, *args, **kwargs):
        super().__init(*args, **kwargs)
    def forward(self, data):
        pass

#standardization of only continuous features
class CustomEventsDataset1(EventsDataset):
    def __init__(self, k=4, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate mean and standard deviation across all graphs in the dataset
        self.mean_values, self.std_values = self.calculate_mean_std()

        # Set the parameter k for the Laplacian Eigenvector PE transformation
        self.k = k

    def calculate_mean_std(self):
        # Initialize list to store all features of all graphs
        all_features = []

        # Iterate through all graphs in the dataset
        for idx in range(len(self)):
            data = super().__getitem__(idx)
            all_features.append(data.x.numpy())

        # Concatenate features from all graphs
        all_features = np.concatenate(all_features, axis=0)

        # Calculate mean and standard deviation for each feature
        mean_values = np.mean(all_features, axis=0)
        std_values = np.std(all_features, axis=0)

        return mean_values, std_values

    def custom_transform(self, data):
        # Apply the AddLaplacianEigenvectorPE transformation to the data with the chosen k
        lap_pos = AddLaplacianEigenvectorPE(k=self.k)
        data = lap_pos(data)

        # Now 'laplacian_eigenvector_pe' should be available in the data object
        pe = data.laplacian_eigenvector_pe

        # Standardize only even-indexed features in data.x
        standardized_x = data.x.clone()  # Clone to avoid modifying original data.x directly
        even_indices = torch.arange(0, data.x.size(1), 2)  # Select even feature indices
        standardized_x[:, even_indices] = (data.x[:, even_indices] - self.mean_values[even_indices]) / (self.std_values[even_indices] + 1e-6)

        return pe, standardized_x

    def __getitem__(self, idx):
        # Retrieve the data item using the superclass method
        data = super().__getitem__(idx)

        # Apply the custom transformation to obtain the Laplacian eigenvector features and normalized data
        pe, data_norm = self.custom_transform(data)

        # Add the 'pe' attribute to the data
        data.pe = pe

        # Add the 'data_norm' attribute to the data
        data.data_norm = data_norm

        return data
    
    
class CustomEventsDataset2(EventsDataset):
    def __init__(self, *args, k=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.mean_values, self.std_values = self.calculate_mean_std()

    def calculate_mean_std(self):
        # Initialize lists to store all features of all graphs
        all_features = []

        # Iterate through all graphs in the dataset
        for idx in range(len(self)):
            data = super().__getitem__(idx)
            all_features.append(data.x.numpy())

        # Concatenate features from all graphs
        all_features = np.concatenate(all_features, axis=0)

        # Calculate mean and standard deviation
        mean_values = np.mean(all_features, axis=0)
        std_values = np.std(all_features, axis=0)

        return mean_values, std_values

    def custom_transform(self, data):
        # Apply the AddLaplacianEigenvectorPE transformation to the data with the chosen k
        lap_pos = AddLaplacianEigenvectorPE(k=self.k)
        data = lap_pos(data)

        # Now 'laplacian_eigenvector_pe' should be available in the data object
        pe = data.laplacian_eigenvector_pe

        # Standardize the features using mean and standard deviation
        standardized_x = (data.x - self.mean_values) / (self.std_values + 1e-6)

        return pe, standardized_x

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # Apply the custom transformation to obtain the Laplacian eigenvector features and normalized data
        pe, data_norm = self.custom_transform(data)

        # Add the 'pe' attribute to the data
        data.pe = pe

        # Add the 'data_norm' attribute to the data
        data.data_norm = data_norm

        return data
