
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
# Define model ( in your class_model_gnn.py)
class StudentModel(nn.Module):
    def __init__(self,n_features=50,n_classes=121):
        super().__init__()
        input_size = n_features
        hidden_size = 350
        output_size = n_classes
        num_heads = 4
        n_layers = 3
        dropout = 0
        self.layers = nn.ModuleList()
        self.nonlinearity = F.leaky_relu

        self.layers.append(
            graphnn.GATConv(input_size, hidden_size, heads=num_heads, dropout=dropout)
        )
        for _ in range(1, n_layers - 1):
            self.layers.append(
                graphnn.GATConv(
                    hidden_size * num_heads,
                    hidden_size,
                    heads=num_heads,
                    dropout=dropout,
                )
            )

        self.layers.append(nn.Linear(hidden_size * num_heads, output_size))

    def forward(self, x, edge_index):
        for _, layer in enumerate(self.layers[:-1]):
            x = self.nonlinearity(layer(x, edge_index))
        x = self.layers[-1](x)
        return x
if __name__ == "__main__":
    n_features = 1000
    n_classes = 2
    # Initialize model
    model = StudentModel()


    ### This is the part we will run in the inference to grade your model
    ## Load the model
    model = StudentModel()  # !  Important : No argument
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()
    print("Model loaded successfully")