import torch
import torch_geometric


def image_to_graph(
    image: torch.Tensor, conv2d: torch.nn.Conv2d  = None
) -> torch_geometric.data.Data:
    """
    Converts an image tensor to a PyTorch Geometric Data object.
    Arguments:
    ----------
    image : torch.Tensor
        Image tensor of shape (C, H, W).
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None.
        Used to determine the size of the receptive field.

    Returns:
    --------
    torch_geometric.data.Data
        Graph representation of the image.
    """
    # Assumptions (as per the docstring)
    assert image.dim() == 3, f"Expected 3D tensor, got {image.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    C, H, W = image.shape

    # flatten the image
    x = image.view(C, H * W).t().contiguous()

    # Create edges and edge attributes.
    offsets = [(dy, dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]]

    edges = []
    edge_attrs = []

    for i in range(H):
        for j in range(W):
            center_index = i * W + j
            for dy, dx in offsets:
                ni, nj = i + dy, j + dx
                # Check that the neighbor is within image bounds.
                if 0 <= ni < H and 0 <= nj < W:
                    neighbor_index = ni * W + nj
                    # In message passing, messages are passed from neighbor (source) to center (target)
                    edges.append((neighbor_index, center_index))
                    edge_attrs.append([dy, dx])

    # Convert lists to tensors.
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape [2, num_edges]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)  # shape [num_edges, 2]
    data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
def graph_to_image(
    data: torch.Tensor, height: int, width: int, conv2d: torch.nn.Conv2d = None
) -> torch.Tensor:
    """
    Converts a graph representation of an image to an image tensor.

    Arguments:
    ----------
    data : torch.Tensor
        Graph data representation of the image, expected to be of shape (H*W, C).
    height : int
        Height of the image.
    width : int
        Width of the image.
    conv2d : torch.nn.Conv2d, optional
        Conv2d layer to simulate, by default None.
        Its hyper-parameters are verified but not used in the conversion.

    Returns:
    --------
    torch.Tensor
        Image tensor of shape (C, H, W).
    """
    # Assumptions (as per the docstring)
    assert data.dim() == 2, f"Expected 2D tensor, got {data.dim()}D tensor."
    if conv2d is not None:
        assert conv2d.padding[0] == conv2d.padding[1] == 1, "Expected padding of 1 on both sides."
        assert conv2d.kernel_size[0] == conv2d.kernel_size[1] == 3, "Expected kernel size of 3x3."
        assert conv2d.stride[0] == conv2d.stride[1] == 1, "Expected stride of 1."

    num_nodes, C = data.shape
    assert num_nodes == height * width, (
        f"Mismatch between graph nodes ({num_nodes}) and image dimensions (height x width = {height * width})."
    )

    # the passage from graph to image is just a reshape
    image = data.view(height, width, C).permute(2, 0, 1).contiguous()
    return image

class Conv2dMessagePassing(torch_geometric.nn.MessagePassing):
    """
    A Message Passing layer that simulates a given Conv2d layer.
    """

    def __init__(self, conv2d: torch.nn.Conv2d):
        # We assume conv2d is 3x3 with padding 1 and stride 1.
        # Use "add" aggregation to sum the messages (like a convolution).
        super().__init__(aggr='add')
        self.conv2d = conv2d
        # set the conv2d wieght
        self.weight = conv2d.weight
    def forward(self, data):
        self.edge_index = data.edge_index

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Computes the message to be passed for each edge.
        For each edge e = (u, v) in the graph indexed by i,
        the message trough the edge e (ie from node u to node v)
        should be returned as the i-th line of the output tensor.
        (The message is phi(u, v, e) in the formalism.)
        To do this you can access the features of the source node
        in x_j[i] and the attributes of the edge in edge_attr[i].

        Arguments:
        ----------
        x_j : torch.Tensor
            The features of the souce node for each edge (of size E x in_channels).
        edge_attr : torch.Tensor
            The attributes of the edge (of size E x edge_attr_dim).

        Returns:
        --------
        torch.Tensor
            The message to be passed for each edge (of size COMPLETE)
        """
        # get the corresponding index

        index_y = (edge_attr[:, 0] + 1).long()  
        index_x = (edge_attr[:, 1] + 1).long()  

        # get the corresponding weight
        W = self.weight[:, :, index_y, index_x]
        W = W.permute(2, 0, 1)

        # x_j has shape (E, in_channels). For each edge, we perform:
        #   message[i] = W[i] @ x_j[i]

        message = torch.bmm(W, x_j.unsqueeze(-1)).squeeze(-1)  # shape: (E, out_channels)
        return message
    


