import torch
import numpy as np
from nn import gradient
from matplotlib import pyplot as plt
from matplotlib import colors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_vector_field_meshgrid(X_value, Y_value):
    # Create a grid of points
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    X, Y = np.meshgrid(x, y)

    # Define the constant vector field (e.g., constant vector [1, 1] everywhere)
    U = np.ones_like(X) * X_value  # constant x-component
    V = np.ones_like(Y) * Y_value  # constant y-component


def get_sdf_image(numpy_sdf, eps=0.005):
    numpy_sdf_max = np.ones(numpy_sdf.shape) - np.maximum(
        numpy_sdf, np.zeros(numpy_sdf.shape)
    )
    numpy_sdf_max = numpy_sdf_max - np.multiply(
        numpy_sdf_max, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
    )
    numpy_sdf_min = np.ones(numpy_sdf.shape) - np.maximum(
        -numpy_sdf, np.zeros(numpy_sdf.shape)
    )
    numpy_sdf_min = numpy_sdf_min - np.multiply(
        numpy_sdf_min, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
    )
    numpy_sdf_both = (
        np.ones(numpy_sdf.shape)
        - np.maximum(numpy_sdf, np.zeros(numpy_sdf.shape))
        - np.maximum(-numpy_sdf, np.zeros(numpy_sdf.shape))
    )
    numpy_sdf_both = numpy_sdf_both - np.multiply(
        numpy_sdf_both, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
    )
    return np.concatenate(
        [
            numpy_sdf_min[:, :, np.newaxis],
            numpy_sdf_both[:, :, np.newaxis],
            numpy_sdf_max[:, :, np.newaxis],
        ],
        axis=2,
    )


def display_sdfColor(f, resolution, time, filename=None):
    """Displays the values of the function f, evaluated over a regular grid defined between -1 and
    1 and of resolution (resolution x resolution)"""
    coords = torch.stack(
        list(
            torch.meshgrid(
                [torch.linspace(-1, 1, resolution, device=device)] * 2, indexing="xy"
            )
        ),
        dim=2,
    ).reshape(-1, 2)
    coords = torch.concat(
        (
            coords,
            time * torch.ones((coords.shape[0], 1), device=device),
        ),
        dim=1,
    )

    coords.requires_grad = True
    sdf = f(coords).reshape(resolution, resolution)
    numpy_sdf = sdf.detach().cpu().numpy()

    img_plot = get_sdf_image(numpy_sdf)

    plt.axis("off")
    plt.imshow(img_plot)
    plt.contour(numpy_sdf, 10, colors="k", linewidths=0.4, linestyles="solid")
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def display_grad_norm(f, resolution, time, filename=None):
    """Displays the norm of the gradient of the function f, evaluated at points of a regular grid
    defined between -1 and 1 and of resolution (resolution x resolution)"""

    coords = torch.stack(
        list(
            torch.meshgrid(
                [torch.linspace(-1, 1, resolution, device=device)] * 2, indexing="xy"
            )
        ),
        dim=2,
    ).reshape(-1, 2)
    coords = torch.concat(
        (
            coords,
            time * torch.ones((coords.shape[0], 1), device=device),
        ),
        dim=1,
    )

    coords.requires_grad = True

    coords.requires_grad = True
    sdf = f(coords)
    grad = (
        gradient(sdf, coords)[:, 0:2]
        .norm(dim=1)
        .detach()
        .cpu()
        .numpy()
        .reshape(resolution, resolution)
    )

    plt.axis("off")
    plt.imshow(grad, cmap="nipy_spectral", vmin=0.0, vmax=1.5)  # 0.25, 1.25
    plt.colorbar()
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def display_grad_quiver(f, resolution, time, filename=None):
    """Displays the norm of the gradient of the function f, evaluated at points of a regular grid
    defined between -1 and 1 and of resolution (resolution x resolution)"""

    coords = torch.stack(
        list(
            torch.meshgrid(
                [torch.linspace(-1, 1, resolution, device=device)] * 2, indexing="xy"
            )
        ),
        dim=2,
    ).reshape(-1, 2)
    coords = torch.concat(
        (
            coords,
            time * torch.ones((coords.shape[0], 1), device=device),
        ),
        dim=1,
    )

    coords.requires_grad = True
    sdf = f(coords)
    grad = gradient(sdf, coords)[:, 0:2].detach().cpu().numpy()
    coords = coords.detach().cpu().numpy()

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.quiver(coords[:, 0], coords[:, 1], grad[:, 0], grad[:, 1])
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def display_multi_slices(f, resolution, filename=None, figsize=(14, 5)):
    """Displays the values of the function f, evaluated over a regular grid defined between -1 and
    1 and of resolution (resolution x resolution)"""
    fig, plots = plt.subplots(2, 5, figsize=figsize)

    for time in range(10):
        i = time // 5
        j = time - i * 5
        t = 0.1 * time
        coords = torch.stack(
            list(
                torch.meshgrid(
                    [torch.linspace(-1, 1, resolution, device=device)] * 2,
                    indexing="xy",
                )
            ),
            dim=2,
        ).reshape(-1, 2)
        coords = torch.concat(
            (
                coords,
                t * torch.ones((coords.shape[0], 1), device=device),
            ),
            dim=1,
        )

        coords.requires_grad = True
        sdf = f(coords).reshape(resolution, resolution)
        numpy_sdf = sdf.detach().cpu().numpy()

        eps = 0.005
        numpy_sdf_max = np.ones(numpy_sdf.shape) - np.maximum(
            numpy_sdf, np.zeros(numpy_sdf.shape)
        )
        numpy_sdf_max = numpy_sdf_max - np.multiply(
            numpy_sdf_max, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
        )
        numpy_sdf_min = np.ones(numpy_sdf.shape) - np.maximum(
            -numpy_sdf, np.zeros(numpy_sdf.shape)
        )
        numpy_sdf_min = numpy_sdf_min - np.multiply(
            numpy_sdf_min, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
        )
        numpy_sdf_both = (
            np.ones(numpy_sdf.shape)
            - np.maximum(numpy_sdf, np.zeros(numpy_sdf.shape))
            - np.maximum(-numpy_sdf, np.zeros(numpy_sdf.shape))
        )
        numpy_sdf_both = numpy_sdf_both - np.multiply(
            numpy_sdf_both, np.multiply(numpy_sdf <= eps, numpy_sdf >= -eps)
        )

        plots[i, j].axis("off")
        plots[i, j].imshow(
            np.concatenate(
                [
                    numpy_sdf_min[:, :, np.newaxis],
                    numpy_sdf_both[:, :, np.newaxis],
                    numpy_sdf_max[:, :, np.newaxis],
                ],
                axis=2,
            )
        )
        plots[i, j].contour(
            numpy_sdf, 10, colors="k", linewidths=0.4, linestyles="solid"
        )

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()
