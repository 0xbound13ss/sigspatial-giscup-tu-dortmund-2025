import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import json


from sklearn.mixture import GaussianMixture


rnge = 20
reso = rnge * 2 + 1

steps = [
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 5),
    (1, 5),
    (0, -4),
    (0, -5),
    (0, -6),
    (0, -5),
    (1, -5),
]


def generate_heatmap(axs, coords):
    # Define the grid

    # Convert to numpy array for easier manipulation
    coords = np.array(coords)
    # Define the Gaussian function# Define the grid boundaries
    x_edges = np.linspace(-rnge, rnge, reso)  # Adjust as needed
    y_edges = np.linspace(-rnge, rnge, reso)  # Adjust as needed

    c = 0
    for coord in coords:
        if coord[0] == 0 or coord[1] == 0:
            c += 1
    # print(c)

    # Compute 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=[x_edges, y_edges]
    )

    # axs.scatter(coords[:, 0], coords[:, 1], alpha=0.01)

    # plt.figure(figsize=(8, 6))
    axs.imshow(
        heatmap.T,
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        origin="lower",
        cmap="viridis",
    )
    # axs.set_colorbar(label='Counts')
    # axs.set_xlabel('X')
    # axs.ylabel('Y')
    # plt.show()


def fit_data(axs, steps):
    data = np.array(steps)
    # Define the number of Gaussian components
    n_components = 1

    # Fit a Gaussian Mixture Model to the data
    gmm = GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(data)

    x = np.linspace(-rnge, rnge, 10 * reso)
    y = np.linspace(-rnge, rnge, 10 * reso)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    # Compute the density
    log_density = gmm.score_samples(grid)
    density = np.exp(log_density)
    Z = density.reshape(X.shape)

    # plt.figure(figsize=(8, 6))
    axs.imshow(
        Z,
        extent=(-rnge, rnge, -rnge, rnge),
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    return Z
    # axs.colorbar(label='Intensity')
    # axs.title('2D Heatmap Centered Around (0,0) with Squares')
    # axs.xlabel('X')
    # axs.ylabel('Y')
    # plt.show()


fig, axs = plt.subplots(1, 2, figsize=(12, 6))

generate_heatmap(axs[0], steps)
# generate_heatmap(axs[0,1], steps_cor)
fit_data(axs[1], steps)

plt.tight_layout()
plt.show()
