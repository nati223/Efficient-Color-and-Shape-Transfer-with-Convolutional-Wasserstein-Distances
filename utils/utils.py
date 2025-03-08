import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

def plot_2d_histogram(hist):
    
    plt.imshow(hist, interpolation='nearest', cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title('2D Histogram')
    plt.xlabel('A axis')
    plt.ylabel('B axis')
    plt.show()

def compute_gamma(mu_list, percentile=1):
    """
    Computes gamma as a small percentage of the median transport cost
    by estimating pairwise squared distances between distributions.

    Parameters:
      mu_list   : List of probability distributions (each flattened 1D).
      percentile: Percentage of the median transport cost (default 1%).

    Returns:
      gamma     : Optimal entropy regularization parameter.
    """
    # Get bin positions in a 2D grid
    n_bins = int(np.sqrt(len(mu_list[0])))  # Assume square grid
    x, y = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
    positions = np.vstack([x.ravel(), y.ravel()]).T  # Flattened 2D positions

    # Compute pairwise squared distances
    pairwise_dists = cdist(positions, positions, metric='sqeuclidean')

    # Compute transport cost using mass-weighted distances
    transport_costs = []
    for mu in mu_list:
        for mu_target in mu_list:
            cost = np.sum(mu[:, None] * pairwise_dists * mu_target[None, :])  # Transport cost
            transport_costs.append(cost)

    # Compute the median cost
    median_cost = np.median(transport_costs)

    # Set gamma as 1% of the median cost
    gamma = (percentile / 100) * median_cost
    return gamma