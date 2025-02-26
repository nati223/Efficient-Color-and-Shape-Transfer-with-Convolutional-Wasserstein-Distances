import numpy as np
import matplotlib.pyplot as plt
from lib.algorithms import *


# if __name__ == "__main__":
    
#     # Define image parameters
#     size = 128  # Create 128x128 images
#     image_shape = (size, size)
    
#     # Create two synthetic distributions (for instance, a circle and a square)
#     def create_circle(size, radius, center=None):
#         if center is None:
#             center = (size // 2, size // 2)
#         Y, X = np.ogrid[:size, :size]
#         distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#         circle = (distance <= radius).astype(np.float64)
#         return circle

#     def create_square(size, square_size, top_left=None):
#         square = np.zeros((size, size), dtype=np.float64)
#         if top_left is None:
#             start = size // 2 - square_size // 2
#         else:
#             start = top_left[0]
#         square[start:start+square_size, start:start+square_size] = 1.0
#         return square

#     # Create images and normalize them to get probability distributions
#     circle_img = create_circle(size, radius=30)
#     square_img = create_square(size, square_size=50)
    
#     mu0 = circle_img / np.sum(circle_img)
#     mu1 = square_img / np.sum(square_img)
    
#     # Visualize the distributions
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(mu0, cmap='gray')
#     plt.title("Circle Distribution")
#     plt.axis("off")
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(mu1, cmap='gray')
#     plt.title("Square Distribution")
#     plt.axis("off")
#     plt.show()
    
#     # Flatten the distributions for processing
#     mu0_flat = mu0.flatten()
#     mu1_flat = mu1.flatten()
    
#     # Set parameters for the kernel and regularization.
#     sigma = 2.0    # Controls the diffusion in the heat kernel approximation.
#     gamma = 0.1    # Regularization parameter.
    
#     # Run the Sinkhorn iterations
#     v, w, a = sinkhorn(mu0_flat, mu1_flat, image_shape, sigma, tol=1e-6, max_iter=1000)
    
#     # Compute the approximate regularized Wasserstein distance
#     W2_squared = compute_convolutional_distance(mu0_flat, mu1_flat, v, w, a, gamma)
#     print("Approximate Regularized Wasserstein Distance (squared):", W2_squared)

#==================================================
#==================================================
#==================================================

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     def create_circle(size, radius, center=None):
#         """Creates a binary image with a filled circle."""
#         if center is None:
#             center = (size // 2, size // 2)
#         Y, X = np.ogrid[:size, :size]
#         dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#         return (dist <= radius).astype(np.float64)

#     def create_square(size, square_size, top_left=None):
#         """Creates a binary image with a filled square."""
#         img = np.zeros((size, size), dtype=np.float64)
#         if top_left is None:
#             start = size // 2 - square_size // 2
#         else:
#             start = top_left[0]
#         img[start:start+square_size, start:start+square_size] = 1.0
#         return img

#     # Parameters
#     size = 128
#     image_shape = (size, size)
#     sigma = 1.0  # Gaussian kernel std for heat kernel approximation
    
#     # Create a few synthetic distributions.
#     circle = create_circle(size, radius=30)
#     square = create_square(size, square_size=50)
#     triangle = np.tri(size, size, k=size//4).astype(np.float64)  # a simple triangular pattern

#     # Normalize each image so they sum to 1
#     def normalize(img):
#         return img / np.sum(img)
    
#     mu0 = normalize(circle).flatten()
#     mu1 = normalize(square).flatten()
#     mu2 = normalize(triangle).flatten()
    
#     # Choose barycenter weights, e.g., equal weights.
#     alpha_list = [1/3, 1/3, 1/3]
#     mu_list = [mu0, mu1, mu2]
    
#     # Compute the barycenter
#     mu_bar, v_list, w_list = wasserstein_barycenter(mu_list, alpha_list, image_shape, sigma, tol=1e-6, max_iter=200)
    
#     # Reshape for visualization
#     mu_bar_img = mu_bar.reshape(image_shape)
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 4, 1)
#     plt.imshow(circle, cmap='gray')
#     plt.title("Circle")
#     plt.axis("off")
    
#     plt.subplot(1, 4, 2)
#     plt.imshow(square, cmap='gray')
#     plt.title("Square")
#     plt.axis("off")
    
#     plt.subplot(1, 4, 3)
#     plt.imshow(triangle, cmap='gray')
#     plt.title("Triangle")
#     plt.axis("off")
    
#     plt.subplot(1, 4, 4)
#     plt.imshow(mu_bar_img, cmap='gray')
#     plt.title("Wasserstein Barycenter")
#     plt.axis("off")
    
#     plt.show()
    
#==================================================
#==================================================
#==================================================

# if __name__ == "__main__":
#     # Example: Create two synthetic distributions (e.g., circle and square)
#     size = 128
#     image_shape = (size, size)
#     sigma = 2.0  # Gaussian standard deviation for heat kernel approximation

#     def create_circle(size, radius, center=None):
#         if center is None:
#             center = (size // 2, size // 2)
#         Y, X = np.ogrid[:size, :size]
#         distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#         circle = (distance <= radius).astype(np.float64)
#         return circle

#     def create_square(size, square_size, top_left=None):
#         square = np.zeros((size, size), dtype=np.float64)
#         if top_left is None:
#             start = size // 2 - square_size // 2
#         else:
#             start = top_left[0]
#         square[start:start+square_size, start:start+square_size] = 1.0
#         return square

#     def normalize(img):
#         return img / np.sum(img)

#     # Create two images: one with a circle, one with a square.
#     circle_img = create_circle(size, radius=30)
#     square_img = create_square(size, square_size=50)

#     # Normalize and flatten the images to obtain probability distributions.
#     mu0 = normalize(circle_img).flatten()
#     mu1 = normalize(square_img).flatten()

#     # Define a set of t values for the interpolation.
#     t_values = np.linspace(0, 1, 6)  # e.g., 0, 0.2, 0.4, 0.6, 0.8, 1
#     mu_t_list = []

#     for t in t_values:
#         print(f"\nComputing displacement interpolation for t = {t:.2f}")
#         mu_t = displacement_interpolation(mu0, mu1, t, image_shape, sigma, tol=1e-6, max_iter=200, verbose=True)
#         mu_t_list.append(mu_t.reshape(image_shape))

#     # Plot the original distributions and the interpolated ones.
#     plt.figure(figsize=(15, 4))
    
#     plt.subplot(1, len(t_values) + 2, 1)
#     plt.imshow(mu0.reshape(image_shape), cmap='gray')
#     plt.title("mu0 (Circle)")
#     plt.axis("off")
    
#     plt.subplot(1, len(t_values) + 2, len(t_values) + 2)
#     plt.imshow(mu1.reshape(image_shape), cmap='gray')
#     plt.title("mu1 (Square)")
#     plt.axis("off")
    
#     for idx, (t, mu_t_img) in enumerate(zip(t_values, mu_t_list), start=2):
#         plt.subplot(1, len(t_values) + 2, idx)
#         plt.imshow(mu_t_img, cmap='gray')
#         plt.title(f"t = {t:.2f}")
#         plt.axis("off")
    
#     plt.tight_layout()
#     plt.show()

#==================================================
#==================================================
#==================================================

# # === Example Usage ===
# if __name__ == "__main__":
#     # For demonstration, we use a simple graph with 3 vertices.
#     # Let vertices 0 and 2 be fixed with given distributions and vertex 1 be unknown.
#     vertices = [0, 1, 2]
#     # Define edges: (0, 1) and (1, 2) with weight 1.
#     edges = [(0, 1, 1.0), (1, 2, 1.0)]
    
#     size = 64  # Using a smaller image for faster computation
#     image_shape = (size, size)
#     sigma = 2.0
    
#     # Create fixed distributions for vertex 0 and vertex 2.
#     def create_circle(size, radius, center=None):
#         if center is None:
#             center = (size//2, size//2)
#         Y, X = np.ogrid[:size, :size]
#         dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#         return (dist <= radius).astype(np.float64)
    
#     def create_square(size, square_size):
#         img = np.zeros((size, size), dtype=np.float64)
#         start = size//2 - square_size//2
#         img[start:start+square_size, start:start+square_size] = 1.0
#         return img
    
#     def normalize(img):
#         return img / np.sum(img)
    
#     circle_img = create_circle(size, radius=15)
#     square_img = create_square(size, square_size=20)
    
#     mu_fixed = {
#         0: normalize(circle_img).flatten(),
#         2: normalize(square_img).flatten()
#     }
    
#     fixed = [0, 2]
    
#     # Run the Wasserstein propagation.
#     mu = wasserstein_propagation(vertices, edges, fixed, mu_fixed, image_shape, sigma, tol=1e-6, max_iter=100, verbose=True)
    
#     # Visualize the propagated distributions.
#     plt.figure(figsize=(12,4))
#     for v in vertices:
#         plt.subplot(1, len(vertices), v+1)
#         plt.imshow(mu[v].reshape(image_shape), cmap='gray')
#         plt.title(f"Vertex {v}")
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()


# === Example Usage for Soft Maps ===
# if __name__ == "__main__":
#     # Define a simple source graph.
#     # Suppose we have 3 vertices in the source (M0). Vertices 0 and 2 are fixed (with given soft correspondences),
#     # and vertex 1 is unknown.
#     vertices = [0, 1, 2]
#     # Edges: (0, 1) and (1, 2) with weight 1.
#     edges = [(0, 1, 1.0), (1, 2, 1.0)]
    
#     # Define target domain: here a 64x64 grid.
#     size = 64
#     image_shape = (size, size)
#     sigma = 2.0
    
#     # Create two fixed soft correspondences for vertices 0 and 2.
#     def create_circle(size, radius, center=None):
#         if center is None:
#             center = (size//2, size//2)
#         Y, X = np.ogrid[:size, :size]
#         dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
#         return (dist <= radius).astype(np.float64)
    
#     def create_square(size, square_size):
#         img = np.zeros((size, size), dtype=np.float64)
#         start = size//2 - square_size//2
#         img[start:start+square_size, start:start+square_size] = 1.0
#         return img
    
#     def normalize(img):
#         return img / np.sum(img)
    
#     circle_img = create_circle(size, radius=15)
#     square_img = create_square(size, square_size=20)
    
#     mu_fixed = {
#         0: normalize(circle_img).flatten(),
#         2: normalize(square_img).flatten()
#     }
    
#     fixed = [0, 2]
    
#     # For each vertex in the source graph, we also define a compatibility function.
#     # Here we simply create a synthetic compatibility vector. For example, for vertex 0 we may want target pixels near the center to be more compatible,
#     # while for vertex 2 the compatibility may favor pixels near the boundary.
#     # In practice, these would be derived from geometric descriptors.
#     def create_compatibility(size, mode='center'):
#         Y, X = np.indices((size, size))
#         if mode == 'center':
#             cx, cy = size/2, size/2
#             comp = np.sqrt((X - cx)**2 + (Y - cy)**2)
#         elif mode == 'edge':
#             comp = np.maximum(X, Y)
#         else:
#             comp = np.ones((size, size))
#         # Normalize the compatibility so that smaller values indicate higher compatibility.
#         comp = comp - comp.min()
#         comp = comp / (comp.max() + 1e-16)
#         return comp.flatten()
    
#     compatibility = {
#         0: create_compatibility(size, mode='center'),
#         1: create_compatibility(size, mode='uniform'),  # for unknown vertex, use uniform compatibility
#         2: create_compatibility(size, mode='edge')
#     }
    
#     # Set parameters for the soft maps penalty.
#     tau = 0.5
#     gamma = 0.1
    
#     # Run the soft maps propagation.
#     mu_soft = soft_maps_propagation(vertices, edges, fixed, mu_fixed, 
#                                     compatibility, tau, gamma, image_shape, sigma, 
#                                     tol=1e-6, max_iter=100, verbose=True)
    
#     # Visualize the soft maps: distributions on the target domain for each source vertex.
#     plt.figure(figsize=(12,4))
#     for v in vertices:
#         plt.subplot(1, len(vertices), v+1)
#         plt.imshow(mu_soft[v].reshape(image_shape), cmap='gray')
#         plt.title(f"Vertex {v}")
#         plt.axis("off")
#     plt.tight_layout()
#     plt.show()