import numpy as np
import matplotlib.pyplot as plt
import cv2
from lib.algo import convolutional_wasserstein_barycenter
from utils.shapes_2d import create_shape

# Define grid size and gamma for heat kernel
image_shape = (128, 128)
gamma = 0.001  # Regularization parameter
entropy_reg = 0.1
interp_num = 5

# Define the 4 corner shapes
f1 = create_shape("circle", image_shape) # Will be at Top-left
f2 = create_shape("parentheses", image_shape)  # Will be at Bottom-left
f3 = create_shape("plus", image_shape)  # Will be at Top-right
f4 = create_shape("star", image_shape)  # Will be at Bottom-right

# Stack the shapes into a single array
A = np.array([f1, f2, f3, f4])

# Define interpolation grid
interpolated_shapes = np.zeros((interp_num, interp_num, *image_shape))

# Perform structured interpolation exactly as in your logic
fig, axes = plt.subplots(interp_num, interp_num, figsize=(10, 10))

v1 = np.array((1, 0, 0, 0))
v2 = np.array((0, 1, 0, 0))
v3 = np.array((0, 0, 1, 0))
v4 = np.array((0, 0, 0, 1))


# Loop over the grid with bilinear interpolation
for i in range(interp_num):
    for j in range(interp_num):
        tx = float(i) / (interp_num - 1)
        ty = float(j) / (interp_num - 1)

        # Compute weights using bilinear interpolation
        tmp1 = (1 - tx) * v1 + tx * v2
        tmp2 = (1 - tx) * v3 + tx * v4
        weights = (1 - ty) * tmp1 + ty * tmp2
        
        print(weights)

        # Assign corners directly
        if i == 0 and j == 0:
            axes[i, j].imshow(f1, cmap="gray")
        elif i == 0 and j == (interp_num - 1):
            axes[i, j].imshow(f3, cmap="gray")
        elif i == (interp_num - 1) and j == 0:
            axes[i, j].imshow(f2, cmap="gray")
        elif i == (interp_num - 1) and j == (interp_num - 1):
            axes[i, j].imshow(f4, cmap="gray")
        else:
            # Compute barycenter for intermediate shapes
            mu_interp, _, _ = convolutional_wasserstein_barycenter(A, gamma, weights, stop_threshold=1e-5, verbose=True)
            axes[i, j].imshow(mu_interp.reshape(image_shape), cmap="gray")

        axes[i, j].axis("off")

plt.suptitle("Wasserstein Interpolated Shapes Grid (5x5)", fontsize=16)
plt.tight_layout()
plt.savefig("shapes_grid.png")