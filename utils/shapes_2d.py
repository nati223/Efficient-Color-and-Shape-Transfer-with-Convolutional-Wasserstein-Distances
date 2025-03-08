import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.algorithms import convolutional_wasserstein_barycenter
from utils.bilateral_shape_interpolation import cross_bilateral_wasserstein_barycenter

def create_shape(shape_type, image_shape):
    img = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape

    if shape_type == "circle":
        cv2.circle(img, (w//2, h//2), w//4, 1, -1)  # Filled circle
    elif shape_type == "plus":
        img[h//4:3*h//4, w//2-5:w//2+5] = 1
        img[h//2-5:h//2+5, w//4:3*w//4] = 1
    elif shape_type == "star":
        center = (w // 2, h // 2)
        radius = w // 4
        points = []
        for i in range(10):  # 10 points (5 inner, 5 outer)
            angle = np.pi / 5 * i
            r = radius if i % 2 == 0 else radius // 2.5
            x = int(center[0] + np.cos(angle) * r)
            y = int(center[1] - np.sin(angle) * r)
            points.append((x, y))
        points = np.array(points, np.int32)
        cv2.fillPoly(img, [points], 1)
    elif shape_type == "parentheses":
        y, x = np.ogrid[:h, :w]

        # Define center positions along the main diagonal
        center1 = (w // 4, h // 4)  # Top-left quarter
        center2 = (3 * w // 4, 3 * h // 4)  # Bottom-right quarter
        radius = w // 6  # Define radius for the circular shape

        # Create circular masks at the diagonal positions
        mask1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius ** 2
        mask2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius ** 2
        img[mask1 | mask2] = 1

    return img.astype(float) / img.sum()  # Normalize to probability distribution

def create_plot_4x4_shapes_grid(A, image_shape, gamma=0.004, sharpen_entropy = False,
                                entropy_reg = None, cwd=True, save_path=None, verbose=False):
    
    interp_num = 5

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
            
            if verbose:
                print(weights)

            # Assign corners directly
            if i == 0 and j == 0:
                axes[i, j].imshow(A[0], cmap="gray")
            elif i == 0 and j == (interp_num - 1):
                axes[i, j].imshow(A[2], cmap="gray")
            elif i == (interp_num - 1) and j == 0:
                axes[i, j].imshow(A[1], cmap="gray")
            elif i == (interp_num - 1) and j == (interp_num - 1):
                axes[i, j].imshow(A[3], cmap="gray")
            else:
                # Compute barycenter for intermediate shapes
                if cwd:
                    mu_interp, _, _, _ = convolutional_wasserstein_barycenter(A, gamma, weights, stop_threshold=1e-5, verbose=verbose, entropy_sharpening=sharpen_entropy, max_iterations=2000, H0=entropy_reg)
                else:
                    mu_interp = cross_bilateral_wasserstein_barycenter(A, weights, verbose=verbose)
                axes[i, j].imshow(mu_interp.reshape(image_shape), cmap="gray")

            axes[i, j].axis("off")
        
    plt.suptitle("Wasserstein Interpolated Shapes Grid (5x5)", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

