import numpy as np
import cv2

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