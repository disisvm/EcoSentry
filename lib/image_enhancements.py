import cv2
import numpy as np

def enhance_image(image_path, target_size):
    """
    Enhance a single image.

    Args:
        image_path (str): Path to the input image file.
        target_size (tuple): Desired dimensions (width, height) for resizing.

    Returns:
        np.ndarray: Enhanced image array.
    """
    # Load image
    image = cv2.imread(image_path)

    # Resize image
    resized_image = cv2.resize(image, target_size)

    # Perform noise reduction
    denoised_image = cv2.fastNlMeansDenoisingColored(resized_image, None, 10, 10, 7, 21)

    # Perform sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
    sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

    # Perform contrast adjustment
    lab = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    contrast_adjusted_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return contrast_adjusted_image

