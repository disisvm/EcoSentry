import cv2
import os

import numpy as np


def image_enhancements(input_directory, output_directory, target_size):
    """
    Resize images in a directory and adjust contrast.

    Args:
        input_directory (str): Path to the input directory containing images.
        output_directory (str): Path to the output directory to save processed images.
        target_size (tuple): Desired dimensions (width, height) for resizing.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get list of all image files in input directory
    image_files = [f for f in os.listdir(input_directory) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Load image
        image_path = os.path.join(input_directory, image_file)
        image = cv2.imread(image_path)

        # Resize image
        resized_image = cv2.resize(image, target_size)

        # Perform noise reduction
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Perform sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Sharpening kernel
        sharpened_image = cv2.filter2D(denoised_image, -1, kernel)

        # Perform contrast adjustment
        lab = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        contrast_adjusted_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Save processed image to output directory
        output_path = os.path.join(output_directory, image_file)
        cv2.imwrite(output_path, contrast_adjusted_image)

'''
# Sample Usage
input_dir = 'input_images/'
output_dir = 'processed_images/'
target_size = (800, 600)  # Set your desired dimensions

resize_and_adjust_contrast(input_dir, output_dir, target_size)
'''