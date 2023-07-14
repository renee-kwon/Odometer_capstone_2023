import numpy as np
from PIL import Image
import os

def calculate_average_contrast(image):
    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Calculate the average pixel value
    avg_pixel = np.mean(np.array(gray_image))

    # Calculate the contrast of each pixel
    contrast = np.abs(np.array(gray_image) - avg_pixel)

    # Calculate the average contrast
    avg_contrast = np.mean(contrast)

    return avg_contrast

def process_image(file_path):
    # Open the image using PIL
    image = Image.open(file_path)

    # Calculate the average contrast
    avg_contrast = calculate_average_contrast(image)

    # Get the resolution
    resolution = image.size

    return {'file_name': os.path.basename(file_path), 'avg_contrast': avg_contrast, 'resolution': resolution}