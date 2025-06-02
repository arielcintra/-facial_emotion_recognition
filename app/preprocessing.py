import cv2
import numpy as np

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, img_size, img_size, 1)
    return reshaped
