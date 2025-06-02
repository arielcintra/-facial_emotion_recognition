import os
import numpy as np
import cv2

def load_data(data_dir, emotions, img_size):
    images, labels = [], []
    for label, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, str(label))
        if os.path.isdir(emotion_dir):
            for file in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized = cv2.resize(img, (img_size, img_size))
                images.append(resized)
                labels.append(label)
    return np.array(images), np.array(labels)
