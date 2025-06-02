import os
import cv2
import numpy as np
from tqdm import tqdm
import kagglehub
from kagglehub import KaggleDatasetAdapter

IMG_SIZE = 48
DATASET_NAME = "tapakah68/facial-emotion-recognition"
DATA_DIR = "data"

# Load dataset metadata
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    DATASET_NAME,
    "images/labels.csv"
)

# Create folders for each emotion class
for label in df["emotion"].unique():
    os.makedirs(os.path.join(DATA_DIR, str(label)), exist_ok=True)

# Extract and save images
for idx, row in tqdm(df.iterrows(), total=len(df)):
    label = str(row["emotion"])
    image_path = os.path.join("images", row["pixels"])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        save_path = os.path.join(DATA_DIR, label, f"{idx}.png")
        cv2.imwrite(save_path, resized)
