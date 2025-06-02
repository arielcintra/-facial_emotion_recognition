from app.data_loader import load_data
from app.preprocessing import preprocess_image
from app.visualization import show_samples, plot_accuracy
from app.model import build_model, train_model, evaluate_model, predict_sample

import os

DATA_DIR = './data/'
IMG_SIZE = 48
EMOTIONS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

if __name__ == "__main__":
    images, labels = load_data(DATA_DIR, EMOTIONS, IMG_SIZE)
    show_samples(images, labels, EMOTIONS)

    model, history, X_test, y_test = train_model(images, labels, IMG_SIZE, EMOTIONS)
    evaluate_model(model, X_test, y_test)
    plot_accuracy(history)

    sample_path = os.path.join(DATA_DIR, "0", os.listdir(os.path.join(DATA_DIR, "0"))[0])
    predict_sample(model, sample_path, EMOTIONS, IMG_SIZE)