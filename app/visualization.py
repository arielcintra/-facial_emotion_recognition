import matplotlib.pyplot as plt
import numpy as np

def show_samples(images, labels, emotions):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(emotions[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./images/samples.png')

def plot_accuracy(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./images/accuracy_plot.png')
