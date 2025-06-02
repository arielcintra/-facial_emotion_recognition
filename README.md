# Facial Emotion Recognition with CNN 😄😠😢😲

This project uses a Convolutional Neural Network (CNN) to recognize human facial emotions from grayscale images. It is based on the [FER2013-like dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition) and supports 8 emotion classes.

## 🎯 Emotion Classes

- 0: Anger
- 1: Contempt
- 2: Disgust
- 3: Fear
- 4: Happy
- 5: Neutral
- 6: Sad
- 7: Surprised

## 🧠 Features

- Image preprocessing and data loading
- Modular CNN model definition and training
- Accuracy tracking and visualization
- Sample prediction
- Dataset extraction from KaggleHub

## 📁 Project Structure

```
facial_emotion_recognition/
├── app/
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── visualization.py
│ ├── model.py
│ └── init.py
├── data/ # Created after running extract script
│ ├── 0/, ..., 7/
├── images/
│ ├── samples.png
│ └── accuracy_plot.png
├── models/
│ └── trained_model.h5
├── extract_and_prepare_data.py
├── facial_emotion_recognition.py
├── requirements.txt
└── README.md
```

## 📥 Dataset Preparation

To automatically download and prepare the dataset, run:

```bash
python extract_and_prepare_data.py
```
This will download the dataset using KaggleHub and organize it into folders from data/0 to data/7.

## 🚀 How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```
2. Train the model and evaluate:

```bash
python facial_emotion_recognition.py
```

This will:
- Train the CNN for 20 epochs
- Save the trained model to models/trained_model.h5
- Generate the following outputs:
    - images/samples.png: Sample grid of 25 training images
    - images/accuracy_plot.png: Accuracy plot over training epochs

## 📈 Sample Outputs
Sample Images:
Accuracy Plot:

## 🧪 Model Evaluation
The final model achieved good classification accuracy.
Next steps will be further tune the architecture and add real-time webcam prediction using OpenCV.
