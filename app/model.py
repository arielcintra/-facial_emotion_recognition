import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from app.preprocessing import preprocess_image

def build_model(img_size, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(images, labels, img_size, emotions):
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    X_train = X_train.reshape(-1, img_size, img_size, 1)
    X_test = X_test.reshape(-1, img_size, img_size, 1)
    model = build_model(img_size, len(emotions))
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save('./models/trained_model.h5')
    return model, history, X_test, y_test

def evaluate_model(model, X_test, y_test):
    model.evaluate(X_test, y_test)

def predict_sample(model, img_path, emotions, img_size):
    img = preprocess_image(img_path, img_size)
    prediction = model.predict(img)
    label = np.argmax(prediction)
    print('Predicted emotion:', emotions[label])
