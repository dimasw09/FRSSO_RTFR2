from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from keras_preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'dataset/faces'
MODEL_PATH = 'face_recognition_model_best.keras'
CLASS_INDICES_PATH = 'class_indices.json'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Load model and class indices if available
model = None
labels = {}
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
if os.path.exists(CLASS_INDICES_PATH):
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
        labels = {v: k for k, v in class_indices.items()}

def save_face(image_path, label_name):
    detector = MTCNN()
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)
    label_dir = os.path.join(DATASET_FOLDER, label_name)
    os.makedirs(label_dir, exist_ok=True)

    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        face_img = img[y:y+h, x:x+w]
        face_path = os.path.join(label_dir, f"{label_name}_{i+1}.jpg")
        cv2.imwrite(face_path, face_img)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    label_name = request.form.get('label_name')
    if not label_name:
        return jsonify({"error": "Label name is missing."}), 400

    files = request.files
    if not files:
        return jsonify({"error": "No images provided."}), 400

    for key, file in files.items():
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            save_face(filepath, label_name)  # Save the face detected from the image

    return jsonify({"message": "10 images captured and saved successfully."})


@app.route('/train', methods=['POST'])
def train():
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2
    )

    train_generator = datagen.flow_from_directory(
        DATASET_FOLDER, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='categorical', 
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_FOLDER, 
        target_size=(150, 150), 
        batch_size=32, 
        class_mode='categorical', 
        subset='validation',
        shuffle=False
    )

    global model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam', 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    with open(CLASS_INDICES_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)

    return jsonify({"message": "Model trained successfully."})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded. Train the model first."}), 400

    file = request.files['image']
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (150, 150))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        predictions = model.predict(img)    
        max_index = np.argmax(predictions[0])
        confidence = predictions[0][max_index]

        if confidence < 0.6:
            return jsonify({"label": "unknown", "confidence": float(confidence)})
        else:
            label = labels[max_index]
            return jsonify({"label": label, "confidence": float(confidence)})

    return jsonify({"error": "No file provided."}), 400

if __name__ == '__main__':
    app.run(debug=True)
