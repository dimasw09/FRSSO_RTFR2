import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# Muat model yang telah dilatih
model = tf.keras.models.load_model('face_recognition_model.h5')

# Muat class indices dari file JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Membalik dictionary untuk mendapatkan label berdasarkan index
labels = {v: k for k, v in class_indices.items()}

# Fungsi untuk memprediksi wajah dari input gambar
def predict_face(img, threshold=0.6):  # Threshold yang lebih tinggi untuk meningkatkan akurasi
    img = cv2.resize(img, (150, 150))  # Resize gambar sesuai ukuran input model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    predictions = model.predict(img)
    max_index = np.argmax(predictions[0])
    confidence = predictions[0][max_index]
    
    if confidence < threshold:
        return "unknown", confidence
    else:
        label = labels[max_index]
        return label, confidence

# Fungsi untuk mendeteksi dan mengenali wajah dalam gambar dari file path
def recognize_face_from_image(image_path, threshold=0.6):
    img = cv2.imread(image_path)
    
    # Muat detektor wajah yang lebih canggih, seperti MTCNN atau dlib
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Deteksi wajah dalam gambar
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        label, confidence = predict_face(face, threshold)
        
        # Menampilkan label prediksi dan confidence pada gambar
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f'{label} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Tampilkan gambar dengan hasil prediksi
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path ke gambar yang ingin dikenali
image_path = 'WIN_20240826_14_39_03_Pro.jpg'

# Panggil fungsi untuk mengenali wajah
recognize_face_from_image(image_path)
