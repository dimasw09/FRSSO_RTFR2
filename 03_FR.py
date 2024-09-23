# FaceRecognition.py
from mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import json

model = tf.keras.models.load_model('face_recognition_model_best.keras')

# Muat class indices dari file JSON
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Membalik dictionary untuk mendapatkan label berdasarkan index
labels = {v: k for k, v in class_indices.items()}

# Inisialisasi MTCNN
detector = MTCNN()

# Membuat direktori untuk menyimpan hasil prediksi jika belum ada
output_dir = "predict"
os.makedirs(output_dir, exist_ok=True)

# Fungsi untuk memprediksi wajah dari input gambar
def predict_face(img, threshold=0.85):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert ke grayscale
    img = cv2.resize(img, (150, 150))  # Resize gambar sesuai ukuran input model
    img = np.stack((img,) * 3, axis=-1)  # Duplicate grayscale channel to 3 channels
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

# Fungsi untuk memproses gambar dari kamera
def process_from_camera():
    cap = cv2.VideoCapture(0)
    img_count = 0  # Counter untuk nama file
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi wajah menggunakan MTCNN
        faces = detector.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face['box']
            face_img = frame[y:y+h, x:x+w]
            label, confidence = predict_face(face_img, threshold=0.6)
            
            # Menyimpan gambar hasil prediksi
            img_count += 1
            img_filename = os.path.join(output_dir, f"{label}_{img_count}.jpg")
            cv2.imwrite(img_filename, face_img)
            
            # Menampilkan label prediksi dan confidence pada gambar
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f'{label} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)

        # Berhenti jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Fungsi untuk memproses gambar dari direktori
def process_from_directory(directory):
    img_count = 0
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path)
        faces = detector.detect_faces(img)

        for face in faces:
            x, y, w, h = face['box']
            face_img = img[y:y+h, x:x+w]
            label, confidence = predict_face(face_img, threshold=0.6)
            
            # Menyimpan gambar hasil prediksi
            img_count += 1
            img_filename = os.path.join(output_dir, f"{label}_{img_count}.jpg")
            cv2.imwrite(img_filename, face_img)
            
            # Menampilkan label prediksi dan confidence pada gambar
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f'{label} ({confidence*100:.2f}%)', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# Menu untuk memilih mode
def main():
    print("Pilih mode:")
    print("1. Realtime Camera")
    print("2. Pilih Gambar dari Directory")
    choice = input("Masukkan pilihan (1/2): ")

    if choice == '1':
        process_from_camera()
    elif choice == '2':
        directory = input("Masukkan path ke directory gambar: ")
        process_from_directory(directory)
    else:
        print("Pilihan tidak valid. Harap coba lagi.")

if __name__ == "__main__":
    main()
