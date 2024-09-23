# FaceDataset.py
import cv2
import os

# Fungsi untuk mengambil gambar wajah dan menyimpannya dengan label tertentu
def capture_faces(label_name):
    # Membuat direktori untuk label jika belum ada
    dataset_dir = "dataset/faces"
    label_dir = os.path.join(dataset_dir, label_name)
    
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Menginisialisasi kamera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mengubah ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y+h, x:x+w]
            
            # Menyimpan gambar wajah dengan label
            face_filename = os.path.join(label_dir, f"{label_name}_{count}.jpg")
            cv2.imwrite(face_filename, face)
            
            # Menampilkan kotak wajah pada frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('Capturing Faces', frame)

        # Berhenti jika tombol 'q' ditekan atau setelah 100 gambar
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()

label_name = "Dimas Wahyudi"  
capture_faces(label_name)
