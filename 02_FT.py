# FaceTraining.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

# Pastikan direktori penyimpanan model dan class_indices ada
os.makedirs('RTFR', exist_ok=True)

# Preprocessing dan Augmentasi Data
datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2,
    horizontal_flip=True,  # Tambahkan augmentasi untuk meningkatkan keberagaman data
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2
)

train_generator = datagen.flow_from_directory(
    'dataset/faces', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training',
    shuffle=True  # Pastikan data diacak selama pelatihan
)

validation_generator = datagen.flow_from_directory(
    'dataset/faces', 
    target_size=(150, 150), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation',
    shuffle=False
)

# Membangun model CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # dropout untuk mengurangi overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# Kompilasi model
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Menyusun callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'face_recognition_model_best.keras',
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )
]

# Melatih model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Menyimpan model akhir ke format Keras
model.save('face_recognition_model_final.keras')

# Menyimpan class indices ke dalam file JSON
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as f:
    json.dump(class_indices, f)

# Evaluasi model pada data validasi
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

# Menyimpan riwayat pelatihan ke file JSON
with open('history.json', 'w') as f:
    json.dump(history.history, f)

